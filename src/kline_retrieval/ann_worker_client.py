"""Client-side protocol for a FAISS search process isolated from PyTorch."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .retrieval import RetrievalSample, SampleEmbeddings, SimilarityMatch


ANN_RETRIEVAL_ENGINE = "faiss-ann-temporal-global-exact-rerank-v3"
LEGACY_ANN_RETRIEVAL_ENGINES = (
    "faiss-ann-global-exact-rerank-v2",
    "faiss-ann-exact-rerank-v1",
)


@dataclass(frozen=True, slots=True)
class ANNManifestMetadata:
    path: Path
    model: str
    model_version: str
    version_id: str
    source_hash: str
    dimension: int
    checkpoint_sha256: str | None
    symbols: tuple[str, ...]
    sample_count: int


def read_ann_manifest_metadata(path: str | Path) -> ANNManifestMetadata:
    """Read ANN identity and source-shard coverage without importing FAISS."""

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "kline-ann-manifest-v1":
        raise ValueError("ANN 清单格式不受支持")
    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError("ANN 清单没有分片")

    symbols: set[str] = set()
    sample_count = 0
    source_checkpoint: str | None = None
    for shard in shards:
        source = _resolve_path(manifest_path, str(shard["source"]))
        with sqlite3.connect(source) as connection:
            row = connection.execute(
                "SELECT source_hash, config_json, symbols_json, model, dimension, sample_count "
                "FROM feature_versions WHERE status='ready' "
                "ORDER BY updated_at_utc DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError(f"ANN 源分片没有 ready 版本: {source}")
        shard_source_hash, config_json, symbols_json, model, dimension, shard_count = row
        if str(shard_source_hash) != str(manifest["source_hash"]):
            raise ValueError("ANN 源分片行情快照与清单不一致")
        if str(model) != str(manifest["model"]) or int(dimension) != int(manifest["dimension"]):
            raise ValueError("ANN 源分片模型或向量维度与清单不一致")
        config = json.loads(config_json)
        checkpoint = config.get("checkpoint_sha256")
        if source_checkpoint is not None and checkpoint != source_checkpoint:
            raise ValueError("ANN 源分片绑定了不同的 CNN 检查点")
        source_checkpoint = checkpoint or source_checkpoint
        symbols.update(json.loads(symbols_json))
        sample_count += int(shard_count)

    manifest_checkpoint = manifest.get("checkpoint_sha256")
    if manifest_checkpoint is not None and source_checkpoint != manifest_checkpoint:
        raise ValueError("ANN 清单的 CNN 检查点哈希与源分片不一致")
    return ANNManifestMetadata(
        path=manifest_path,
        model=str(manifest["model"]),
        model_version=str(manifest.get("model_version", manifest["model"])),
        version_id=str(manifest["version_id"]),
        source_hash=str(manifest["source_hash"]),
        dimension=int(manifest["dimension"]),
        checkpoint_sha256=source_checkpoint or manifest_checkpoint,
        symbols=tuple(sorted(symbols)),
        sample_count=sample_count,
    )


class ANNWorkerClient:
    """Keep one FAISS-only child process alive for many CNN queries."""

    def __init__(
        self,
        database: str | Path,
        ann_index: str | Path,
        *,
        candidate_k: int,
        nprobe: int,
        price_basis: str,
    ) -> None:
        self._next_id = 1
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kline_retrieval.ann_search_worker",
                "--database",
                str(database),
                "--ann-index",
                str(ann_index),
                "--candidate-k",
                str(candidate_k),
                "--nprobe",
                str(nprobe),
                "--price-basis",
                price_basis,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_response()
        if ready.get("type") != "ready":
            self.close()
            raise RuntimeError(f"ANN worker 启动失败: {ready.get('error', ready)}")

    def __enter__(self) -> "ANNWorkerClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def search(
        self,
        query: RetrievalSample,
        *,
        month_weight: float,
        week_weight: float,
        top_k: int,
        embargo_trading_days: int,
    ) -> tuple[list[SimilarityMatch], dict[str, int | float]]:
        request_id = self._next_id
        self._next_id += 1
        self._write(
            {
                "type": "search",
                "id": request_id,
                "symbol": query.symbol,
                "date": query.date.isoformat(),
                "embeddings": {
                    "monthly": query.embeddings.monthly,
                    "weekly": query.embeddings.weekly,
                    "daily": query.embeddings.daily,
                },
                "month_weight": month_weight,
                "week_weight": week_weight,
                "top_k": top_k,
                "embargo_trading_days": embargo_trading_days,
            }
        )
        response = self._read_response()
        if response.get("id") != request_id:
            raise RuntimeError("ANN worker 返回了不匹配的请求编号")
        if not response.get("ok"):
            raise RuntimeError(f"ANN worker 检索失败: {response.get('error', 'unknown error')}")
        return _decode_matches(response.get("matches", [])), dict(response.get("diagnostics", {}))

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._write({"type": "close"})
            process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _write(self, payload: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ANN worker 标准输入不可用")
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _read_response(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("ANN worker 标准输出不可用")
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"ANN worker 意外退出，退出码 {code}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ANN worker 返回了无效 JSON: {line[:200]}") from exc


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_matches(rows: list[dict[str, Any]]) -> list[SimilarityMatch]:
    empty = SampleEmbeddings((), (), ())
    return [
        SimilarityMatch(
            rank=int(row["rank"]),
            sample=RetrievalSample(
                str(row["symbol"]), date.fromisoformat(str(row["date"])), empty
            ),
            distance=float(row["distance"]),
            monthly_distance=float(row["monthly_distance"]),
            weekly_distance=float(row["weekly_distance"]),
            daily_distance=float(row["daily_distance"]),
        )
        for row in rows
    ]


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()
