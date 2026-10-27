"""Build resumable CNN feature shards aligned with the ColProj universe batches."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sqlite3
from time import perf_counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

from .charts import build_query_charts
from .cnn import CNNEmbedder
from .config import ProjectConfig
from .market_data import MarketData
from .offline_index import INDEX_SCHEMA, _index_config, _pack_vector


def build_cnn_shard(
    database_path: str | Path,
    colproj_shard_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    config: ProjectConfig,
    batch_size: int = 256,
    device: str = "auto",
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    checkpoint = Path(checkpoint_path)
    checkpoint_hash = _file_hash(checkpoint)
    embedder = CNNEmbedder(checkpoint, device=device)

    with sqlite3.connect(colproj_shard_path) as source:
        feature = source.execute(
            "SELECT version_id, source_hash, config_json, symbols_json, sample_count "
            "FROM feature_versions WHERE status='ready' ORDER BY updated_at_utc DESC LIMIT 1"
        ).fetchone()
        if feature is None:
            raise ValueError(f"ColProj 分片没有 ready 版本: {colproj_shard_path}")
        colproj_version_id, source_hash, source_config_json, symbols_json, expected_count = feature
        source_config = json.loads(source_config_json)
        expected_config = _index_config(config)
        if source_config != expected_config:
            raise ValueError("ColProj 分片配置与当前行情图像配置不一致")
        symbols = tuple(json.loads(symbols_json))

    market = MarketData.from_sqlite(database_path, symbols=symbols)
    if market.source_hash != source_hash:
        raise ValueError("CNN 输入行情版本与 ColProj 分片版本不一致")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    version_id = f"cnn-{checkpoint_hash[:12]}-{hashlib.sha256(source_hash.encode()).hexdigest()[:10]}-{Path(colproj_shard_path).stem}"
    config_json = json.dumps(
        {
            **_index_config(config, model="cnn", dimension=embedder.dimension),
            "model_version": embedder.model_version,
            "checkpoint_sha256": checkpoint_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    now = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    inserted = 0
    with sqlite3.connect(output, timeout=120) as destination, sqlite3.connect(
        colproj_shard_path, timeout=120
    ) as source:
        destination.execute("PRAGMA journal_mode=WAL")
        destination.execute("PRAGMA synchronous=NORMAL")
        destination.executescript(INDEX_SCHEMA)
        destination.execute(
            """INSERT INTO feature_versions
            (version_id, source_hash, config_json, symbols_json, model, dimension,
             status, sample_count, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, 'cnn', ?, 'building', 0, ?, ?)
            ON CONFLICT(version_id) DO UPDATE SET status='building', updated_at_utc=excluded.updated_at_utc""",
            (version_id, source_hash, config_json, symbols_json, embedder.dimension, now, now),
        )
        existing_keys = {
            (symbol, sample_date)
            for symbol, sample_date in destination.execute(
                "SELECT symbol, date FROM samples WHERE version_id=?", (version_id,)
            )
        }

        cursor = source.execute(
            """SELECT s.symbol, s.date, l.base_close, l.return_1, l.return_5, l.return_10, l.return_20
            FROM samples s JOIN return_labels l
              ON l.version_id=s.version_id AND l.symbol=s.symbol AND l.date=s.date
            WHERE s.version_id=? ORDER BY s.date, s.symbol""",
            (colproj_version_id,),
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            rows = [row for row in rows if (row[0], row[1]) not in existing_keys]
            if not rows:
                continue
            charts = [
                build_query_charts(market, row[0], date.fromisoformat(row[1]), config)
                for row in rows
            ]
            embeddings = embedder.encode_query_charts_batch(charts, batch_size=batch_size)
            sample_rows = []
            label_rows = []
            for row, vectors in zip(rows, embeddings):
                symbol, date_text, base_close, ret1, ret5, ret10, ret20 = row
                sample_rows.append(
                    (
                        version_id,
                        symbol,
                        date_text,
                        _pack_vector(vectors.monthly),
                        _pack_vector(vectors.weekly),
                        _pack_vector(vectors.daily),
                    )
                )
                label_rows.append(
                    (version_id, symbol, date_text, base_close, ret1, ret5, ret10, ret20)
                )
            destination.executemany(
                "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?)", sample_rows
            )
            destination.executemany(
                "INSERT OR REPLACE INTO return_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                label_rows,
            )
            destination.commit()
            inserted += len(rows)
            existing_keys.update((row[0], row[1]) for row in rows)

        count = destination.execute(
            "SELECT COUNT(*) FROM samples WHERE version_id=?", (version_id,)
        ).fetchone()[0]
        if count != expected_count:
            raise ValueError(f"CNN 样本数与 ColProj 源分片不一致: {count} != {expected_count}")
        destination.execute(
            "UPDATE feature_versions SET status='ready', sample_count=?, updated_at_utc=? WHERE version_id=?",
            (count, datetime.now(timezone.utc).isoformat(), version_id),
        )
        destination.commit()

    return {
        "status": "ready",
        "version_id": version_id,
        "model_version": embedder.model_version,
        "checkpoint_sha256": checkpoint_hash,
        "source_hash": source_hash,
        "symbol_count": len(symbols),
        "sample_count": count,
        "inserted_count": inserted,
        "resumed_count": count - inserted,
        "index_path": str(output),
        "device": embedder.device,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def build_manifest(
    database_path: str | Path,
    colproj_manifest_path: str | Path,
    checkpoint_path: str | Path,
    index_dir: str | Path,
    manifest_path: str | Path,
    *,
    config: ProjectConfig,
    batch_id: str,
    batch_size: int = 256,
    device: str = "auto",
) -> dict[str, object]:
    base_manifest = json.loads(Path(colproj_manifest_path).read_text(encoding="utf-8"))
    selected = next(
        (item for item in base_manifest.get("indexes", []) if Path(item).stem == batch_id),
        None,
    )
    if selected is None:
        raise ValueError(f"ColProj 清单中不存在批次 {batch_id}")
    source_path = Path(colproj_manifest_path).parent / selected
    output_path = Path(index_dir) / f"{batch_id}.sqlite3"
    result = build_cnn_shard(
        database_path,
        source_path,
        checkpoint_path,
        output_path,
        config=config,
        batch_size=batch_size,
        device=device,
    )

    manifest_file = Path(manifest_path)
    with _manifest_lock(manifest_file):
        previous = {}
        if manifest_file.is_file():
            try:
                previous = json.loads(manifest_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
        index_paths = set(previous.get("indexes", []))
        if index_paths and (
            previous.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or previous.get("source_hash") not in (None, result["source_hash"])
        ):
            raise ValueError("CNN 清单已绑定其他模型检查点或行情快照，请使用新的清单路径")
        relative = str(output_path.resolve().relative_to(manifest_file.parent.resolve()))
        index_paths.add(relative)
        shards = sorted(index_paths)
        payload = {
            "format": "kline-cnn-index-manifest-v1",
            "model": "cnn",
            "model_version": result["model_version"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "source_hash": result["source_hash"],
            "config": {
                "chart_size": list(config.chart_size),
                "future_horizons": list(config.future_horizons),
                "dimension": CNNEmbedder.dimension,
                "source_config": _index_config(config),
            },
            "indexes": shards,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_file)
    return {**result, "manifest": str(manifest_file), "ready_shards": len(shards)}


@contextmanager
def _manifest_lock(path: Path):
    lock = Path(f"{path}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建一个可续传的全市场 CNN 特征分片")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--colproj-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/cnn_stage12"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/cnn_stage12.json"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args(argv)
    config = ProjectConfig.from_env()
    try:
        result = build_manifest(
            args.database,
            args.colproj_manifest,
            args.checkpoint,
            args.index_dir,
            args.manifest,
            config=config,
            batch_id=args.batch_id,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        print(f"CNN 分片构建失败: {exc}")
        return 2
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
