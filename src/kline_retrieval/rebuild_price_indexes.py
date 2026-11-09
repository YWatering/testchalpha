"""Rebuild ColProj/CNN shard pairs for an active reconstructed-price snapshot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .cnn import CNNEmbedder
from .cnn_index import build_cnn_shard
from .config import ProjectConfig
from .index_manifest import publish_index_manifest
from .market_data import MarketData
from .offline_index import OfflineFeatureIndex, build_colproj_index


def rebuild_all_shards(
    database: str | Path,
    colproj_manifest: str | Path,
    cnn_manifest: str | Path,
    checkpoint: str | Path,
    *,
    config: ProjectConfig,
    batch_size: int = 256,
    device: str = "auto",
    start_batch: str | None = None,
) -> dict[str, object]:
    database_path = Path(database)
    colproj_manifest_path = Path(colproj_manifest)
    cnn_manifest_path = Path(cnn_manifest)
    checkpoint_path = Path(checkpoint)
    colproj_paths = _manifest_paths(colproj_manifest_path)
    cnn_payload = json.loads(cnn_manifest_path.read_text(encoding="utf-8"))
    cnn_paths = _manifest_paths(cnn_manifest_path)
    first_symbols = tuple(json.loads(_feature_metadata(colproj_paths[0])["symbols_json"]))
    market = MarketData.from_sqlite(
        database_path, symbols=first_symbols, price_basis="active"
    )
    if market.price_basis != "reconstructed":
        raise ValueError("正式行情库当前未激活收益率重建价格")

    staging = colproj_manifest_path.parent / "stage13_rebuild_tmp"
    staging.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    skipped: list[str] = []
    begin = start_batch is None
    for old_colproj_path, old_cnn_path in zip(colproj_paths, cnn_paths):
        batch_id = old_colproj_path.stem
        if old_cnn_path.stem != batch_id:
            raise ValueError("ColProj 与 CNN 清单的分片名称不一致")
        if not begin:
            if batch_id == start_batch:
                begin = True
            else:
                continue

        if _is_rebuilt_pair(old_colproj_path, old_cnn_path, market.source_hash):
            skipped.append(batch_id)
            print(f"[{batch_id}] already rebuilt; skip", flush=True)
            continue

        old_metadata = _feature_metadata(old_colproj_path)
        symbols = tuple(json.loads(old_metadata["symbols_json"]))
        expected_samples = int(old_metadata["sample_count"])
        temporary_colproj = staging / f"{batch_id}-colproj.sqlite3"
        temporary_cnn = staging / f"{batch_id}-cnn.sqlite3"

        rebuild_colproj = not _is_rebuilt_index(
            old_colproj_path, market.source_hash, expected_samples
        )
        if rebuild_colproj:
            colproj_result = build_colproj_index(
                database_path,
                temporary_colproj,
                symbols=symbols,
                config=config,
                replace=True,
                batch_size=500,
            )
            if colproj_result.sample_count != expected_samples:
                raise ValueError(
                    f"{batch_id} ColProj 样本数变化: {colproj_result.sample_count} != {expected_samples}"
                )
            colproj_source_path = temporary_colproj
        else:
            colproj_source_path = old_colproj_path

        cnn_result = build_cnn_shard(
            database_path,
            colproj_source_path,
            checkpoint_path,
            temporary_cnn,
            config=config,
            batch_size=batch_size,
            device=device,
        )
        _checkpoint_sqlite(temporary_cnn)
        if rebuild_colproj:
            _checkpoint_sqlite(temporary_colproj)
        colproj_index = OfflineFeatureIndex(
            temporary_colproj if rebuild_colproj else old_colproj_path
        )
        cnn_index = OfflineFeatureIndex(temporary_cnn)
        if (
            cnn_result["sample_count"] != expected_samples
            or colproj_index.source_hash != market.source_hash
            or cnn_index.source_hash != market.source_hash
            or colproj_index.config.get("price_basis") != "reconstructed"
            or cnn_index.config.get("price_basis") != "reconstructed"
            or not colproj_index.is_compatible(
                market.source_hash, config, price_basis="reconstructed"
            )
            or not cnn_index.is_compatible(
                market.source_hash, config, price_basis="reconstructed"
            )
        ):
            raise ValueError(f"{batch_id} 新分片校验失败，保留旧分片")

        gc.collect()
        _ensure_not_open(old_colproj_path)
        _ensure_not_open(old_cnn_path)
        _remove_sqlite_sidecars(old_cnn_path)
        if rebuild_colproj:
            _remove_sqlite_sidecars(old_colproj_path)
            os.replace(temporary_colproj, old_colproj_path)
        os.replace(temporary_cnn, old_cnn_path)
        _remove_sqlite_sidecars(temporary_colproj)
        _remove_sqlite_sidecars(temporary_cnn)
        completed.append(batch_id)
        print(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "symbols": len(symbols),
                    "sample_count": expected_samples,
                    "colproj_bytes": old_colproj_path.stat().st_size,
                    "cnn_bytes": old_cnn_path.stat().st_size,
                    "source_hash": market.source_hash,
                    "status": "replaced",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not begin:
        raise ValueError(f"清单中不存在起始批次: {start_batch}")

    # Publish only when every shard in both model families belongs to the new snapshot.
    for colproj_path, cnn_path in zip(colproj_paths, cnn_paths):
        if not _is_rebuilt_pair(colproj_path, cnn_path, market.source_hash):
            raise ValueError(f"索引尚未全部重建，未更新清单: {colproj_path.stem}")
    publish_index_manifest(colproj_manifest_path, colproj_paths)
    embedder = CNNEmbedder(checkpoint_path, device=device)
    source_config = {
        "monthly_period": config.monthly_period,
        "weekly_period": config.weekly_period,
        "daily_window": config.daily_window,
        "future_chart_window": config.future_chart_window,
        "chart_size": list(config.chart_size),
        "future_horizons": list(config.future_horizons),
        "price_basis": "reconstructed",
        "model": "colproj",
        "dimension": config.chart_width,
    }
    cnn_payload.update(
        {
            "model_version": embedder.model_version,
            "checkpoint_sha256": _file_hash(checkpoint_path),
            "source_hash": market.source_hash,
            "config": {
                "chart_size": list(config.chart_size),
                "future_horizons": list(config.future_horizons),
                "dimension": embedder.dimension,
                "source_config": source_config,
            },
            "indexes": [os.path.relpath(path, cnn_manifest_path.parent) for path in cnn_paths],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_write_json(cnn_manifest_path, cnn_payload)
    try:
        staging.rmdir()
    except OSError:
        pass
    return {
        "status": "ready",
        "source_hash": market.source_hash,
        "price_basis": market.price_basis,
        "completed_batches": completed,
        "skipped_batches": skipped,
        "colproj_shards": len(colproj_paths),
        "cnn_shards": len(cnn_paths),
    }


def _manifest_paths(path: Path) -> tuple[Path, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = tuple((path.parent / item).resolve() for item in payload.get("indexes", []))
    if not paths or any(not item.is_file() for item in paths):
        raise FileNotFoundError(f"索引清单为空或分片缺失: {path}")
    return paths


def _feature_metadata(path: Path) -> dict[str, object]:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT symbols_json, sample_count FROM feature_versions "
            "WHERE status='ready' ORDER BY updated_at_utc DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError(f"分片没有 ready 版本: {path}")
    return {"symbols_json": row[0], "sample_count": row[1]}


def _is_rebuilt_pair(colproj_path: Path, cnn_path: Path, source_hash: str) -> bool:
    try:
        colproj = OfflineFeatureIndex(colproj_path)
        cnn = OfflineFeatureIndex(cnn_path)
    except (OSError, ValueError, sqlite3.Error):
        return False
    return (
        colproj.source_hash == source_hash
        and cnn.source_hash == source_hash
        and colproj.config.get("price_basis") == "reconstructed"
        and cnn.config.get("price_basis") == "reconstructed"
        and colproj.sample_count == cnn.sample_count
    )


def _is_rebuilt_index(path: Path, source_hash: str, sample_count: int) -> bool:
    try:
        index = OfflineFeatureIndex(path)
    except (OSError, ValueError, sqlite3.Error):
        return False
    return (
        index.source_hash == source_hash
        and index.config.get("price_basis") == "reconstructed"
        and index.sample_count == sample_count
    )


def _checkpoint_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path, timeout=120)
    try:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        connection.close()
    if result is not None and result[0] != 0:
        raise RuntimeError(f"SQLite WAL checkpoint 未完成: {path}")
    _remove_sqlite_sidecars(path)


def _ensure_not_open(path: Path) -> None:
    result = subprocess.run(
        ["lsof", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode == 0 and len(result.stdout.splitlines()) > 1:
        raise RuntimeError(
            f"索引文件仍被进程占用，拒绝替换: {path}\n{result.stdout.strip()}"
        )


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--colproj-manifest", type=Path, required=True)
    parser.add_argument("--cnn-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--start-batch")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ProjectConfig.from_env()
    config = replace(
        config,
        data_path=args.database,
        price_basis="active",
        active_symbols=(),
    )
    try:
        result = rebuild_all_shards(
            args.database,
            args.colproj_manifest,
            args.cnn_manifest,
            args.checkpoint,
            config=config,
            batch_size=args.batch_size,
            device=args.device,
            start_batch=args.start_batch,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"阶段十三索引重建失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
