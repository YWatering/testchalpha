"""Build the next pending ColProj shard and atomically publish ready shards."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig
from .index_manifest import publish_index_manifest
from .offline_index import build_colproj_index
from .sqlite_store import SCHEMA
from .universe import prepare_universe_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/colproj_stage10"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/colproj_stage10.json"))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--batch-id")
    return parser


def build_next_batch(
    database_path: str | Path,
    index_dir: str | Path,
    manifest_path: str | Path,
    *,
    config: ProjectConfig,
    batch_size: int = 200,
    batch_id: str | None = None,
) -> dict[str, object]:
    prepare_universe_batches(database_path, config=config, batch_size=batch_size)
    database = Path(database_path)
    with sqlite3.connect(database, timeout=30) as connection:
        connection.executescript(SCHEMA)
        if batch_id:
            row = connection.execute(
                "SELECT batch_id, symbols_json, status, attempts FROM universe_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT batch_id, symbols_json, status, attempts FROM universe_batches WHERE status IN ('pending', 'failed') ORDER BY batch_number LIMIT 1"
            ).fetchone()
        if row is None:
            return {"status": "complete", "message": "没有待处理批次"}
        selected_id, symbols_json, status, attempts = row
        symbols = tuple(json.loads(symbols_json))
        next_attempt = int(attempts) + 1
        connection.execute(
            "UPDATE universe_batches SET status='building', attempts=?, last_error=NULL, updated_at_utc=? WHERE batch_id=?",
            (next_attempt, datetime.now(timezone.utc).isoformat(), selected_id),
        )
        connection.commit()

    index_path = Path(index_dir) / f"{selected_id}.sqlite3"
    try:
        result = build_colproj_index(
            database,
            index_path,
            symbols=symbols,
            config=config,
            batch_size=500,
        )
    except Exception as exc:
        with sqlite3.connect(database, timeout=30) as connection:
            connection.execute(
                "UPDATE universe_batches SET status='failed', last_error=?, updated_at_utc=? WHERE batch_id=?",
                (str(exc), datetime.now(timezone.utc).isoformat(), selected_id),
            )
            connection.commit()
        raise

    with sqlite3.connect(database, timeout=30) as connection:
        connection.execute(
            "UPDATE universe_batches SET status='ready', last_error=NULL, updated_at_utc=? WHERE batch_id=?",
            (datetime.now(timezone.utc).isoformat(), selected_id),
        )
        ready_ids = [
            (row[0],)
            for row in connection.execute(
                "SELECT batch_id FROM universe_batches WHERE status='ready' ORDER BY batch_number"
            )
        ]
        connection.commit()
    ready_paths = [Path(index_dir) / f"{item[0]}.sqlite3" for item in ready_ids if (Path(index_dir) / f"{item[0]}.sqlite3").is_file()]
    manifest = publish_index_manifest(manifest_path, ready_paths) if ready_paths else None
    return {
        "status": "ready",
        "batch_id": selected_id,
        "index_path": str(index_path),
        "sample_count": result.sample_count,
        "inserted_count": result.inserted_count,
        "ready_shards": len(ready_paths),
        "manifest": str(manifest_path) if manifest else None,
        "manifest_content": manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_next_batch(
            args.database,
            args.index_dir,
            args.manifest,
            config=ProjectConfig.from_env(),
            batch_size=args.batch_size,
            batch_id=args.batch_id,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"批次索引构建失败: {exc}")
        return 2
    for key, value in result.items():
        if key != "manifest_content":
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

