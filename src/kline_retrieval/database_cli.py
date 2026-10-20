"""Import a validated CSV or Parquet market snapshot into SQLite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from .data_sync import build_instrument_metadata
from .market_data import MarketData, MarketDataError
from .sqlite_store import write_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/a_share_daily.sqlite3"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        market_data = MarketData.from_file(args.input)
        rows = _read_rows(args.input)
        manifest = _read_manifest(args.input)
        source_hash = market_data.source_hash or ""
        manifest.setdefault("provider", "local_file")
        manifest["sha256"] = source_hash
        manifest["symbols"] = list(market_data.symbols)
        manifest["row_count"] = len(rows)
        write_snapshot(
            args.database,
            rows,
            source_hash=source_hash,
            manifest=manifest,
            instruments=build_instrument_metadata(market_data.symbols),
        )
    except (MarketDataError, OSError, ValueError) as exc:
        print(f"行情数据库导入失败: {exc}")
        return 2
    print(f"database: {args.database}")
    print(f"symbols: {len(market_data.symbols)}")
    print(f"rows: {len(rows)}")
    print(f"sha256: {source_hash}")
    return 0


def _read_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    try:
        import pandas as pd
    except ImportError as exc:
        raise ValueError("导入 Parquet 需要 pandas 和 pyarrow") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def _read_manifest(path: Path) -> dict[str, object]:
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据清单不是有效 JSON: {manifest_path}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
