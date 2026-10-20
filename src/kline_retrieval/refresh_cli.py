"""Refresh the local A-share SQLite snapshot once or on an interval."""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Sequence

from .data_provider import AKShareProvider, DataProviderError
from .data_sync import (
    DEFAULT_A_SHARE_SYMBOLS,
    DataSyncError,
    build_instrument_metadata,
    sync_a_share_daily,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_A_SHARE_SYMBOLS))
    parser.add_argument("--all-listed", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--adjust", choices=("qfq", "hfq", "raw"), default="qfq")
    parser.add_argument("--endpoint", choices=("eastmoney", "sina"), default="eastmoney")
    parser.add_argument("--output", type=Path, default=Path("data/processed/a_share_daily.csv"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/a_share_daily.sqlite3"))
    parser.add_argument("--request-delay", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--watch", action="store_true", help="持续刷新，默认每 5 分钟一次")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval_seconds <= 0:
        print("--interval-seconds 必须大于 0")
        return 2
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        print("--offset 不能为负数，--limit 必须大于 0")
        return 2
    try:
        provider = AKShareProvider(endpoint=args.endpoint)
        start_date = date.fromisoformat(args.start_date)
    except (ValueError, DataProviderError) as exc:
        print(f"A 股数据刷新失败: {exc}")
        return 2

    while True:
        try:
            instruments = provider.list_a_share_universe() if args.all_listed else ()
            if instruments:
                selected = instruments[
                    args.offset : args.offset + args.limit if args.limit else None
                ]
                symbols = tuple(item["symbol"] for item in selected)
            else:
                symbols = tuple(item.strip() for item in args.symbols.split(","))
                instruments = build_instrument_metadata(symbols)
            result = sync_a_share_daily(
                provider,
                symbols,
                start_date,
                date.today(),
                args.output,
                adjust="" if args.adjust == "raw" else args.adjust,
                database_path=args.database,
                continue_on_error=args.all_listed,
                request_delay=args.request_delay,
                retries=args.retries,
                instruments=instruments,
            )
            print(
                f"updated={result.end_date} symbols={result.symbol_count} "
                f"rows={result.row_count} failed={len(result.failed_symbols)} "
                f"sha256={result.source_hash}",
                flush=True,
            )
        except (DataProviderError, DataSyncError, OSError) as exc:
            print(f"A 股数据刷新失败: {exc}", flush=True)
            if not args.watch:
                return 2
        if not args.watch:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
