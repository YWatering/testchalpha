"""CLI for downloading and validating A-share daily bars with AKShare."""

from __future__ import annotations

import argparse
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
from .market_data import MarketData


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate A-share daily history through AKShare."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_A_SHARE_SYMBOLS),
        help="Comma-separated six-digit A-share codes",
    )
    parser.add_argument(
        "--all-listed",
        action="store_true",
        help="Discover currently listed Shanghai/Shenzhen/Beijing A shares",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip listed symbols for batching")
    parser.add_argument("--limit", type=int, help="Maximum listed symbols for this batch")
    parser.add_argument("--start-date", default="2018-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--adjust", choices=("qfq", "hfq", "raw"), default="qfq")
    parser.add_argument(
        "--endpoint",
        choices=("eastmoney", "sina"),
        default="eastmoney",
        help="AKShare upstream endpoint; use sina only when Eastmoney is unavailable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/a_share_daily.csv"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/a_share_daily.sqlite3"),
        help="SQLite snapshot path used by the live website",
    )
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
        provider = AKShareProvider(endpoint=args.endpoint)
        instruments = provider.list_a_share_universe() if args.all_listed else ()
        if args.offset < 0 or (args.limit is not None and args.limit <= 0):
            raise ValueError("--offset 不能为负数，--limit 必须大于 0")
        if instruments:
            selected = instruments[args.offset : args.offset + args.limit if args.limit else None]
            symbols = tuple(item["symbol"] for item in selected)
        else:
            symbols = tuple(item.strip() for item in args.symbols.split(","))
            instruments = build_instrument_metadata(symbols)
        result = sync_a_share_daily(
            provider,
            symbols,
            start_date,
            end_date,
            args.output,
            adjust="" if args.adjust == "raw" else args.adjust,
            incremental=not args.full_refresh,
            request_delay=args.request_delay,
            retries=args.retries,
            database_path=args.database,
            continue_on_error=args.all_listed,
            instruments=instruments,
        )
    except (ValueError, DataProviderError, DataSyncError, OSError) as exc:
        print(f"A 股数据同步失败: {exc}")
        return 2

    print(f"provider: {provider.name} {provider.version}")
    print(f"data: {result.data_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"symbols: {result.symbol_count}")
    print(f"rows: {result.row_count}")
    print(f"date_range: {result.start_date} .. {result.end_date}")
    print(f"sha256: {result.source_hash}")
    print(f"failed_symbols: {len(result.failed_symbols)}")
    market_data = MarketData.from_file(result.data_path)
    queryable = sum(
        market_data.validate_sample(symbol, trading_day).valid
        for symbol in market_data.symbols
        for trading_day in market_data.dates(symbol)
    )
    print(f"queryable_samples: {queryable}")
    for symbol in market_data.symbols:
        dates = market_data.dates(symbol)
        print(f"coverage[{symbol}]: {dates[0]} .. {dates[-1]} ({len(dates)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
