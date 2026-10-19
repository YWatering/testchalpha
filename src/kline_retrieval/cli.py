"""Command-line entry point for validating and inspecting a market-data file."""

from __future__ import annotations

import argparse
from datetime import date
import logging
from typing import Sequence

from .config import ProjectConfig
from .logging_config import configure_logging
from .market_data import MarketData, MarketDataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and validate market data for the K-line retrieval project."
    )
    parser.add_argument("--data", help="CSV or Parquet market-data file")
    parser.add_argument("--symbol", help="Optional symbol to validate")
    parser.add_argument("--date", dest="trading_date", help="Optional YYYY-MM-DD sample date")
    parser.add_argument("--history-count", type=int, default=240)
    parser.add_argument("--future-count", type=int, default=21)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging(getattr(logging, args.log_level))
    config = ProjectConfig.from_env()
    data_path = args.data or str(config.data_path)

    if bool(args.symbol) != bool(args.trading_date):
        logger.error("--symbol 和 --date 必须同时提供")
        return 2

    try:
        market_data = MarketData.from_file(data_path)
    except MarketDataError as exc:
        logger.error("加载行情数据失败: %s", exc)
        return 2

    logger.info(
        "行情数据加载成功: path=%s symbols=%d bars=%d sha256=%s",
        data_path,
        len(market_data.symbols),
        sum(len(market_data.dates(symbol)) for symbol in market_data.symbols),
        market_data.source_hash or "n/a",
    )
    print(f"symbols: {', '.join(market_data.symbols)}")

    if args.symbol and args.trading_date:
        try:
            validation = market_data.validate_sample(
                args.symbol,
                date.fromisoformat(args.trading_date),
                history_count=args.history_count,
                future_count=args.future_count,
            )
        except (ValueError, MarketDataError) as exc:
            logger.error("样本校验失败: %s", exc)
            return 2

        print(f"sample_valid: {validation.valid}")
        if validation.reasons:
            print("reasons:")
            for reason in validation.reasons:
                print(f"- {reason}")

    return 0
