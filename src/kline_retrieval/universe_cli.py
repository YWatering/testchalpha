"""Prepare full-market backfill batches and print coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig
from .universe import prepare_universe_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = prepare_universe_batches(
            args.database,
            config=ProjectConfig.from_env(),
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as exc:
        print(f"全市场批次准备失败: {exc}")
        return 2
    print(f"database: {args.database}")
    for key in (
        "total_symbols",
        "ready_symbols",
        "failed_symbols",
        "coverage_ratio",
        "valid_sample_count",
        "latest_trading_date",
        "batches",
    ):
        print(f"{key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

