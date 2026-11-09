"""Materialize return-reconstructed OHLC prices in a project SQLite snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from .price_reconstruction import materialize_reconstructed_prices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument(
        "--anchor-date",
        type=date.fromisoformat,
        help="统一锚点日期；未指定时每只股票以其最新交易日为锚点",
    )
    parser.add_argument("--anchor-close", type=float)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        report = materialize_reconstructed_prices(
            args.database,
            quality_report_path=args.quality_report,
            anchor_date=args.anchor_date,
            anchor_close=args.anchor_close,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as exc:
        print(f"收益率重建 OHLC 失败: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status_counts"].get("reconstructed") == report["symbol_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
