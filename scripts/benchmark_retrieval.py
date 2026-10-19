"""Measure cold and warm ColProj retrieval on a local market-data file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import MarketData, SampleEmbeddingCache, search_market_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/demo_market_data.csv")
    parser.add_argument("--symbol", help="默认选择第一只股票")
    parser.add_argument("--date", help="默认选择最后一个交易日")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    market_data = MarketData.from_file(args.data)
    symbol = args.symbol or market_data.symbols[0]
    trading_dates = market_data.dates(symbol)
    trading_date = args.date or trading_dates[-1].isoformat()
    cache = SampleEmbeddingCache()
    runs: list[dict[str, int | float]] = []
    for _ in range(2):
        diagnostics: dict[str, int | float] = {}
        matches = search_market_data(
            market_data,
            symbol,
            trading_date,
            top_k=args.top_k,
            embedding_cache=cache,
            diagnostics=diagnostics,
        )
        diagnostics["returned_matches"] = len(matches)
        runs.append(diagnostics)

    print(
        json.dumps(
            {
                "source_hash": market_data.source_hash,
                "symbol": symbol,
                "date": trading_date,
                "runs": [{"run": "cold" if i == 0 else "warm", **run} for i, run in enumerate(runs)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
