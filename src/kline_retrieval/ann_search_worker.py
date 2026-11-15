"""JSON-lines FAISS search worker intentionally isolated from PyTorch."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .ann_index import MultiScaleANNIndex
from .market_data import MarketData
from .retrieval import RetrievalSample, SampleEmbeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ann-index", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument(
        "--price-basis", choices=("active", "raw", "reconstructed"), required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ann = MultiScaleANNIndex(
            args.ann_index, candidate_k=args.candidate_k, nprobe=args.nprobe
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _write({"type": "error", "error": str(exc)})
        return 2
    _write({"type": "ready", "model": ann.model, "version_id": ann.version_id})
    loaded_symbol: str | None = None
    market: MarketData | None = None
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("type") == "close":
                return 0
            if request.get("type") != "search":
                raise ValueError("不支持的 ANN worker 请求类型")
            symbol = str(request["symbol"])
            if market is None or loaded_symbol != symbol:
                market = MarketData.from_sqlite(
                    args.database, symbols=(symbol,), price_basis=args.price_basis
                )
                loaded_symbol = symbol
            query = _decode_query(request)
            matches, diagnostics = ann.search(
                query,
                market,
                month_weight=float(request["month_weight"]),
                week_weight=float(request["week_weight"]),
                top_k=int(request["top_k"]),
                embargo_trading_days=int(request["embargo_trading_days"]),
            )
            _write(
                {
                    "id": request.get("id"),
                    "ok": True,
                    "matches": [
                        {
                            "rank": match.rank,
                            "symbol": match.sample.symbol,
                            "date": match.sample.date.isoformat(),
                            "distance": match.distance,
                            "monthly_distance": match.monthly_distance,
                            "weekly_distance": match.weekly_distance,
                            "daily_distance": match.daily_distance,
                        }
                        for match in matches
                    ],
                    "diagnostics": diagnostics,
                }
            )
        except (KeyError, TypeError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
            _write({"id": request.get("id"), "ok": False, "error": str(exc)})
    return 0


def _decode_query(request: dict[str, Any]) -> RetrievalSample:
    values = request["embeddings"]
    return RetrievalSample(
        symbol=str(request["symbol"]),
        date=date.fromisoformat(str(request["date"])),
        embeddings=SampleEmbeddings(
            monthly=tuple(float(value) for value in values["monthly"]),
            weekly=tuple(float(value) for value in values["weekly"]),
            daily=tuple(float(value) for value in values["daily"]),
        ),
    )


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
