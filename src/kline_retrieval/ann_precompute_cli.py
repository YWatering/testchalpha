"""Precompute ColProj retrieval results through an isolated FAISS worker."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .ann_index import MultiScaleANNIndex
from .config import ProjectConfig
from .market_data import MarketData
from .retrieval import build_sample_embedding
from .result_cache import RetrievalResultCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ann-index", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--symbols", required=True, help="逗号分隔的股票代码")
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--month-weight", type=float, default=1.0)
    parser.add_argument("--week-weight", type=float, default=1.0)
    parser.add_argument("--embargo-trading-days", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.max_queries <= 0 or args.top_k <= 0 or args.embargo_trading_days < 0:
            raise ValueError("max-queries/top-k 必须大于 0，embargo 不能小于 0")
        config = ProjectConfig.from_env()
        ann = MultiScaleANNIndex(
            args.ann_index,
            candidate_k=config.ann_candidate_k,
            nprobe=config.ann_nprobe,
        )
        if ann.model != "colproj":
            raise ValueError("当前隔离 ANN worker 仅支持 ColProj，CNN 需单独运行时隔离")
        symbols = tuple(dict.fromkeys(item.strip() for item in args.symbols.split(",") if item.strip()))
        if not symbols or any(symbol not in ann.symbols for symbol in symbols):
            raise ValueError("指定股票为空或不在 ANN 索引覆盖范围内")
        cache = RetrievalResultCache(args.cache)
        parameters = {
            "month_weight": args.month_weight,
            "week_weight": args.week_weight,
            "day_weight": 1.0,
            "top_k": args.top_k,
            "embargo_trading_days": args.embargo_trading_days,
            "retrieval_engine": "faiss-ann-exact-rerank-v1",
            "ann_candidate_k": config.ann_candidate_k,
            "ann_nprobe": config.ann_nprobe,
        }
        selected: list[tuple[str, date]] = []
        markets: dict[str, MarketData] = {}
        history_count = config.monthly_period * config.daily_window
        future_count = max(config.future_horizons)
        for symbol in symbols:
            market = MarketData.from_sqlite(args.database, symbols=(symbol,), price_basis=config.price_basis)
            markets[symbol] = market
            dates = market.dates(symbol)
            first = history_count - 1
            last = len(dates) - future_count - 1
            if last < first:
                continue
            selected.extend(
                (symbol, trading_date)
                for trading_date in dates[first : last + 1]
                if (args.date_from is None or trading_date >= args.date_from)
                and (args.date_to is None or trading_date <= args.date_to)
            )
        selected = selected[: args.max_queries]
        job_key = cache.begin_job(
            model="colproj",
            index_version=ann.version_id,
            source_hash=ann.source_hash,
            parameters=parameters,
            target_count=len(selected),
            scope={"symbols": symbols, "worker": "isolated-faiss"},
        )
        generated = 0
        completed = 0
        started = perf_counter()
        for symbol, query_date in selected:
            if cache.get(
                model="colproj",
                index_version=ann.version_id,
                source_hash=ann.source_hash,
                symbol=symbol,
                query_date=query_date,
                parameters=parameters,
            ) is not None:
                completed += 1
                continue
            market = markets[symbol]
            query = build_sample_embedding(market, symbol, query_date, config=config)
            matches, diagnostics = ann.search(
                query,
                market,
                month_weight=args.month_weight,
                week_weight=args.week_weight,
                top_k=args.top_k,
                embargo_trading_days=args.embargo_trading_days,
            )
            cache.put(
                matches,
                model="colproj",
                index_version=ann.version_id,
                source_hash=ann.source_hash,
                symbol=symbol,
                query_date=query_date,
                parameters=parameters,
            )
            generated += 1
            completed += 1
            print(
                f"progress: {generated}/{len(selected)} symbol={symbol} date={query_date} "
                f"candidate_count={diagnostics['ann_candidate_count']} ann_ms={diagnostics['ann_ms']}",
                flush=True,
            )
        cache.update_job(
            job_key,
            completed,
            status="complete" if completed == len(selected) else "partial",
        )
        print(f"cache: {args.cache}")
        print(f"selected_query_keys: {len(selected)}")
        print(f"newly_built: {generated}")
        print(f"elapsed_seconds: {perf_counter() - started:.3f}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ANN 预计算失败: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
