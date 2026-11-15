"""Precompute ColProj or CNN retrieval results through isolated FAISS search."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .ann_worker_client import (
    ANN_RETRIEVAL_ENGINE,
    ANNWorkerClient,
    checkpoint_sha256,
    read_ann_manifest_metadata,
)
from .cnn import CNNEmbedder, build_cnn_sample_embedding
from .config import ProjectConfig
from .market_data import MarketData
from .retrieval import build_sample_embedding
from .result_cache import RetrievalResultCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ann-index", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="CNN 检查点；ColProj 不需要")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--symbols", help="逗号分隔的股票代码")
    parser.add_argument("--all", action="store_true", help="处理当前 ANN 清单覆盖的全部股票")
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="ANN 候选基数；默认等权综合检索会据此生成有界全局候选池",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--month-weight", type=float, default=1.0)
    parser.add_argument("--week-weight", type=float, default=1.0)
    parser.add_argument("--embargo-trading-days", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache: RetrievalResultCache | None = None
    job_key: str | None = None
    completed = 0
    try:
        if args.max_queries <= 0 or args.top_k <= 0 or args.embargo_trading_days < 0:
            raise ValueError("max-queries/top-k 必须大于 0，embargo 不能小于 0")
        if not args.all and not args.symbols:
            raise ValueError("请指定 --symbols，或显式使用 --all")
        if args.database.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("ANN 预计算要求使用 SQLite 行情数据库")

        config = ProjectConfig.from_env()
        metadata = read_ann_manifest_metadata(args.ann_index)
        candidate_k = args.candidate_k or (
            config.cnn_ann_candidate_k if metadata.model == "cnn" else config.ann_candidate_k
        )
        if candidate_k <= 0:
            raise ValueError("candidate-k 必须大于 0")
        symbols = (
            metadata.symbols
            if args.all
            else tuple(dict.fromkeys(item.strip() for item in args.symbols.split(",") if item.strip()))
        )
        if not symbols or any(symbol not in metadata.symbols for symbol in symbols):
            raise ValueError("指定股票为空或不在 ANN 索引覆盖范围内")

        embedder: CNNEmbedder | None = None
        if metadata.model == "cnn":
            checkpoint = args.checkpoint or config.cnn_checkpoint_path or Path(
                "data/processed/cnn_stage11/cnn_stage11.pt"
            )
            if not checkpoint.is_file():
                raise ValueError("CNN 检查点不存在，请传入 --checkpoint")
            if metadata.checkpoint_sha256 != checkpoint_sha256(checkpoint):
                raise ValueError("CNN 检查点与 ANN 源索引绑定的检查点不一致")
            embedder = CNNEmbedder(checkpoint, device=args.device)
        elif metadata.model != "colproj":
            raise ValueError(f"不支持的 ANN 模型: {metadata.model}")

        parameters = {
            "month_weight": args.month_weight,
            "week_weight": args.week_weight,
            "day_weight": 1.0,
            "top_k": args.top_k,
            "embargo_trading_days": args.embargo_trading_days,
            "retrieval_engine": ANN_RETRIEVAL_ENGINE,
            "ann_candidate_k": candidate_k,
            "ann_nprobe": config.ann_nprobe,
        }
        history_count = config.monthly_period * config.daily_window
        future_count = max(config.future_horizons)
        target_count = _count_queries(
            args.database,
            symbols,
            config=config,
            history_count=history_count,
            future_count=future_count,
            date_from=args.date_from,
            date_to=args.date_to,
            limit=None if args.all else args.max_queries,
        )
        cache = RetrievalResultCache(args.cache)
        job_key = cache.begin_job(
            model=metadata.model,
            index_version=metadata.version_id,
            source_hash=metadata.source_hash,
            parameters=parameters,
            target_count=target_count,
            scope={
                "symbols": symbols,
                "date_from": args.date_from.isoformat() if args.date_from else None,
                "date_to": args.date_to.isoformat() if args.date_to else None,
                "worker": "isolated-faiss",
            },
        )

        worker_context = (
            ANNWorkerClient(
                args.database,
                args.ann_index,
                candidate_k=candidate_k,
                nprobe=config.ann_nprobe,
                price_basis=config.price_basis,
            )
            if metadata.model == "cnn"
            else nullcontext(None)
        )
        generated = 0
        visited = 0
        started = perf_counter()
        with worker_context as worker:
            if metadata.model == "colproj":
                from .ann_index import MultiScaleANNIndex

                ann = MultiScaleANNIndex(
                    args.ann_index, candidate_k=candidate_k, nprobe=config.ann_nprobe
                )
            else:
                ann = None
            for symbol in symbols:
                market = MarketData.from_sqlite(
                    args.database, symbols=(symbol,), price_basis=config.price_basis
                )
                query_dates = _valid_query_dates(
                    market.dates(symbol),
                    history_count,
                    future_count,
                    args.date_from,
                    args.date_to,
                )
                for query_date in query_dates:
                    if not args.all and visited >= args.max_queries:
                        break
                    visited += 1
                    if cache.get(
                        model=metadata.model,
                        index_version=metadata.version_id,
                        source_hash=metadata.source_hash,
                        symbol=symbol,
                        query_date=query_date,
                        parameters=parameters,
                    ) is not None:
                        completed += 1
                        continue
                    query = (
                        build_cnn_sample_embedding(
                            market, symbol, query_date, embedder, config=config
                        )
                        if embedder is not None
                        else build_sample_embedding(market, symbol, query_date, config=config)
                    )
                    if worker is not None:
                        matches, diagnostics = worker.search(
                            query,
                            month_weight=args.month_weight,
                            week_weight=args.week_weight,
                            top_k=args.top_k,
                            embargo_trading_days=args.embargo_trading_days,
                        )
                    else:
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
                        model=metadata.model,
                        index_version=metadata.version_id,
                        source_hash=metadata.source_hash,
                        symbol=symbol,
                        query_date=query_date,
                        parameters=parameters,
                    )
                    generated += 1
                    completed += 1
                    if completed % 10 == 0:
                        cache.update_job(job_key, completed, status="running")
                    print(
                        f"progress: {completed}/{target_count} model={metadata.model} "
                        f"symbol={symbol} date={query_date} "
                        f"candidate_count={diagnostics['ann_candidate_count']} "
                        f"ann_ms={diagnostics['ann_ms']}",
                        flush=True,
                    )
                if not args.all and visited >= args.max_queries:
                    break
        cache.update_job(
            job_key,
            completed,
            status="complete" if completed == target_count else "partial",
        )
        print(f"model: {metadata.model}")
        print(f"ann_version: {metadata.version_id}")
        print(f"cache: {args.cache}")
        print(f"selected_query_keys: {target_count}")
        print(f"newly_built: {generated}")
        print(f"elapsed_seconds: {perf_counter() - started:.3f}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        if cache is not None and job_key is not None:
            cache.update_job(job_key, completed, status="failed", error=str(exc))
        print(f"ANN 预计算失败: {exc}")
        return 2


def _count_queries(
    database: Path,
    symbols: tuple[str, ...],
    *,
    config: ProjectConfig,
    history_count: int,
    future_count: int,
    date_from: date | None,
    date_to: date | None,
    limit: int | None,
) -> int:
    count = 0
    for symbol in symbols:
        market = MarketData.from_sqlite(database, symbols=(symbol,), price_basis=config.price_basis)
        count += len(
            _valid_query_dates(
                market.dates(symbol), history_count, future_count, date_from, date_to
            )
        )
        if limit is not None and count >= limit:
            return limit
    return count


def _valid_query_dates(
    dates: tuple[date, ...],
    history_count: int,
    future_count: int,
    date_from: date | None,
    date_to: date | None,
) -> tuple[date, ...]:
    first = history_count - 1
    last = len(dates) - future_count - 1
    if last < first:
        return ()
    return tuple(
        item
        for item in dates[first : last + 1]
        if (date_from is None or item >= date_from) and (date_to is None or item <= date_to)
    )


if __name__ == "__main__":
    raise SystemExit(main())
