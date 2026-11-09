"""Materialize exact retrieval results in restartable, bounded batches."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .api import _load_cnn_embedder, _load_offline_index
from .cnn import build_cnn_sample_embedding
from .config import ProjectConfig
from .market_data import MarketData
from .retrieval import build_sample_embedding
from .result_cache import RetrievalResultCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="默认优先读取本地 CSMAR SQLite 数据库")
    parser.add_argument("--index", type=Path, help="ColProj/CNN 索引清单路径")
    parser.add_argument("--checkpoint", type=Path, help="CNN 检查点路径")
    parser.add_argument("--symbols", help="逗号分隔的股票代码；默认按索引股票代码顺序")
    parser.add_argument("--model", choices=("colproj", "cnn"), required=True)
    parser.add_argument("--cache", type=Path, help="默认使用 KLINE_RETRIEVAL_CACHE_PATH 或数据库旁缓存")
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--all", action="store_true", help="处理当前全部有效查询日期；规模可能极大")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--month-weight", type=float, default=1.0)
    parser.add_argument("--week-weight", type=float, default=1.0)
    parser.add_argument("--embargo-trading-days", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.all and args.max_queries <= 0:
            raise ValueError("--max-queries 必须大于 0；全量任务需显式指定 --all")
        if args.top_k <= 0 or args.embargo_trading_days < 0:
            raise ValueError("top-k 必须大于 0，embargo 不能小于 0")
        config = ProjectConfig.from_env()
        default_database = Path("data/processed/csmar_a_share_daily.sqlite3")
        database = args.database or (
            default_database if default_database.is_file() else config.data_path
        )
        cache_path = args.cache or config.retrieval_cache_path or database.with_name(
            f"{database.stem}.retrieval-cache.sqlite3"
        )
        index_path = args.index or (
            config.cnn_index_path if args.model == "cnn" else config.index_path
        )
        if index_path is None:
            local_manifest = Path(
                "data/processed/cnn_stage12.json"
                if args.model == "cnn"
                else "data/processed/colproj_stage10.json"
            )
            index_path = local_manifest if local_manifest.is_file() else None
        if index_path is None:
            raise ValueError(f"未配置 {args.model} 索引清单，请传入 --index")
        symbols = (
            tuple(dict.fromkeys(code.strip() for code in args.symbols.split(",") if code.strip()))
            if args.symbols
            else None
        )
        if database.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("预计算要求使用 SQLite 行情数据库，以便逐股票加载并控制内存")
        first_symbol = symbols[0] if symbols else None
        market = MarketData.from_sqlite(
            database,
            symbols=(first_symbol,) if first_symbol else None,
            price_basis=config.price_basis,
        )
        index = _load_offline_index(index_path, market, config)
        if index is None:
            raise ValueError(f"{args.model} 索引缺失或与当前行情快照/配置不兼容")
        embedder = None
        if args.model == "cnn":
            checkpoint = args.checkpoint or config.cnn_checkpoint_path or Path(
                "data/processed/cnn_stage11/cnn_stage11.pt"
            )
            embedder = _load_cnn_embedder(checkpoint)
            if embedder is None:
                raise ValueError("CNN 检查点未配置或无法加载")
        selected_symbols = tuple(symbol for symbol in (symbols or index.symbols) if symbol in index.symbols)
        if symbols and len(selected_symbols) != len(symbols):
            raise ValueError("指定股票不在当前模型索引覆盖范围内")
        parameters = {
            "month_weight": args.month_weight,
            "week_weight": args.week_weight,
            "day_weight": 1.0,
            "top_k": args.top_k,
            "embargo_trading_days": args.embargo_trading_days,
        }
        history_count = config.monthly_period * config.daily_window
        future_count = max(config.future_horizons)
        date_ranges: dict[str, tuple[date, date]] = {}
        target_count = 0
        for symbol in selected_symbols:
            symbol_market = MarketData.from_sqlite(
                database, symbols=(symbol,), price_basis=config.price_basis
            )
            valid_dates = _valid_query_dates(
                symbol_market.dates(symbol), history_count, future_count, args.date_from, args.date_to
            )
            if valid_dates:
                date_ranges[symbol] = (valid_dates[0], valid_dates[-1])
                target_count += len(valid_dates)
        if args.all and target_count * index.sample_count > 100_000_000:
            raise ValueError(
                f"本次约需 {target_count * index.sample_count:,} 次候选距离计算，"
                "当前仍是精确全扫描引擎，已阻止大型 N² 预计算；"
                "先用 --max-queries 做成本试点，ANN 验收通过后再开放全量任务。"
            )

        cache = RetrievalResultCache(cache_path)
        job_key = cache.begin_job(
            model=args.model,
            index_version=index.version_id,
            source_hash=index.source_hash,
            parameters=parameters,
            target_count=target_count,
            scope={
                "symbols": selected_symbols,
                "date_from": args.date_from.isoformat() if args.date_from else None,
                "date_to": args.date_to.isoformat() if args.date_to else None,
            },
        )
        completed = cache.count_scope(
            model=args.model,
            index_version=index.version_id,
            source_hash=index.source_hash,
            parameters=parameters,
            date_ranges=date_ranges,
        )
        generated = 0
        search_ms_total = 0.0
        started = perf_counter()
        try:
            for symbol in selected_symbols:
                if symbol not in date_ranges:
                    continue
                symbol_market = MarketData.from_sqlite(
                    database, symbols=(symbol,), price_basis=config.price_basis
                )
                query_dates = _valid_query_dates(
                    symbol_market.dates(symbol), history_count, future_count, args.date_from, args.date_to
                )
                for query_date in query_dates:
                    if not args.all and generated >= args.max_queries:
                        break
                    existing = cache.get(
                        model=args.model,
                        index_version=index.version_id,
                        source_hash=index.source_hash,
                        symbol=symbol,
                        query_date=query_date,
                        parameters=parameters,
                    )
                    if existing is not None:
                        continue
                    query = (
                        build_cnn_sample_embedding(
                            symbol_market, symbol, query_date, embedder, config=config
                        )
                        if embedder is not None
                        else build_sample_embedding(
                            symbol_market, symbol, query_date, config=config
                        )
                    )
                    matches, diagnostics = index.search(
                        query,
                        symbol_market,
                        month_weight=args.month_weight,
                        week_weight=args.week_weight,
                        top_k=args.top_k,
                        embargo_trading_days=args.embargo_trading_days,
                    )
                    search_ms_total += float(diagnostics.get("index_ms", 0.0))
                    cache.put(
                        matches,
                        model=args.model,
                        index_version=index.version_id,
                        source_hash=index.source_hash,
                        symbol=symbol,
                        query_date=query_date,
                        parameters=parameters,
                    )
                    generated += 1
                    if generated % 10 == 0:
                        completed = cache.count_scope(
                            model=args.model,
                            index_version=index.version_id,
                            source_hash=index.source_hash,
                            parameters=parameters,
                            date_ranges=date_ranges,
                        )
                        cache.update_job(job_key, completed, status="running")
                        print(f"progress: {completed}/{target_count} newly_built={generated}", flush=True)
                if not args.all and generated >= args.max_queries:
                    break
        except Exception as exc:
            completed = cache.count_scope(
                model=args.model,
                index_version=index.version_id,
                source_hash=index.source_hash,
                parameters=parameters,
                date_ranges=date_ranges,
            )
            cache.update_job(job_key, completed, status="failed", error=str(exc))
            raise

        completed = cache.count_scope(
            model=args.model,
            index_version=index.version_id,
            source_hash=index.source_hash,
            parameters=parameters,
            date_ranges=date_ranges,
        )
        scope_status = "complete" if completed >= target_count else "partial"
        full_universe = (
            symbols is None
            and args.date_from is None
            and args.date_to is None
            and target_count == index.sample_count
        )
        global_status = (
            "complete"
            if full_universe and completed >= index.sample_count
            else "partial"
        )
        cache.update_job(job_key, completed, status=scope_status)
        print(f"model: {args.model}")
        print(f"index_version: {index.version_id}")
        print(f"cache: {cache_path}")
        print(f"selected_query_keys: {target_count}")
        print(f"index_target_samples: {index.sample_count}")
        print(f"selected_scope_cached: {completed}/{target_count}")
        print(
            "estimated_payload_bytes_for_selected_scope: "
            f"{target_count * (6 + args.top_k * 45):,}"
        )
        print(f"cached_queries: {completed}")
        print(f"newly_built: {generated}")
        print(f"index_search_ms_total: {search_ms_total:.3f}")
        print(f"elapsed_seconds: {perf_counter() - started:.3f}")
        print(f"selected_scope_status: {scope_status}")
        print(f"global_coverage_status: {global_status}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"预计算失败: {exc}")
        return 2


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
