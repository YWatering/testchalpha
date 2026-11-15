"""Build stage-fifteen weekly labels and ColProj prediction pilots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

from .analytics import similarity_weights, summarize_returns
from .ann_index import MultiScaleANNIndex
from .ann_worker_client import read_ann_manifest_metadata
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData, MarketDataError
from .retrieval import SimilarityMatch, build_sample_embedding
from .stage15_store import Stage15PredictionStore


@dataclass(frozen=True, slots=True)
class WeeklySchedule:
    sequence: int
    iso_year: int
    iso_week: int
    signal_date: date
    entry_date: date | None
    exit_date: date | None

    @property
    def label_complete(self) -> bool:
        return self.entry_date is not None and self.exit_date is not None


@dataclass(frozen=True, slots=True)
class TradableLabel:
    symbol: str
    signal_date: date
    entry_date: date | None
    exit_date: date | None
    status: str
    entry_open: float | None = None
    exit_close: float | None = None
    entry_amount: float | None = None
    exit_amount: float | None = None
    gross_return: float | None = None

    @property
    def available_date(self) -> date | None:
        return self.exit_date if self.status == "tradable" else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    labels = subparsers.add_parser("build-labels", help="生成周历和50只股票真实收益标签")
    _add_common_arguments(labels)
    labels.add_argument(
        "--calendar-output",
        type=Path,
        default=Path("research/stage15/weekly_calendar.csv"),
    )
    labels.add_argument(
        "--labels-output",
        type=Path,
        default=Path("research/stage15/tradable_labels.csv"),
    )
    labels.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("research/stage15/weekly_dataset_manifest.json"),
    )
    labels.add_argument(
        "--report-output",
        type=Path,
        default=Path("research/stage15/WEEKLY_DATASET.md"),
    )

    predict = subparsers.add_parser("predict-colproj", help="运行ColProj周频预测试点")
    _add_common_arguments(predict)
    _add_colproj_arguments(predict)
    predict.add_argument("--max-queries", type=int)
    predict.add_argument("--symbol", action="append", dest="symbols")
    predict.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("research/stage15/colproj_prediction_pilot.csv"),
    )
    predict.add_argument(
        "--matches-output",
        type=Path,
        default=Path("research/stage15/colproj_prediction_matches_pilot.csv"),
    )
    predict.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("research/stage15/colproj_prediction_pilot_manifest.json"),
    )

    batch = subparsers.add_parser(
        "predict-colproj-batch",
        help="断点续跑ColProj周频批量预测",
    )
    _add_common_arguments(batch)
    _add_colproj_arguments(batch)
    batch.add_argument("--max-new-queries", type=int, default=100)
    batch.add_argument("--all", action="store_true", help="处理范围内全部缺失查询")
    batch.add_argument("--symbol", action="append", dest="symbols")
    batch.add_argument(
        "--store",
        type=Path,
        default=Path("data/processed/stage15_colproj_predictions.sqlite3"),
    )
    batch.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("research/stage15/colproj_validation_predictions.csv"),
    )
    batch.add_argument(
        "--matches-output",
        type=Path,
        default=Path("research/stage15/colproj_validation_matches.csv"),
    )
    batch.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("research/stage15/colproj_validation_manifest.json"),
    )
    return parser


def _add_colproj_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ann-manifest",
        type=Path,
        default=Path("data/processed/colproj_ann_stage14.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--overfetch-k", type=int, default=500)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--minimum-matches", type=int, default=20)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/csmar_a_share_daily.sqlite3"),
    )
    parser.add_argument(
        "--stock-pool",
        type=Path,
        default=Path("research/stage15/stock_pool.csv"),
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-10-23")
    parser.add_argument("--horizon-sessions", type=int, default=5)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-labels":
            result = build_weekly_dataset(
                args.database,
                args.stock_pool,
                start_date=args.start_date,
                end_date=args.end_date,
                horizon_sessions=args.horizon_sessions,
            )
            write_weekly_dataset(
                result,
                calendar_path=args.calendar_output,
                labels_path=args.labels_output,
                manifest_path=args.manifest_output,
                report_path=args.report_output,
            )
            print(f"weekly_signals: {len(result['schedule'])}")
            print(f"labels: {len(result['labels'])}")
            print(f"tradable_labels: {result['manifest']['tradable_label_count']}")
            return 0

        if args.command == "predict-colproj-batch":
            if not args.all and args.max_new_queries <= 0:
                raise ValueError("max-new-queries 必须大于 0，或使用 --all")
            result = run_colproj_batch(
                args.database,
                args.stock_pool,
                args.ann_manifest,
                args.store,
                start_date=args.start_date,
                end_date=args.end_date,
                horizon_sessions=args.horizon_sessions,
                top_k=args.top_k,
                overfetch_k=args.overfetch_k,
                candidate_k=args.candidate_k,
                nprobe=args.nprobe,
                minimum_matches=args.minimum_matches,
                max_new_queries=None if args.all else args.max_new_queries,
                selected_symbols=args.symbols,
                predictions_output=args.predictions_output,
                matches_output=args.matches_output,
            )
            output = Path(args.manifest_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"run_key: {result['run_key']}")
            print(f"progress: {result['completed_count']}/{result['target_count']}")
            print(f"newly_built: {result['newly_built']}")
            print(f"status: {result['status']}")
            return 0

        result = run_colproj_pilot(
            args.database,
            args.stock_pool,
            args.ann_manifest,
            start_date=args.start_date,
            end_date=args.end_date,
            horizon_sessions=args.horizon_sessions,
            top_k=args.top_k,
            overfetch_k=args.overfetch_k,
            candidate_k=args.candidate_k,
            nprobe=args.nprobe,
            minimum_matches=args.minimum_matches,
            max_queries=args.max_queries,
            selected_symbols=args.symbols,
        )
        write_prediction_pilot(
            result,
            predictions_path=args.predictions_output,
            matches_path=args.matches_output,
            manifest_path=args.manifest_output,
        )
        print(f"queries: {len(result['predictions'])}")
        print(f"matches: {len(result['matches'])}")
        return 0
    except (OSError, ValueError, sqlite3.Error, MarketDataError, RuntimeError) as exc:
        print(f"阶段十五周频任务失败: {exc}")
        return 2


def load_market_dates(connection: sqlite3.Connection) -> tuple[date, ...]:
    return tuple(
        date.fromisoformat(str(row[0]))
        for row in connection.execute("SELECT DISTINCT date FROM daily_bars ORDER BY date")
    )


def build_weekly_schedule(
    market_dates: Sequence[date],
    *,
    start_date: date | str,
    end_date: date | str,
    horizon_sessions: int = 5,
) -> tuple[WeeklySchedule, ...]:
    """Use the final observed market session in each ISO week as the signal day."""

    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    normalized_dates = tuple(sorted(set(market_dates)))
    if not normalized_dates:
        raise ValueError("市场交易日不能为空")
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date cannot be later than end_date")

    weekly_last: dict[tuple[int, int], int] = {}
    for index, trading_date in enumerate(normalized_dates):
        iso_year, iso_week, _ = trading_date.isocalendar()
        weekly_last[(iso_year, iso_week)] = index

    schedules: list[WeeklySchedule] = []
    for signal_index in sorted(weekly_last.values()):
        signal_date = normalized_dates[signal_index]
        if signal_date < start or signal_date > end:
            continue
        iso_year, iso_week, _ = signal_date.isocalendar()
        entry_index = signal_index + 1
        exit_index = signal_index + horizon_sessions
        schedules.append(
            WeeklySchedule(
                sequence=len(schedules) + 1,
                iso_year=iso_year,
                iso_week=iso_week,
                signal_date=signal_date,
                entry_date=(
                    normalized_dates[entry_index]
                    if entry_index < len(normalized_dates)
                    else None
                ),
                exit_date=(
                    normalized_dates[exit_index]
                    if exit_index < len(normalized_dates)
                    else None
                ),
            )
        )
    return tuple(schedules)


def calculate_tradable_label(
    symbol: str,
    schedule: WeeklySchedule,
    bars: dict[tuple[str, date], tuple[float, float, float | None]],
) -> TradableLabel:
    """Calculate next-session-open to fifth-session-close return on market dates."""

    if not schedule.label_complete:
        return TradableLabel(
            symbol=symbol,
            signal_date=schedule.signal_date,
            entry_date=schedule.entry_date,
            exit_date=schedule.exit_date,
            status="incomplete_market_horizon",
        )
    if (symbol, schedule.signal_date) not in bars:
        return _empty_label(symbol, schedule, "missing_signal_bar")
    entry = bars.get((symbol, schedule.entry_date))
    if entry is None:
        return _empty_label(symbol, schedule, "missing_entry_bar")
    exit_bar = bars.get((symbol, schedule.exit_date))
    if exit_bar is None:
        return _empty_label(symbol, schedule, "missing_exit_bar", entry=entry)

    entry_open, _, entry_amount = entry
    _, exit_close, exit_amount = exit_bar
    base = TradableLabel(
        symbol=symbol,
        signal_date=schedule.signal_date,
        entry_date=schedule.entry_date,
        exit_date=schedule.exit_date,
        status="tradable",
        entry_open=entry_open,
        exit_close=exit_close,
        entry_amount=entry_amount,
        exit_amount=exit_amount,
    )
    if entry_amount is None or entry_amount <= 0:
        return replace(base, status="entry_not_liquid")
    if exit_amount is None or exit_amount <= 0:
        return replace(base, status="exit_not_liquid")
    if not math.isfinite(entry_open) or not math.isfinite(exit_close) or entry_open <= 0 or exit_close <= 0:
        return replace(base, status="invalid_price")
    return replace(base, gross_return=exit_close / entry_open - 1.0)


def _empty_label(
    symbol: str,
    schedule: WeeklySchedule,
    status: str,
    *,
    entry: tuple[float, float, float | None] | None = None,
) -> TradableLabel:
    return TradableLabel(
        symbol=symbol,
        signal_date=schedule.signal_date,
        entry_date=schedule.entry_date,
        exit_date=schedule.exit_date,
        status=status,
        entry_open=entry[0] if entry else None,
        entry_amount=entry[2] if entry else None,
    )


def build_weekly_dataset(
    database_path: str | Path,
    stock_pool_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    horizon_sessions: int = 5,
) -> dict[str, object]:
    database = Path(database_path)
    symbols = load_stock_pool(stock_pool_path)
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        market_dates = load_market_dates(connection)
        schedule = build_weekly_schedule(
            market_dates,
            start_date=start_date,
            end_date=end_date,
            horizon_sessions=horizon_sessions,
        )
        bars = _load_bars(
            connection,
            symbols,
            min(item.signal_date for item in schedule),
            max((item.exit_date or item.signal_date) for item in schedule),
        )
        metadata = dict(connection.execute("SELECT key, value FROM dataset_meta"))

    labels = tuple(
        calculate_tradable_label(symbol, item, bars)
        for item in schedule
        for symbol in symbols
    )
    status_counts: dict[str, int] = {}
    for label in labels:
        status_counts[label.status] = status_counts.get(label.status, 0) + 1
    manifest = {
        "schema_version": "stage15-weekly-dataset-v1",
        "database": str(database),
        "source_hash": metadata.get("source_hash"),
        "reconstructed_source_hash": metadata.get("reconstructed_source_hash"),
        "feature_price_basis": metadata.get("price_basis"),
        "execution_price_basis": "raw_open_and_raw_close",
        "start_date": start_date,
        "end_date": end_date,
        "horizon_sessions": horizon_sessions,
        "signal_rule": "last_market_session_of_iso_week",
        "entry_rule": "next_market_session_raw_open",
        "exit_rule": f"market_session_t_plus_{horizon_sessions}_raw_close",
        "gross_return_formula": "exit_raw_close / entry_raw_open - 1",
        "stock_count": len(symbols),
        "weekly_signal_count": len(schedule),
        "label_count": len(labels),
        "tradable_label_count": status_counts.get("tradable", 0),
        "status_counts": status_counts,
        "limitations": [
            "Limit-up and limit-down execution checks are not available in the current daily table.",
            "Corporate-action total-return semantics still require manual source-field verification.",
            "This dataset contains gross price returns before commissions, stamp duty and slippage.",
        ],
    }
    return {"schedule": schedule, "labels": labels, "manifest": manifest}


def load_stock_pool(path: str | Path) -> tuple[str, ...]:
    file_path = Path(path)
    if not file_path.is_file():
        raise OSError(f"股票池不存在: {file_path}")
    with file_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    symbols = tuple(str(row.get("symbol", "")).strip() for row in rows)
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("股票池缺少有效 symbol")
    if len(set(symbols)) != len(symbols):
        raise ValueError("股票池包含重复 symbol")
    return symbols


def _load_bars(
    connection: sqlite3.Connection,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
) -> dict[tuple[str, date], tuple[float, float, float | None]]:
    placeholders = ",".join("?" for _ in symbols)
    rows = connection.execute(
        f"""
        SELECT symbol, date, open, close, amount
        FROM daily_bars
        WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
        """,
        (*symbols, start_date.isoformat(), end_date.isoformat()),
    )
    return {
        (str(symbol), date.fromisoformat(str(date_text))): (
            float(open_price),
            float(close_price),
            float(amount) if amount is not None else None,
        )
        for symbol, date_text, open_price, close_price, amount in rows
    }


def filter_labeled_matches(
    matches: Sequence[SimilarityMatch],
    *,
    query_date: date,
    schedule_by_date: dict[date, WeeklySchedule],
    labels: dict[tuple[str, date], TradableLabel],
    top_k: int,
) -> tuple[tuple[SimilarityMatch, TradableLabel], ...]:
    """Keep weekly candidates whose complete label was observable by query close."""

    eligible: list[tuple[SimilarityMatch, TradableLabel]] = []
    for match in matches:
        candidate_date = match.sample.date
        schedule = schedule_by_date.get(candidate_date)
        if schedule is None or schedule.exit_date is None or schedule.exit_date > query_date:
            continue
        label = labels.get((match.sample.symbol, candidate_date))
        if label is None or label.status != "tradable" or label.gross_return is None:
            continue
        eligible.append((replace(match, rank=len(eligible) + 1), label))
        if len(eligible) >= top_k:
            break
    return tuple(eligible)


def run_colproj_pilot(
    database_path: str | Path,
    stock_pool_path: str | Path,
    ann_manifest_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    horizon_sessions: int = 5,
    top_k: int = 20,
    overfetch_k: int = 500,
    candidate_k: int = 1000,
    nprobe: int = 64,
    minimum_matches: int = 20,
    max_queries: int | None = None,
    selected_symbols: Sequence[str] | None = None,
    config: ProjectConfig = DEFAULT_CONFIG,
    completed_keys: set[tuple[str, date]] | None = None,
    on_query: Callable[[dict[str, object], list[dict[str, object]]], None] | None = None,
    progress: bool = False,
) -> dict[str, object]:
    if (
        top_k <= 0
        or overfetch_k < top_k
        or minimum_matches <= 0
        or minimum_matches > top_k
    ):
        raise ValueError("top_k、overfetch_k 和 minimum_matches 参数无效")
    all_symbols = load_stock_pool(stock_pool_path)
    symbols = tuple(selected_symbols or all_symbols)
    unknown = sorted(set(symbols) - set(all_symbols))
    if unknown:
        raise ValueError(f"指定股票不在阶段十五股票池: {', '.join(unknown)}")

    database = Path(database_path)
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        market_dates = load_market_dates(connection)
        all_schedule = build_weekly_schedule(
            market_dates,
            start_date=market_dates[0],
            end_date=end_date,
            horizon_sessions=horizon_sessions,
        )
        query_schedule = tuple(
            item for item in all_schedule if _as_date(start_date) <= item.signal_date <= _as_date(end_date)
        )
        schedule_by_date = {item.signal_date: item for item in all_schedule}
        metadata = dict(connection.execute("SELECT key, value FROM dataset_meta"))

    market_data = MarketData.from_sqlite(database, symbols=symbols, price_basis="active")
    ann_index = MultiScaleANNIndex(
        ann_manifest_path,
        candidate_k=candidate_k,
        nprobe=nprobe,
    )
    if not ann_index.is_compatible(market_data.source_hash, config, price_basis=market_data.price_basis):
        raise ValueError("ANN索引与当前重建行情快照不兼容")

    predictions: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    label_cache: dict[tuple[str, date], TradableLabel] = {}
    previously_completed = completed_keys or set()
    processed = 0

    def record_query(
        prediction: dict[str, object],
        query_matches: list[dict[str, object]],
    ) -> None:
        predictions.append(prediction)
        match_rows.extend(query_matches)
        if on_query is not None:
            on_query(prediction, query_matches)
        if progress:
            print(
                f"progress: new={len(predictions)} symbol={prediction['symbol']} "
                f"date={prediction['signal_date']} status={prediction['status']} "
                f"ann_ms={prediction.get('ann_ms', '')}",
                flush=True,
            )

    try:
        with sqlite3.connect(uri, uri=True, timeout=60) as connection:
            for schedule in query_schedule:
                for symbol in symbols:
                    query_key = (symbol, schedule.signal_date)
                    if query_key in previously_completed:
                        continue
                    if max_queries is not None and processed >= max_queries:
                        break
                    processed += 1
                    try:
                        query = build_sample_embedding(
                            market_data,
                            symbol,
                            schedule.signal_date,
                            config=config,
                        )
                    except MarketDataError as exc:
                        record_query(
                            _failed_prediction(symbol, schedule, "query_unavailable", str(exc)),
                            [],
                        )
                        continue
                    matches, diagnostics = ann_index.search(
                        query,
                        market_data,
                        month_weight=config.month_weight,
                        week_weight=config.week_weight,
                        day_weight=config.day_weight,
                        top_k=overfetch_k,
                        eligible_candidate_dates={
                            item.signal_date
                            for item in all_schedule
                            if item.exit_date is not None
                            and item.exit_date <= schedule.signal_date
                        },
                    )
                    missing_keys = [
                        (match.sample.symbol, match.sample.date)
                        for match in matches
                        if (match.sample.symbol, match.sample.date) not in label_cache
                    ]
                    label_cache.update(
                        _load_candidate_labels(
                            connection,
                            missing_keys,
                            schedule_by_date,
                        )
                    )
                    eligible = filter_labeled_matches(
                        matches,
                        query_date=schedule.signal_date,
                        schedule_by_date=schedule_by_date,
                        labels=label_cache,
                        top_k=top_k,
                    )
                    if len(eligible) < minimum_matches:
                        record_query(
                            {
                                **_prediction_identity(symbol, schedule),
                                "status": "insufficient_eligible_matches",
                                "error": "",
                                "eligible_match_count": len(eligible),
                                **diagnostics,
                            },
                            [],
                        )
                        continue
                    selected_matches = tuple(item[0] for item in eligible)
                    selected_labels = tuple(item[1] for item in eligible)
                    weights = similarity_weights(selected_matches, config.similarity_temperature)
                    statistics = summarize_returns(
                        horizon_sessions,
                        (item.gross_return for item in selected_labels if item.gross_return is not None),
                        weights=weights,
                    )
                    actual = _load_candidate_labels(
                        connection,
                        [(symbol, schedule.signal_date)],
                        schedule_by_date,
                    ).get((symbol, schedule.signal_date))
                    prediction = {
                        **_prediction_identity(symbol, schedule),
                        "status": "ok",
                        "error": "",
                        "eligible_match_count": len(eligible),
                        "predicted_return": statistics.weighted_mean,
                        "weighted_std": statistics.weighted_std,
                        "signal_strength": statistics.signal_strength,
                        "positive_probability": statistics.weighted_win_rate,
                        "effective_sample_size": statistics.effective_sample_size,
                        "standard_error": statistics.standard_error,
                        "lower_bound": statistics.lower_bound,
                        "upper_bound": statistics.upper_bound,
                        "minimum_distance": min(item.distance for item in selected_matches),
                        "maximum_distance": max(item.distance for item in selected_matches),
                        "actual_status": actual.status if actual else "missing_label",
                        "actual_gross_return": actual.gross_return if actual else None,
                        **diagnostics,
                    }
                    current_match_rows: list[dict[str, object]] = []
                    for match, label, weight in zip(selected_matches, selected_labels, weights):
                        current_match_rows.append(
                            {
                                "query_symbol": symbol,
                                "query_date": schedule.signal_date.isoformat(),
                                "rank": match.rank,
                                "candidate_symbol": match.sample.symbol,
                                "candidate_date": match.sample.date.isoformat(),
                                "candidate_label_available_date": label.exit_date.isoformat(),
                                "distance": match.distance,
                                "monthly_distance": match.monthly_distance,
                                "weekly_distance": match.weekly_distance,
                                "daily_distance": match.daily_distance,
                                "similarity_weight": weight,
                                "candidate_gross_return": label.gross_return,
                            }
                        )
                    record_query(prediction, current_match_rows)
                if max_queries is not None and processed >= max_queries:
                    break
    finally:
        ann_index.close()

    manifest = {
        "schema_version": "stage15-colproj-pilot-v1",
        "database": str(database),
        "reconstructed_source_hash": metadata.get("reconstructed_source_hash"),
        "stock_pool": str(stock_pool_path),
        "ann_manifest": str(ann_manifest_path),
        "ann_version_id": ann_index.version_id,
        "model": "colproj",
        "start_date": start_date,
        "end_date": end_date,
        "symbols": list(symbols),
        "query_count": len(predictions),
        "successful_query_count": sum(row["status"] == "ok" for row in predictions),
        "horizon_sessions": horizon_sessions,
        "top_k": top_k,
        "overfetch_k": overfetch_k,
        "candidate_k": candidate_k,
        "nprobe": nprobe,
        "similarity_temperature": config.similarity_temperature,
        "scale_weights": {
            "monthly": config.month_weight,
            "weekly": config.week_weight,
            "daily": config.day_weight,
        },
        "candidate_rule": "weekly_signal_and_label_exit_not_later_than_query_date",
        "execution_return_rule": "raw_close_t_plus_5 / raw_open_t_plus_1 - 1",
    }
    return {"predictions": predictions, "matches": match_rows, "manifest": manifest}


def run_colproj_batch(
    database_path: str | Path,
    stock_pool_path: str | Path,
    ann_manifest_path: str | Path,
    store_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    horizon_sessions: int = 5,
    top_k: int = 20,
    overfetch_k: int = 500,
    candidate_k: int = 1000,
    nprobe: int = 64,
    minimum_matches: int = 20,
    max_new_queries: int | None = 100,
    selected_symbols: Sequence[str] | None = None,
    predictions_output: str | Path,
    matches_output: str | Path,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Run missing weekly queries and persist every completed key atomically."""

    if max_new_queries is not None and max_new_queries <= 0:
        raise ValueError("max_new_queries must be positive or None")
    all_symbols = load_stock_pool(stock_pool_path)
    symbols = tuple(selected_symbols or all_symbols)
    unknown = sorted(set(symbols) - set(all_symbols))
    if unknown:
        raise ValueError(f"指定股票不在阶段十五股票池: {', '.join(unknown)}")

    database = Path(database_path)
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        market_dates = load_market_dates(connection)
    query_schedule = build_weekly_schedule(
        market_dates,
        start_date=start_date,
        end_date=end_date,
        horizon_sessions=horizon_sessions,
    )
    target_count = len(query_schedule) * len(symbols)
    ann_metadata = read_ann_manifest_metadata(ann_manifest_path)
    if ann_metadata.model != "colproj":
        raise ValueError("阶段十五ColProj批处理要求使用ColProj ANN清单")

    parameters = {
        "horizon_sessions": horizon_sessions,
        "top_k": top_k,
        "overfetch_k": overfetch_k,
        "candidate_k": candidate_k,
        "nprobe": nprobe,
        "minimum_matches": minimum_matches,
        "similarity_temperature": config.similarity_temperature,
        "month_weight": config.month_weight,
        "week_weight": config.week_weight,
        "day_weight": config.day_weight,
        "chart_width": config.chart_width,
        "chart_height": config.chart_height,
        "candidate_rule": "weekly_signal_and_label_exit_not_later_than_query_date",
        "execution_return_rule": "raw_close_t_plus_5 / raw_open_t_plus_1 - 1",
    }
    scope = {
        "stock_pool": str(stock_pool_path),
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
    }
    store = Stage15PredictionStore(store_path)
    run_key = store.begin_run(
        model="colproj",
        index_version=ann_metadata.version_id,
        source_hash=ann_metadata.source_hash,
        parameters=parameters,
        scope=scope,
        target_count=target_count,
    )
    completed_before = store.completed_keys(run_key)
    started = perf_counter()
    try:
        result = run_colproj_pilot(
            database,
            stock_pool_path,
            ann_manifest_path,
            start_date=start_date,
            end_date=end_date,
            horizon_sessions=horizon_sessions,
            top_k=top_k,
            overfetch_k=overfetch_k,
            candidate_k=candidate_k,
            nprobe=nprobe,
            minimum_matches=minimum_matches,
            max_queries=max_new_queries,
            selected_symbols=symbols,
            config=config,
            completed_keys=completed_before,
            on_query=lambda prediction, matches: store.put_query(
                run_key, prediction, matches
            ),
            progress=True,
        )
        completed_count = len(store.completed_keys(run_key))
        status = "complete" if completed_count == target_count else "partial"
        store.finish_run(run_key, status=status)
    except Exception as exc:
        store.finish_run(run_key, status="failed", error=str(exc))
        raise

    store.export_csv(
        run_key,
        predictions_path=predictions_output,
        matches_path=matches_output,
    )
    run_status = store.run_status(run_key)
    predictions = store.load_predictions(run_key)
    return {
        **run_status,
        "store": str(store.path),
        "ann_manifest": str(ann_manifest_path),
        "newly_built": len(result["predictions"]),
        "successful_query_count": sum(row.get("status") == "ok" for row in predictions),
        "predictions_output": str(predictions_output),
        "matches_output": str(matches_output),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def _load_candidate_labels(
    connection: sqlite3.Connection,
    keys: Iterable[tuple[str, date]],
    schedule_by_date: dict[date, WeeklySchedule],
) -> dict[tuple[str, date], TradableLabel]:
    unique_keys = tuple(dict.fromkeys(keys))
    if not unique_keys:
        return {}
    relevant = tuple(
        key for key in unique_keys if key[1] in schedule_by_date
    )
    if not relevant:
        return {}
    symbols = tuple(sorted({symbol for symbol, _ in relevant}))
    schedules = tuple({schedule_by_date[signal_date] for _, signal_date in relevant})
    required_dates = {
        current
        for item in schedules
        for current in (item.signal_date, item.entry_date, item.exit_date)
        if current is not None
    }
    symbol_placeholders = ",".join("?" for _ in symbols)
    date_placeholders = ",".join("?" for _ in required_dates)
    rows = connection.execute(
        f"""
        SELECT symbol, date, open, close, amount
        FROM daily_bars
        WHERE symbol IN ({symbol_placeholders}) AND date IN ({date_placeholders})
        """,
        (*symbols, *(item.isoformat() for item in sorted(required_dates))),
    )
    bars = {
        (str(symbol), date.fromisoformat(str(date_text))): (
            float(open_price),
            float(close_price),
            float(amount) if amount is not None else None,
        )
        for symbol, date_text, open_price, close_price, amount in rows
    }
    return {
        key: calculate_tradable_label(key[0], schedule_by_date[key[1]], bars)
        for key in relevant
    }


def _prediction_identity(symbol: str, schedule: WeeklySchedule) -> dict[str, object]:
    return {
        "symbol": symbol,
        "signal_date": schedule.signal_date.isoformat(),
        "entry_date": schedule.entry_date.isoformat() if schedule.entry_date else "",
        "exit_date": schedule.exit_date.isoformat() if schedule.exit_date else "",
    }


def _failed_prediction(
    symbol: str,
    schedule: WeeklySchedule,
    status: str,
    error: str,
) -> dict[str, object]:
    return {**_prediction_identity(symbol, schedule), "status": status, "error": error}


def write_weekly_dataset(
    result: dict[str, object],
    *,
    calendar_path: str | Path,
    labels_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
) -> None:
    schedule = tuple(result["schedule"])
    labels = tuple(result["labels"])
    manifest = dict(result["manifest"])
    _write_csv(
        calendar_path,
        [
            {
                **asdict(item),
                "signal_date": item.signal_date.isoformat(),
                "entry_date": item.entry_date.isoformat() if item.entry_date else "",
                "exit_date": item.exit_date.isoformat() if item.exit_date else "",
                "label_complete": item.label_complete,
            }
            for item in schedule
        ],
    )
    _write_csv(
        labels_path,
        [
            {
                **asdict(item),
                "signal_date": item.signal_date.isoformat(),
                "entry_date": item.entry_date.isoformat() if item.entry_date else "",
                "exit_date": item.exit_date.isoformat() if item.exit_date else "",
                "label_available_date": item.available_date.isoformat() if item.available_date else "",
            }
            for item in labels
        ],
    )
    output_manifest = Path(manifest_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_report = Path(report_path)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_weekly_report(manifest), encoding="utf-8")


def write_prediction_pilot(
    result: dict[str, object],
    *,
    predictions_path: str | Path,
    matches_path: str | Path,
    manifest_path: str | Path,
) -> None:
    _write_csv(predictions_path, list(result["predictions"]))
    _write_csv(matches_path, list(result["matches"]))
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _render_weekly_report(manifest: dict[str, object]) -> str:
    statuses = manifest["status_counts"]
    status_lines = "\n".join(
        f"| `{status}` | {count} |" for status, count in sorted(statuses.items())
    )
    return f"""# 阶段十五周频日历与真实收益标签

## 已完成

- 信号区间：`{manifest['start_date']}` 至 `{manifest['end_date']}`。
- 每周信号日：该 ISO 周在数据库中的最后一个市场交易日，共 {manifest['weekly_signal_count']} 周。
- 股票池：固定 {manifest['stock_count']} 只，共生成 {manifest['label_count']} 条股票周标签。
- 可交易标签：{manifest['tradable_label_count']} 条。
- 入场价格：信号日后第1个市场交易日的原始开盘价。
- 退出价格：信号日后第5个市场交易日的原始收盘价。
- 毛收益：`exit_raw_close / entry_raw_open - 1`。
- 重建 OHLC 仍只用于网站形态图和检索特征，不作为真实成交价。

## 标签状态

| 状态 | 数量 |
| --- | ---: |
{status_lines}

## 防止未来数据泄漏

历史候选必须是同口径周末信号样本，并且候选的 `label_available_date` 不晚于当前查询信号日。候选日期早于查询日但未来5日结果尚未完整发生时，不允许参与加权预测。

## 当前限制

- 当前日行情表没有涨跌停状态，尚未完成无法按开盘价成交的精细规则。
- 公司行为和总回报字段语义仍需人工复核；当前结果标记为未扣费价格收益。
- 佣金、印花税和滑点将在组合交易模块中加入。
"""


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
