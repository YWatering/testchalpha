"""Download and simulate a broad-market ETF benchmark for stage fifteen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .stage15_portfolio import (
    CostConfig,
    RawBar,
    build_equal_weight_benchmark,
    load_predictions,
    simulate_portfolio,
    write_result,
)


class ETFBenchmarkError(RuntimeError):
    """Raised when ETF benchmark data cannot be fetched or validated."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="510300")
    parser.add_argument("--name", default="沪深300ETF")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("research/stage15/colproj_oos_predictions.csv"),
    )
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-10-16")
    parser.add_argument("--download-start-date", default="2024-12-01")
    parser.add_argument("--download-end-date", default="2026-10-23")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--endpoint", choices=("eastmoney", "sina"), default="eastmoney")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--bars-output",
        type=Path,
        default=Path("research/stage15/csi300_etf_510300_daily.csv"),
    )
    parser.add_argument(
        "--data-manifest-output",
        type=Path,
        default=Path("research/stage15/csi300_etf_510300_manifest.json"),
    )
    parser.add_argument(
        "--weekly-output",
        type=Path,
        default=Path("research/stage15/csi300_etf_benchmark_weekly.csv"),
    )
    parser.add_argument(
        "--trades-output",
        type=Path,
        default=Path("research/stage15/csi300_etf_benchmark_trades.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("research/stage15/csi300_etf_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("research/stage15/CSI300_ETF_BENCHMARK_RESULTS.md"),
    )
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
        download_start = date.fromisoformat(args.download_start_date)
        download_end = date.fromisoformat(args.download_end_date)
        if not download_start <= start_date <= end_date <= download_end:
            raise ETFBenchmarkError("下载区间必须完整覆盖模拟区间")
        if args.initial_capital <= 0:
            raise ETFBenchmarkError("initial-capital 必须大于0")
        if args.retries <= 0 or args.retry_delay < 0:
            raise ETFBenchmarkError("retries必须大于0，retry-delay不能为负数")
        if args.refresh_data:
            rows, provider_version = fetch_etf_daily(
                args.symbol,
                download_start,
                download_end,
                endpoint=args.endpoint,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            write_etf_snapshot(
                rows,
                args.bars_output,
                args.data_manifest_output,
                symbol=args.symbol,
                name=args.name,
                provider_version=provider_version,
                endpoint=args.endpoint,
                requested_start=download_start,
                requested_end=download_end,
            )
        if not args.bars_output.exists():
            raise ETFBenchmarkError(
                f"ETF行情快照不存在: {args.bars_output}；请先使用 --refresh-data"
            )
        validate_snapshot_manifest(
            args.bars_output,
            args.data_manifest_output,
            args.symbol,
        )

        bars, market_dates, source_hash = load_etf_snapshot(
            args.bars_output,
            args.symbol,
        )
        predictions = load_predictions(args.predictions, args.start_date, args.end_date)
        benchmark = build_equal_weight_benchmark(predictions, (args.symbol,))
        result = simulate_portfolio(
            benchmark,
            bars,
            {args.symbol: "ETF"},
            market_dates,
            threshold=0.0,
            min_signal_strength=None,
            initial_capital=args.initial_capital,
            costs=CostConfig(
                commission_rate=args.commission_rate,
                stamp_duty_rate=0.0,
                slippage_rate=args.slippage_rate,
                minimum_commission=args.minimum_commission,
                lot_size=args.lot_size,
            ),
            rebalance=False,
            max_position_weight=1.0,
        )
        if len(result["weekly"]) != len(benchmark):
            raise ETFBenchmarkError(
                "ETF行情未完整覆盖预测周历，不能生成可比基准"
            )
        result["parameters"].update(
            {
                "selection_rule": "broad_market_etf_buy_and_hold",
                "threshold": None,
                "benchmark_symbol": args.symbol,
                "benchmark_name": args.name,
                "bars_sha256": source_hash,
                "price_adjustment": "raw",
            }
        )
        write_result(
            result,
            args.weekly_output,
            args.trades_output,
            args.manifest_output,
            args.report_output,
            args.report_output.with_suffix(".thresholds.csv"),
            threshold_grid=(),
            evaluation_period="out-of-sample",
        )
    except (ValueError, OSError, ETFBenchmarkError) as exc:
        print(f"ETF基准失败: {exc}")
        return 2

    print(f"symbol: {args.symbol}")
    print(f"weekly_periods: {len(result['weekly'])}")
    print(f"final_value: {result['metrics']['final_value']:.2f}")
    print(f"cumulative_return: {result['metrics']['cumulative_return']:.4f}")
    print(f"report: {args.report_output}")
    return 0


def fetch_etf_daily(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    fetcher: Callable[..., object] | None = None,
    endpoint: str = "eastmoney",
    retries: int = 1,
    retry_delay: float = 0.0,
) -> tuple[tuple[dict[str, object], ...], str]:
    if endpoint not in {"eastmoney", "sina"}:
        raise ETFBenchmarkError("ETF endpoint必须是eastmoney或sina")
    if fetcher is None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise ETFBenchmarkError("未安装AKShare，无法刷新ETF行情") from exc
        fetcher = (
            ak.fund_etf_hist_em
            if endpoint == "eastmoney"
            else ak.fund_etf_hist_sina
        )
    if retries <= 0 or retry_delay < 0:
        raise ETFBenchmarkError("retries必须大于0，retry_delay不能为负数")
    frame = None
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if endpoint == "eastmoney":
                frame = fetcher(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="",
                )
            else:
                prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
                frame = fetcher(symbol=f"{prefix}{symbol}")
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries and retry_delay:
                time.sleep(retry_delay * (attempt + 1))
    if frame is None:
        raise ETFBenchmarkError(f"AKShare获取ETF {symbol} 失败: {last_error}") from last_error
    if frame is None or getattr(frame, "empty", False) or not hasattr(frame, "to_dict"):
        raise ETFBenchmarkError(f"AKShare未返回ETF {symbol} 行情")
    rows = tuple(
        row
        for row in normalize_etf_rows(
            frame.to_dict(orient="records"),
            symbol,
            endpoint=(
                "fund_etf_hist_em"
                if endpoint == "eastmoney"
                else "fund_etf_hist_sina"
            ),
        )
        if start_date <= date.fromisoformat(str(row["date"])) <= end_date
    )
    if not rows:
        raise ETFBenchmarkError(f"ETF {symbol} 在请求区间内没有行情")
    try:
        provider_version = version("akshare")
    except PackageNotFoundError:
        provider_version = "unknown"
    return rows, provider_version


def normalize_etf_rows(
    records: Sequence[Mapping[str, object]],
    symbol: str,
    *,
    endpoint: str = "fund_etf_hist_em",
) -> tuple[dict[str, object], ...]:
    output: dict[date, dict[str, object]] = {}
    for index, row in enumerate(records, start=1):
        try:
            session = date.fromisoformat(str(_field(row, "日期", "date"))[:10])
            open_price = float(_field(row, "开盘", "open"))
            close = float(_field(row, "收盘", "close"))
            high = float(_field(row, "最高", "high"))
            low = float(_field(row, "最低", "low"))
            volume = float(_optional_field(row, "成交量", "volume") or 0)
            amount = float(_optional_field(row, "成交额", "amount") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ETFBenchmarkError(f"ETF第{index}条行情字段无法解析") from exc
        values = (open_price, high, low, close, volume, amount)
        if not all(math.isfinite(value) for value in values):
            raise ETFBenchmarkError(f"ETF第{index}条行情包含非有限值")
        if min(open_price, high, low, close) <= 0 or high < max(open_price, close) or low > min(open_price, close):
            raise ETFBenchmarkError(f"ETF第{index}条行情OHLC关系无效")
        output[session] = {
            "symbol": symbol,
            "date": session.isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "adjust": "raw",
            "provider": "akshare",
            "endpoint": endpoint,
        }
    if not output:
        raise ETFBenchmarkError("ETF行情为空")
    return tuple(output[session] for session in sorted(output))


def write_etf_snapshot(
    rows: Sequence[dict[str, object]],
    bars_path: str | Path,
    manifest_path: str | Path,
    *,
    symbol: str,
    name: str,
    provider_version: str,
    endpoint: str,
    requested_start: date,
    requested_end: date,
) -> None:
    output = Path(bars_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "stage15-etf-snapshot-v1",
        "symbol": symbol,
        "name": name,
        "provider": "akshare",
        "provider_version": provider_version,
        "endpoint": (
            "fund_etf_hist_em" if endpoint == "eastmoney" else "fund_etf_hist_sina"
        ),
        "adjust": "raw",
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "actual_start": rows[0]["date"],
        "actual_end": rows[-1]["date"],
        "row_count": len(rows),
        "sha256": source_hash,
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_etf_snapshot(
    path: str | Path,
    symbol: str,
) -> tuple[dict[tuple[str, date], RawBar], tuple[date, ...], str]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ETFBenchmarkError("ETF行情快照为空")
    bars: dict[tuple[str, date], RawBar] = {}
    previous_close: float | None = None
    sessions: list[date] = []
    for row in sorted(rows, key=lambda item: item["date"]):
        if row.get("symbol") != symbol:
            raise ETFBenchmarkError("ETF行情快照包含其他证券代码")
        session = date.fromisoformat(row["date"])
        bar = RawBar(
            symbol=symbol,
            session=session,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            pre_close=previous_close,
        )
        bars[(symbol, session)] = bar
        sessions.append(session)
        previous_close = bar.close
    return bars, tuple(sessions), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_snapshot_manifest(
    bars_path: str | Path,
    manifest_path: str | Path,
    symbol: str,
) -> None:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise ETFBenchmarkError(f"ETF行情来源清单不存在: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ETFBenchmarkError("ETF行情来源清单无法解析") from exc
    actual_hash = hashlib.sha256(Path(bars_path).read_bytes()).hexdigest()
    if manifest.get("symbol") != symbol:
        raise ETFBenchmarkError("ETF行情来源清单代码不匹配")
    if manifest.get("adjust") != "raw":
        raise ETFBenchmarkError("ETF宽基必须使用不复权原始价格快照")
    if manifest.get("sha256") != actual_hash:
        raise ETFBenchmarkError("ETF行情快照哈希与来源清单不一致")


def _field(row: Mapping[str, object], chinese: str, english: str) -> object:
    if chinese in row:
        return row[chinese]
    return row[english]


def _optional_field(row: Mapping[str, object], chinese: str, english: str) -> object:
    if chinese in row:
        return row[chinese]
    return row.get(english, 0)


if __name__ == "__main__":
    raise SystemExit(main())
