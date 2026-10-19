"""Normalize, validate and persist locally fetched A-share daily history."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .data_provider import DataProviderError, MarketDataProvider
from .market_data import DataValidationError, MarketData


CSV_FIELDS = ("symbol", "date", "open", "high", "low", "close", "volume")
PROVIDER_FIELDS = {
    "date": "日期",
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "close": "收盘",
    "volume": "成交量",
}
SINA_PROVIDER_FIELDS = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}
DEFAULT_A_SHARE_SYMBOLS = ("000001", "600519", "300750")
VALID_ADJUSTMENTS = ("", "qfq", "hfq")


class DataSyncError(RuntimeError):
    """Raised when a download cannot safely be committed as a dataset."""


@dataclass(frozen=True, slots=True)
class DataSyncResult:
    data_path: Path
    manifest_path: Path
    symbol_count: int
    row_count: int
    start_date: date
    end_date: date
    source_hash: str


def normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(symbol).strip() for symbol in symbols))
    if not normalized or any(len(symbol) != 6 or not symbol.isdigit() for symbol in normalized):
        raise DataSyncError("A 股代码必须是六位数字，例如 000001,600519,300750")
    return normalized


def normalize_provider_rows(
    symbol: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, 1):
        source_fields = (
            PROVIDER_FIELDS
            if all(source in row for source in PROVIDER_FIELDS.values())
            else SINA_PROVIDER_FIELDS
        )
        missing = [source for source in source_fields.values() if source not in row]
        if missing:
            raise DataSyncError(
                f"AKShare {symbol} 第 {row_number} 条记录缺少字段: {', '.join(missing)}"
            )
        try:
            trading_date = _normalize_date(row[source_fields["date"]])
            values = {
                field: float(row[source])
                for field, source in source_fields.items()
                if field != "date"
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataSyncError(f"AKShare {symbol} 第 {row_number} 条记录包含无效值") from exc
        normalized.append(
            {
                "symbol": symbol,
                "date": trading_date.isoformat(),
                **values,
            }
        )
    return normalized


def sync_a_share_daily(
    provider: MarketDataProvider,
    symbols: Iterable[str],
    start_date: date,
    end_date: date,
    output_path: str | Path,
    *,
    adjust: str = "qfq",
    incremental: bool = True,
    refresh_overlap_days: int = 7,
    request_delay: float = 0.8,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> DataSyncResult:
    """Fetch and atomically persist a validated, deduplicated daily dataset."""

    normalized_symbols = normalize_symbols(symbols)
    if start_date > end_date:
        raise DataSyncError("开始日期不能晚于结束日期")
    if adjust not in VALID_ADJUSTMENTS:
        raise DataSyncError("adjust 必须是空字符串、qfq 或 hfq")
    if refresh_overlap_days < 0 or request_delay < 0 or retries < 1:
        raise DataSyncError("重试次数必须为正数，刷新窗口和请求间隔不能为负数")

    data_path = Path(output_path)
    manifest_path = _manifest_path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, object]] = []
    old_manifest: dict[str, object] | None = None
    if incremental and data_path.exists():
        if not manifest_path.is_file():
            raise DataSyncError(
                f"已有数据文件缺少来源清单: {manifest_path}；使用 --full-refresh 或新输出路径"
            )
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old_manifest.get("provider") != provider.name:
                raise DataSyncError("已有数据来源不同；请使用 --full-refresh 重新生成")
            if old_manifest.get("endpoint", "eastmoney") != provider.endpoint:
                raise DataSyncError("已有数据 AKShare 端点不同；请使用 --full-refresh 重新生成")
            if old_manifest.get("adjust") != (adjust or "raw"):
                raise DataSyncError("已有数据复权口径不同；请使用 --full-refresh 重新生成")
            actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
            if old_manifest.get("sha256") != actual_hash:
                raise DataSyncError("现有行情文件与数据清单哈希不一致；请使用 --full-refresh")
            old_rows = _read_csv_rows(data_path)
            MarketData.from_rows(old_rows)
        except DataSyncError:
            raise
        except (OSError, json.JSONDecodeError, DataValidationError) as exc:
            raise DataSyncError(f"无法读取现有行情快照: {exc}") from exc

    ranges_by_symbol = _date_ranges(old_rows)
    fetched_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(normalized_symbols):
        fetch_start = start_date
        if incremental and symbol in ranges_by_symbol:
            earliest, latest = ranges_by_symbol[symbol]
            if start_date < earliest:
                fetch_start = start_date
            else:
                fetch_start = max(start_date, latest - timedelta(days=refresh_overlap_days))
        if fetch_start > end_date:
            continue

        raw_rows: Sequence[dict[str, object]] = ()
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                raw_rows = provider.fetch_daily(symbol, fetch_start, end_date, adjust)
                last_error = None
                break
            except DataProviderError as exc:
                last_error = exc
                if attempt + 1 < retries:
                    sleep(min(2**attempt, 8))
        if last_error is not None:
            raise DataSyncError(str(last_error)) from last_error
        if not raw_rows:
            raise DataSyncError(f"AKShare 未返回 {symbol} 在请求区间内的日线数据")
        fetched_rows.extend(normalize_provider_rows(symbol, raw_rows))
        if symbol_index + 1 < len(normalized_symbols) and request_delay:
            sleep(request_delay)

    merged = _merge_rows(old_rows if incremental else (), fetched_rows)
    if not merged:
        raise DataSyncError("没有可保存的行情数据")
    try:
        validated = MarketData.from_rows(merged)
    except DataValidationError as exc:
        raise DataSyncError(f"下载数据未通过行情校验，未覆盖现有文件: {exc}") from exc

    csv_bytes = _serialize_csv(merged)
    source_hash = hashlib.sha256(csv_bytes).hexdigest()
    all_dates = [_normalize_date(row["date"]) for row in merged]
    manifest: dict[str, object] = {
        "provider": provider.name,
        "endpoint": provider.endpoint,
        "provider_version": provider.version,
        "adjust": adjust or "raw",
        "symbols": list(validated.symbols),
        "requested_symbols": list(normalized_symbols),
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "data_start_date": min(all_dates).isoformat(),
        "data_end_date": max(all_dates).isoformat(),
        "row_count": len(merged),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": source_hash,
    }
    _atomic_write(data_path, csv_bytes)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return DataSyncResult(
        data_path=data_path,
        manifest_path=manifest_path,
        symbol_count=len(validated.symbols),
        row_count=len(merged),
        start_date=min(all_dates),
        end_date=max(all_dates),
        source_hash=source_hash,
    )


def _normalize_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10] if "-" in pattern else text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"无效日期: {text}")


def _manifest_path(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.stem}.manifest.json")


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _date_ranges(rows: Sequence[dict[str, object]]) -> dict[str, tuple[date, date]]:
    ranges: dict[str, tuple[date, date]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        current = _normalize_date(row["date"])
        if symbol not in ranges:
            ranges[symbol] = (current, current)
        else:
            start, end = ranges[symbol]
            ranges[symbol] = (min(start, current), max(end, current))
    return ranges


def _merge_rows(
    previous: Sequence[dict[str, object]],
    fetched: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in previous:
        by_key[(str(row["symbol"]), str(row["date"])[:10])] = _standard_row(row)
    for row in fetched:
        by_key[(str(row["symbol"]), str(row["date"])[:10])] = _standard_row(row)
    return [by_key[key] for key in sorted(by_key)]


def _standard_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "symbol": str(row["symbol"]),
        "date": _normalize_date(row["date"]).isoformat(),
        **{field: float(row[field]) for field in ("open", "high", "low", "close", "volume")},
    }


def _serialize_csv(rows: Sequence[dict[str, object]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
