"""Validated market-data access for the staged K-line retrieval project.

The module deliberately keeps the first data layer dependency-free. CSV files
work out of the box; Parquet support is available when pandas and pyarrow are
installed in the runtime environment.
"""

from __future__ import annotations

import csv
import hashlib
import math
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_COLUMNS = ("symbol", "date", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume",)


class MarketDataError(ValueError):
    """Base exception for invalid or unusable market data."""


class DataValidationError(MarketDataError):
    """Raised when one or more input rows fail validation."""


class UnknownSymbolError(MarketDataError):
    """Raised when a request references a symbol that is not loaded."""


class UnknownTradingDateError(MarketDataError):
    """Raised when a request references a date absent from a symbol's data."""


class InsufficientHistoryError(MarketDataError):
    """Raised when a sample does not have enough historical observations."""


class InsufficientFutureError(MarketDataError):
    """Raised when a sample does not have enough future observations."""


@dataclass(frozen=True, slots=True)
class MarketBar:
    """One validated daily OHLC record and its optional source metadata."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    return_rate: float | None = None
    pre_close: float | None = None
    raw_open: float | None = None
    raw_high: float | None = None
    raw_low: float | None = None
    raw_close: float | None = None
    price_basis: str = "raw"
    reconstruction_status: str = "raw"
    reconstruction_scale: float | None = None
    reconstruction_anchor_date: date | None = None
    reconstruction_anchor_close: float | None = None

    def __post_init__(self) -> None:
        for field in ("raw_open", "raw_high", "raw_low", "raw_close"):
            if getattr(self, field) is None:
                object.__setattr__(self, field, float(getattr(self, field[4:])))
        if self.price_basis not in {"raw", "reconstructed"}:
            raise DataValidationError(f"不支持的价格口径: {self.price_basis}")

    @property
    def label_close(self) -> float:
        """Return the active close used by the current research price basis."""

        return self.close


@dataclass(frozen=True, slots=True)
class SampleValidation:
    """Validation details for a potential query or candidate sample."""

    symbol: str
    date: date
    valid: bool
    history_available: int
    future_available: int
    reasons: tuple[str, ...]


def _parse_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        raise DataValidationError("date 不能为空")

    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise DataValidationError(f"date 不是有效的 ISO 日期: {text!r}") from exc


def _parse_number(value: object, field: str, row_number: int) -> float:
    if value is None or str(value).strip() == "":
        raise DataValidationError(f"第 {row_number} 行的 {field} 不能为空")

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"第 {row_number} 行的 {field} 不是有效数字: {value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise DataValidationError(f"第 {row_number} 行的 {field} 必须是有限数字")
    return parsed


def _normalize_row(row: Mapping[str, object], row_number: int) -> MarketBar:
    values = {str(key).strip().lower(): value for key, value in row.items()}
    missing = [field for field in REQUIRED_COLUMNS if field not in values]
    if missing:
        raise DataValidationError(
            f"第 {row_number} 行缺少必要字段: {', '.join(missing)}"
        )

    symbol = str(values["symbol"]).strip()
    if not symbol:
        raise DataValidationError(f"第 {row_number} 行的 symbol 不能为空")

    parsed_date = _parse_date(values["date"])
    prices = {
        field: _parse_number(values[field], field, row_number)
        for field in ("open", "high", "low", "close")
    }

    if any(value <= 0 for value in prices.values()):
        raise DataValidationError(f"第 {row_number} 行的 OHLC 价格必须大于 0")
    if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
        raise DataValidationError(f"第 {row_number} 行的 high 小于 OHLC 其他价格")
    if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
        raise DataValidationError(f"第 {row_number} 行的 low 大于 OHLC 其他价格")

    volume: float | None = None
    if "volume" in values and values["volume"] is not None:
        raw_volume = str(values["volume"]).strip()
        if raw_volume:
            volume = _parse_number(values["volume"], "volume", row_number)
            if volume < 0:
                raise DataValidationError(f"第 {row_number} 行的 volume 不能小于 0")

    def optional_number(field: str) -> float | None:
        raw = values.get(field)
        if raw is None or str(raw).strip() == "":
            return None
        return _parse_number(raw, field, row_number)

    raw_prices = {
        field: optional_number(f"raw_{field}")
        for field in ("open", "high", "low", "close")
    }
    anchor_date = values.get("reconstruction_anchor_date")

    return MarketBar(
        symbol=symbol,
        date=parsed_date,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        volume=volume,
        return_rate=optional_number("return_rate"),
        pre_close=optional_number("pre_close"),
        raw_open=raw_prices["open"],
        raw_high=raw_prices["high"],
        raw_low=raw_prices["low"],
        raw_close=raw_prices["close"],
        price_basis=str(values.get("price_basis", "raw")),
        reconstruction_status=str(values.get("reconstruction_status", "raw")),
        reconstruction_scale=optional_number("reconstruction_scale"),
        reconstruction_anchor_date=(
            _parse_date(anchor_date) if anchor_date not in (None, "") else None
        ),
        reconstruction_anchor_close=optional_number("reconstruction_anchor_close"),
    )


class MarketData:
    """In-memory, validated and date-indexed market data."""

    def __init__(
        self,
        bars: Sequence[MarketBar],
        *,
        source_path: str | None = None,
        source_hash: str | None = None,
        price_basis: str = "raw",
    ) -> None:
        if not bars:
            raise DataValidationError("行情数据不能为空")

        by_symbol: dict[str, list[MarketBar]] = {}
        seen: set[tuple[str, date]] = set()
        for bar in bars:
            key = (bar.symbol, bar.date)
            if key in seen:
                raise DataValidationError(
                    f"发现重复行情记录: symbol={bar.symbol}, date={bar.date.isoformat()}"
                )
            seen.add(key)
            by_symbol.setdefault(bar.symbol, []).append(bar)

        self._by_symbol = {
            symbol: tuple(sorted(symbol_bars, key=lambda item: item.date))
            for symbol, symbol_bars in by_symbol.items()
        }
        self._dates = {
            symbol: tuple(bar.date for bar in symbol_bars)
            for symbol, symbol_bars in self._by_symbol.items()
        }
        self.source_path = source_path
        self.source_hash = source_hash
        self.price_basis = price_basis

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        source_path: str | None = None,
        source_hash: str | None = None,
        price_basis: str = "raw",
    ) -> "MarketData":
        bars = [_normalize_row(row, row_number) for row_number, row in enumerate(rows, 1)]
        return cls(
            bars,
            source_path=source_path,
            source_hash=source_hash,
            price_basis=price_basis,
        )

    @classmethod
    def from_csv(cls, path: str | Path, *, price_basis: str = "raw") -> "MarketData":
        file_path = Path(path)
        if not file_path.is_file():
            raise MarketDataError(f"行情文件不存在: {file_path}")

        raw_bytes = file_path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise MarketDataError(f"行情文件不是 UTF-8 编码: {file_path}") from exc

        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise DataValidationError("CSV 文件缺少表头")
        normalized_fields = {field.strip().lower() for field in reader.fieldnames}
        missing = [field for field in REQUIRED_COLUMNS if field not in normalized_fields]
        if missing:
            raise DataValidationError(f"CSV 文件缺少必要字段: {', '.join(missing)}")

        return cls.from_rows(
            reader,
            source_path=str(file_path),
            source_hash=hashlib.sha256(raw_bytes).hexdigest(),
            price_basis=price_basis,
        )

    @classmethod
    def from_parquet(cls, path: str | Path, *, price_basis: str = "raw") -> "MarketData":
        """Load Parquet data through optional pandas/pyarrow dependencies."""

        file_path = Path(path)
        if not file_path.is_file():
            raise MarketDataError(f"行情文件不存在: {file_path}")

        try:
            import pandas as pd
        except ImportError as exc:
            raise MarketDataError(
                "读取 Parquet 需要安装 pandas 和 pyarrow；当前 CSV 加载无需额外依赖"
            ) from exc

        try:
            frame = pd.read_parquet(file_path)
        except Exception as exc:  # pandas exposes engine-specific exceptions.
            raise MarketDataError(f"读取 Parquet 失败: {file_path}") from exc

        raw_bytes = file_path.read_bytes()
        return cls.from_rows(
            frame.to_dict(orient="records"),
            source_path=str(file_path),
            source_hash=hashlib.sha256(raw_bytes).hexdigest(),
            price_basis=price_basis,
        )

    @classmethod
    def from_file(cls, path: str | Path, *, price_basis: str = "active") -> "MarketData":
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            if price_basis not in {"active", "raw"}:
                raise MarketDataError("CSV 文件不包含预计算的重建 OHLC；请使用 raw 口径")
            return cls.from_csv(file_path)
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return cls.from_sqlite(file_path, price_basis=price_basis)
        if suffix in {".parquet", ".pq"}:
            if price_basis not in {"active", "raw"}:
                raise MarketDataError("Parquet 文件不包含预计算的重建 OHLC；请使用 raw 口径")
            return cls.from_parquet(file_path)
        raise MarketDataError(f"不支持的行情文件格式: {file_path.suffix or '<无扩展名>'}")

    @classmethod
    def from_sqlite(
        cls,
        path: str | Path,
        symbols: Sequence[str] | None = None,
        *,
        price_basis: str = "active",
    ) -> "MarketData":
        """Load a validated daily-bar snapshot from SQLite."""

        file_path = Path(path)
        if not file_path.is_file():
            raise MarketDataError(f"行情数据库不存在: {file_path}")
        loaded_source_hash = None
        try:
            with sqlite3.connect(file_path) as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM dataset_meta").fetchall()
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(daily_bars)")
                }
                selected_basis = metadata.get("price_basis") == "return_reconstructed_v1" if price_basis == "active" else price_basis == "reconstructed"
                if price_basis not in {"active", "raw", "reconstructed"}:
                    raise MarketDataError(f"不支持的价格口径: {price_basis}")
                if selected_basis:
                    required = {"reconstructed_open", "reconstructed_high", "reconstructed_low", "reconstructed_close"}
                    if not required.issubset(columns):
                        raise MarketDataError("数据库尚未生成收益率重建 OHLC；请先运行 kline-reconstruct-prices")
                    if metadata.get("price_basis") != "return_reconstructed_v1":
                        raise MarketDataError("数据库未激活收益率重建价格口径")
                    loaded_source_hash = metadata.get("reconstructed_source_hash")
                    select_fields = """symbol, date, reconstructed_open, reconstructed_high,
                        reconstructed_low, reconstructed_close, volume, return_rate, pre_close,
                        open, high, low, close, reconstruction_scale, reconstruction_status,
                        reconstruction_anchor_date, reconstruction_anchor_close"""
                else:
                    select_fields = """symbol, date, open, high, low, close, volume,
                        return_rate, pre_close"""
                where = ""
                parameters: tuple[object, ...] = ()
                if symbols:
                    placeholders = ",".join("?" for _ in symbols)
                    where = f" WHERE symbol IN ({placeholders})"
                    parameters = tuple(symbols)
                rows = connection.execute(
                    f"SELECT {select_fields} FROM daily_bars{where} ORDER BY symbol, date",
                    parameters,
                ).fetchall()
        except sqlite3.Error as exc:
            raise MarketDataError(f"读取行情数据库失败: {file_path}") from exc
        if not rows:
            raise DataValidationError("行情数据库没有 daily_bars 数据")
        if selected_basis:
            normalized_rows = (
                {
                    "symbol": row[0], "date": row[1], "open": row[2], "high": row[3],
                    "low": row[4], "close": row[5], "volume": row[6],
                    "return_rate": row[7], "pre_close": row[8],
                    "raw_open": row[9], "raw_high": row[10], "raw_low": row[11],
                    "raw_close": row[12], "price_basis": "reconstructed",
                    "reconstruction_scale": row[13], "reconstruction_status": row[14],
                    "reconstruction_anchor_date": row[15],
                    "reconstruction_anchor_close": row[16],
                }
                for row in rows
            )
            loaded_basis = "reconstructed"
        else:
            normalized_rows = (
                {
                    "symbol": row[0], "date": row[1], "open": row[2], "high": row[3],
                    "low": row[4], "close": row[5], "volume": row[6],
                    "return_rate": row[7], "pre_close": row[8],
                }
                for row in rows
            )
            loaded_basis = "raw"
        return cls.from_rows(
            normalized_rows,
            source_path=str(file_path),
            source_hash=loaded_source_hash or metadata.get("source_hash"),
            price_basis=loaded_basis,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_symbol))

    def bars(self, symbol: str) -> tuple[MarketBar, ...]:
        """Return all loaded daily bars for one symbol."""

        self._require_symbol(symbol)
        return self._by_symbol[symbol]

    def dates(self, symbol: str) -> tuple[date, ...]:
        self._require_symbol(symbol)
        return self._dates[symbol]

    def get_bar(self, symbol: str, trading_date: date | str) -> MarketBar:
        index = self._date_index(symbol, trading_date)
        return self._by_symbol[symbol][index]

    def get_history(
        self,
        symbol: str,
        trading_date: date | str,
        count: int,
    ) -> tuple[MarketBar, ...]:
        """Return ``count`` trading bars ending on ``trading_date``."""

        self._validate_count(count)
        index = self._date_index(symbol, trading_date)
        start = index - count + 1
        if start < 0:
            available = index + 1
            raise InsufficientHistoryError(
                f"样本历史数据不足: symbol={symbol}, date={self._dates[symbol][index]}, "
                f"需要 {count} 条，实际可用 {available} 条"
            )
        return self._by_symbol[symbol][start : index + 1]

    def get_future(
        self,
        symbol: str,
        trading_date: date | str,
        count: int,
    ) -> tuple[MarketBar, ...]:
        """Return the next ``count`` trading bars strictly after ``trading_date``."""

        self._validate_count(count)
        index = self._date_index(symbol, trading_date)
        end = index + count + 1
        if end > len(self._by_symbol[symbol]):
            available = len(self._by_symbol[symbol]) - index - 1
            raise InsufficientFutureError(
                f"样本未来数据不足: symbol={symbol}, date={self._dates[symbol][index]}, "
                f"需要 {count} 条，实际可用 {available} 条"
            )
        return self._by_symbol[symbol][index + 1 : end]

    def validate_sample(
        self,
        symbol: str,
        trading_date: date | str,
        *,
        history_count: int = 240,
        future_count: int = 21,
    ) -> SampleValidation:
        """Return validation details without raising for expected data gaps."""

        parsed_date = _parse_date(trading_date)
        reasons: list[str] = []
        if symbol not in self._by_symbol:
            return SampleValidation(
                symbol=symbol,
                date=parsed_date,
                valid=False,
                history_available=0,
                future_available=0,
                reasons=(f"未知股票代码: {symbol}",),
            )

        dates = self._dates[symbol]
        index = bisect_left(dates, parsed_date)
        if index == len(dates) or dates[index] != parsed_date:
            return SampleValidation(
                symbol=symbol,
                date=parsed_date,
                valid=False,
                history_available=index,
                future_available=len(dates) - index,
                reasons=(f"不是有效交易日: {parsed_date.isoformat()}",),
            )

        history_available = index + 1
        future_available = len(dates) - index - 1
        if history_available < history_count:
            reasons.append(f"历史数据不足，需要 {history_count} 条，实际 {history_available} 条")
        if future_available < future_count:
            reasons.append(f"未来数据不足，需要 {future_count} 条，实际 {future_available} 条")

        return SampleValidation(
            symbol=symbol,
            date=parsed_date,
            valid=not reasons,
            history_available=history_available,
            future_available=future_available,
            reasons=tuple(reasons),
        )

    def require_valid_sample(
        self,
        symbol: str,
        trading_date: date | str,
        *,
        history_count: int = 240,
        future_count: int = 21,
    ) -> SampleValidation:
        result = self.validate_sample(
            symbol,
            trading_date,
            history_count=history_count,
            future_count=future_count,
        )
        if not result.valid:
            raise MarketDataError("样本无效: " + "; ".join(result.reasons))
        return result

    def _require_symbol(self, symbol: str) -> None:
        if symbol not in self._by_symbol:
            raise UnknownSymbolError(f"未知股票代码: {symbol}")

    def _date_index(self, symbol: str, trading_date: date | str) -> int:
        self._require_symbol(symbol)
        parsed_date = _parse_date(trading_date)
        dates = self._dates[symbol]
        index = bisect_left(dates, parsed_date)
        if index == len(dates) or dates[index] != parsed_date:
            raise UnknownTradingDateError(
                f"{symbol} 不存在交易日: {parsed_date.isoformat()}"
            )
        return index

    @staticmethod
    def _validate_count(count: int) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count 必须是大于 0 的整数")
