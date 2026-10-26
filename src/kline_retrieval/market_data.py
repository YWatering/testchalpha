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
    """One validated daily OHLC record."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


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

    return MarketBar(
        symbol=symbol,
        date=parsed_date,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        volume=volume,
    )


class MarketData:
    """In-memory, validated and date-indexed market data."""

    def __init__(
        self,
        bars: Sequence[MarketBar],
        *,
        source_path: str | None = None,
        source_hash: str | None = None,
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

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        source_path: str | None = None,
        source_hash: str | None = None,
    ) -> "MarketData":
        bars = [_normalize_row(row, row_number) for row_number, row in enumerate(rows, 1)]
        return cls(bars, source_path=source_path, source_hash=source_hash)

    @classmethod
    def from_csv(cls, path: str | Path) -> "MarketData":
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
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MarketData":
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
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "MarketData":
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return cls.from_csv(file_path)
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return cls.from_sqlite(file_path)
        if suffix in {".parquet", ".pq"}:
            return cls.from_parquet(file_path)
        raise MarketDataError(f"不支持的行情文件格式: {file_path.suffix or '<无扩展名>'}")

    @classmethod
    def from_sqlite(
        cls,
        path: str | Path,
        symbols: Sequence[str] | None = None,
    ) -> "MarketData":
        """Load a validated daily-bar snapshot from SQLite."""

        file_path = Path(path)
        if not file_path.is_file():
            raise MarketDataError(f"行情数据库不存在: {file_path}")
        try:
            with sqlite3.connect(file_path) as connection:
                if symbols:
                    placeholders = ",".join("?" for _ in symbols)
                    rows = connection.execute(
                        f"""
                        SELECT symbol, date, open, high, low, close, volume
                        FROM daily_bars
                        WHERE symbol IN ({placeholders})
                        ORDER BY symbol, date
                        """,
                        tuple(symbols),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT symbol, date, open, high, low, close, volume
                        FROM daily_bars
                        ORDER BY symbol, date
                        """
                    ).fetchall()
                metadata = dict(
                    connection.execute("SELECT key, value FROM dataset_meta").fetchall()
                )
        except sqlite3.Error as exc:
            raise MarketDataError(f"读取行情数据库失败: {file_path}") from exc
        if not rows:
            raise DataValidationError("行情数据库没有 daily_bars 数据")
        return cls.from_rows(
            (
                {
                    "symbol": row[0],
                    "date": row[1],
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "volume": row[6],
                }
                for row in rows
            ),
            source_path=str(file_path),
            source_hash=metadata.get("source_hash"),
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
