"""Optional external market-data providers."""

from __future__ import annotations

import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, Sequence


class DataProviderError(RuntimeError):
    """Raised when a market-data provider is unavailable or returns bad data."""


class MarketDataProvider(Protocol):
    name: str
    version: str
    endpoint: str

    def fetch_daily(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: str,
    ) -> Sequence[dict[str, object]]:
        """Fetch daily rows using the provider's native column names."""


class AKShareProvider:
    """Fetch A-share daily bars through AKShare's historical A-share API."""

    name = "akshare"

    def __init__(
        self,
        fetcher: object | None = None,
        endpoint: str = "eastmoney",
        universe_fetcher: object | None = None,
    ) -> None:
        if sys.version_info < (3, 11):
            raise DataProviderError(
                "AKShare 下载需要 Python 3.11 或更高版本；核心检索仍可在 Python 3.10 运行"
            )
        if endpoint not in {"eastmoney", "sina"}:
            raise DataProviderError("AKShare endpoint 必须是 eastmoney 或 sina")
        self.endpoint = endpoint
        if fetcher is None:
            try:
                import akshare as ak
            except ImportError as exc:
                raise DataProviderError(
                    "未安装 AKShare。请使用 Python 3.11+ 执行: python -m pip install -e '.[a-share]'"
                ) from exc
            fetcher = ak.stock_zh_a_hist if endpoint == "eastmoney" else ak.stock_zh_a_daily
            universe_fetcher = universe_fetcher or ak.stock_info_a_code_name
        self._fetcher = fetcher
        self._universe_fetcher = universe_fetcher
        try:
            self.version = version("akshare")
        except PackageNotFoundError:
            self.version = "unknown"

    def fetch_daily(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: str,
    ) -> Sequence[dict[str, object]]:
        try:
            if self.endpoint == "eastmoney":
                frame = self._fetcher(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust=adjust,
                )
            else:
                if symbol.startswith("6"):
                    prefix = "sh"
                elif symbol.startswith(("4", "8", "9")):
                    prefix = "bj"
                else:
                    prefix = "sz"
                frame = self._fetcher(
                    symbol=f"{prefix}{symbol}",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust=adjust,
                )
        except Exception as exc:
            raise DataProviderError(f"AKShare 获取 {symbol} 日线失败: {exc}") from exc

        if frame is None or getattr(frame, "empty", False):
            return ()
        if not hasattr(frame, "to_dict"):
            raise DataProviderError(f"AKShare 返回了无法解析的 {symbol} 行情结果")
        try:
            return tuple(frame.to_dict(orient="records"))
        except Exception as exc:
            raise DataProviderError(f"AKShare 返回的 {symbol} 行情格式异常") from exc

    def list_a_share_symbols(self) -> tuple[str, ...]:
        """Discover currently listed Shanghai, Shenzhen and Beijing A shares."""

        return tuple(item["symbol"] for item in self.list_a_share_universe())

    def list_a_share_universe(self) -> tuple[dict[str, str], ...]:
        """Return normalized metadata for currently listed mainland A shares."""

        try:
            if self._universe_fetcher is None:
                import akshare as ak

                self._universe_fetcher = ak.stock_info_a_code_name
            frame = self._universe_fetcher()
        except Exception as exc:
            raise DataProviderError(f"AKShare 获取当前 A 股股票池失败: {exc}") from exc
        records: dict[str, str] = {}
        for row in frame.to_dict(orient="records"):
            raw_code = str(row.get("code", "")).strip()
            symbol = raw_code.zfill(6)
            name = str(row.get("name", "")).strip()
            if raw_code and symbol.isdigit() and symbol != "000000" and name:
                records[symbol] = name
        if not records:
            raise DataProviderError("AKShare 当前 A 股股票池为空")
        return tuple(
            {
                "symbol": symbol,
                "name": records[symbol],
                "exchange": _exchange_for_symbol(symbol),
                "board": _board_for_symbol(symbol),
                "status": "listed",
            }
            for symbol in sorted(records)
        )


def _exchange_for_symbol(symbol: str) -> str:
    if symbol.startswith("6"):
        return "SSE"
    if symbol.startswith(("4", "8", "9")):
        return "BSE"
    return "SZSE"


def _board_for_symbol(symbol: str) -> str:
    if symbol.startswith("688"):
        return "STAR"
    if symbol.startswith(("300", "301")):
        return "CHINEXT"
    if symbol.startswith(("4", "8", "9")):
        return "BSE"
    return "MAIN"
