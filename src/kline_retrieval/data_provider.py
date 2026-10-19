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

    def __init__(self, fetcher: object | None = None, endpoint: str = "eastmoney") -> None:
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
        self._fetcher = fetcher
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
                prefix = "sh" if symbol.startswith("6") else "sz"
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
