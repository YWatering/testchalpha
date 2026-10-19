"""Multi-scale OHLC aggregation and dependency-free grayscale rendering."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketBar, MarketData


SCALE_NAMES = ("monthly", "weekly", "daily", "future_12d")


class ChartError(ValueError):
    """Raised when OHLC data cannot be aggregated or rendered."""


@dataclass(frozen=True, slots=True)
class OHLCBar:
    """An OHLC bar aggregated from one or more daily observations."""

    start_date: date
    end_date: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class GrayscaleImage:
    """Immutable 8-bit grayscale image with PNG and PGM serialization."""

    width: int
    height: int
    pixels: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ChartError("图像宽度和高度必须大于 0")
        if len(self.pixels) != self.height:
            raise ChartError("图像像素行数与 height 不一致")
        if any(len(row) != self.width for row in self.pixels):
            raise ChartError("图像像素列数与 width 不一致")
        if any(pixel not in (0, 255) for row in self.pixels for pixel in row):
            raise ChartError("当前图像只允许黑白像素 0 或 255")

    @property
    def black_pixel_count(self) -> int:
        return sum(pixel == 0 for row in self.pixels for pixel in row)

    def to_pgm_bytes(self) -> bytes:
        rows = b"\n".join(bytes(row) for row in self.pixels)
        return f"P5\n{self.width} {self.height}\n255\n".encode("ascii") + rows

    def to_png_bytes(self) -> bytes:
        raw_scanlines = b"".join(b"\x00" + bytes(row) for row in self.pixels)
        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 0, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + b"".join(
            (
                _png_chunk(b"IHDR", ihdr),
                _png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=9)),
                _png_chunk(b"IEND", b""),
            )
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.to_png_bytes())
        return output_path


@dataclass(frozen=True, slots=True)
class SampleBars:
    """OHLC bars for query scales and the non-query future comparison."""

    monthly: tuple[OHLCBar, ...]
    weekly: tuple[OHLCBar, ...]
    daily: tuple[OHLCBar, ...]
    future_12d: tuple[OHLCBar, ...]

    def as_dict(self) -> dict[str, tuple[OHLCBar, ...]]:
        return {
            "monthly": self.monthly,
            "weekly": self.weekly,
            "daily": self.daily,
            "future_12d": self.future_12d,
        }


@dataclass(frozen=True, slots=True)
class QueryBars:
    """Only the three historical scales used by feature extraction."""

    monthly: tuple[OHLCBar, ...]
    weekly: tuple[OHLCBar, ...]
    daily: tuple[OHLCBar, ...]

    def as_dict(self) -> dict[str, tuple[OHLCBar, ...]]:
        return {
            "monthly": self.monthly,
            "weekly": self.weekly,
            "daily": self.daily,
        }


@dataclass(frozen=True, slots=True)
class SampleCharts:
    """Rendered charts for a query sample."""

    monthly: GrayscaleImage
    weekly: GrayscaleImage
    daily: GrayscaleImage
    future_12d: GrayscaleImage

    def as_dict(self) -> dict[str, GrayscaleImage]:
        return {
            "monthly": self.monthly,
            "weekly": self.weekly,
            "daily": self.daily,
            "future_12d": self.future_12d,
        }


@dataclass(frozen=True, slots=True)
class QueryCharts:
    """Rendered historical charts used for retrieval features."""

    monthly: GrayscaleImage
    weekly: GrayscaleImage
    daily: GrayscaleImage

    def as_dict(self) -> dict[str, GrayscaleImage]:
        return {
            "monthly": self.monthly,
            "weekly": self.weekly,
            "daily": self.daily,
        }


def aggregate_ohlc(bars: Sequence[MarketBar], group_size: int) -> tuple[OHLCBar, ...]:
    """Aggregate consecutive daily bars into fixed-size OHLC bars."""

    if group_size <= 0:
        raise ChartError("group_size 必须大于 0")
    if not bars:
        raise ChartError("不能聚合空的 OHLC 数据")
    if len(bars) % group_size:
        raise ChartError(
            f"OHLC 数据长度 {len(bars)} 不能被分组大小 {group_size} 整除"
        )

    aggregated: list[OHLCBar] = []
    for start in range(0, len(bars), group_size):
        group = bars[start : start + group_size]
        aggregated.append(
            OHLCBar(
                start_date=group[0].date,
                end_date=group[-1].date,
                open=group[0].open,
                high=max(bar.high for bar in group),
                low=min(bar.low for bar in group),
                close=group[-1].close,
            )
        )
    return tuple(aggregated)


def build_sample_bars(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> SampleBars:
    """Build 12 bars for each query scale and 12 future daily bars."""

    monthly_daily = market_data.get_history(
        symbol, trading_date, config.monthly_period * config.daily_window
    )
    weekly_daily = market_data.get_history(
        symbol, trading_date, config.weekly_period * config.daily_window
    )
    daily_daily = market_data.get_history(symbol, trading_date, config.daily_window)
    future_bars = build_future_bars(market_data, symbol, trading_date, config)

    return SampleBars(
        monthly=aggregate_ohlc(monthly_daily, config.monthly_period),
        weekly=aggregate_ohlc(weekly_daily, config.weekly_period),
        daily=aggregate_ohlc(daily_daily, 1),
        future_12d=future_bars,
    )


def build_query_bars(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> QueryBars:
    """Build only historical bars so retrieval never requires future data."""

    monthly_daily = market_data.get_history(
        symbol, trading_date, config.monthly_period * config.daily_window
    )
    weekly_daily = market_data.get_history(
        symbol, trading_date, config.weekly_period * config.daily_window
    )
    daily_daily = market_data.get_history(symbol, trading_date, config.daily_window)

    return QueryBars(
        monthly=aggregate_ohlc(monthly_daily, config.monthly_period),
        weekly=aggregate_ohlc(weekly_daily, config.weekly_period),
        daily=aggregate_ohlc(daily_daily, 1),
    )


def build_future_bars(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[OHLCBar, ...]:
    """Build the future-only daily bars used for visual comparison."""

    future_daily = market_data.get_future(symbol, trading_date, config.future_chart_window)
    return aggregate_ohlc(future_daily, 1)


def render_ohlc(
    bars: Sequence[OHLCBar],
    size: tuple[int, int] = (36, 36),
) -> GrayscaleImage:
    """Render OHLC bars as a black-on-white fixed-size image."""

    width, height = size
    if width < 5 or height < 5:
        raise ChartError("图像尺寸至少需要 5×5")
    if not bars:
        raise ChartError("不能渲染空的 OHLC 数据")

    pixels = [[255 for _ in range(width)] for _ in range(height)]
    plot_left, plot_right = 1, width - 2
    plot_top, plot_bottom = 1, height - 2
    low = min(bar.low for bar in bars)
    high = max(bar.high for bar in bars)
    price_range = high - low

    def y_for_price(price: float) -> int:
        if price_range == 0:
            return (plot_top + plot_bottom) // 2
        ratio = (high - price) / price_range
        return round(plot_top + ratio * (plot_bottom - plot_top))

    def set_black(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = 0

    for index, bar in enumerate(bars):
        if len(bars) == 1:
            x = (plot_left + plot_right) // 2
        else:
            x = round(plot_left + index * (plot_right - plot_left) / (len(bars) - 1))

        high_y = y_for_price(bar.high)
        low_y = y_for_price(bar.low)
        for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
            set_black(x, y)

        open_y = y_for_price(bar.open)
        close_y = y_for_price(bar.close)
        set_black(x - 1, open_y)
        set_black(x + 1, close_y)

    return GrayscaleImage(width=width, height=height, pixels=tuple(map(tuple, pixels)))


def build_sample_charts(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> SampleCharts:
    """Build rendered query charts and the future-only comparison chart."""

    sample_bars = build_sample_bars(market_data, symbol, trading_date, config)
    rendered = {
        name: render_ohlc(bars, config.chart_size)
        for name, bars in sample_bars.as_dict().items()
    }
    return SampleCharts(**rendered)


def build_query_charts(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> QueryCharts:
    """Build the three historical charts used by the retrieval model."""

    query_bars = build_query_bars(market_data, symbol, trading_date, config)
    rendered = {
        name: render_ohlc(bars, config.chart_size)
        for name, bars in query_bars.as_dict().items()
    }
    return QueryCharts(**rendered)


def build_future_chart(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> GrayscaleImage:
    """Render only the future comparison chart for a historical sample."""

    return render_ohlc(build_future_bars(market_data, symbol, trading_date, config), config.chart_size)


def save_sample_charts(
    charts: SampleCharts,
    directory: str | Path,
    prefix: str = "sample",
) -> dict[str, Path]:
    """Save all four charts and return their paths by scale name."""

    output_dir = Path(directory)
    return {
        name: image.save(output_dir / f"{prefix}_{name}.png")
        for name, image in charts.as_dict().items()
    }


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    body = chunk_type + payload
    checksum = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", checksum)
