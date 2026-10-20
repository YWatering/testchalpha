"""Project-wide defaults and environment-backed configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return parsed


def _non_negative_float(value: object, name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} 不能小于 0")
    return parsed


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Stable defaults shared by data, chart, retrieval and API layers."""

    data_path: Path = Path("data/demo_market_data.csv")
    monthly_period: int = 20
    weekly_period: int = 5
    daily_window: int = 12
    future_chart_window: int = 12
    chart_width: int = 36
    chart_height: int = 36
    future_horizons: tuple[int, ...] = (1, 5, 10, 12, 21)
    month_weight: float = 1.0
    week_weight: float = 1.0
    day_weight: float = 1.0
    default_top_k: int = 20
    active_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_path", Path(self.data_path))
        object.__setattr__(self, "active_symbols", tuple(self.active_symbols))
        _positive_int(self.monthly_period, "monthly_period")
        _positive_int(self.weekly_period, "weekly_period")
        _positive_int(self.daily_window, "daily_window")
        _positive_int(self.future_chart_window, "future_chart_window")
        _positive_int(self.chart_width, "chart_width")
        _positive_int(self.chart_height, "chart_height")
        _positive_int(self.default_top_k, "default_top_k")

        if not self.future_horizons or any(horizon <= 0 for horizon in self.future_horizons):
            raise ValueError("future_horizons 必须包含正整数")
        if tuple(sorted(set(self.future_horizons))) != self.future_horizons:
            raise ValueError("future_horizons 必须按升序排列且不能重复")

        _non_negative_float(self.month_weight, "month_weight")
        _non_negative_float(self.week_weight, "week_weight")
        if self.day_weight != 1.0:
            raise ValueError("day_weight 在当前产品定义中必须固定为 1.0")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ProjectConfig":
        """Build configuration from ``KLINE_*`` variables over project defaults."""

        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            data_path=env.get("KLINE_DATA_PATH", str(defaults.data_path)),
            monthly_period=int(env.get("KLINE_MONTHLY_PERIOD", defaults.monthly_period)),
            weekly_period=int(env.get("KLINE_WEEKLY_PERIOD", defaults.weekly_period)),
            daily_window=int(env.get("KLINE_DAILY_WINDOW", defaults.daily_window)),
            future_chart_window=int(
                env.get("KLINE_FUTURE_CHART_WINDOW", defaults.future_chart_window)
            ),
            chart_width=int(env.get("KLINE_CHART_WIDTH", defaults.chart_width)),
            chart_height=int(env.get("KLINE_CHART_HEIGHT", defaults.chart_height)),
            month_weight=float(env.get("KLINE_MONTH_WEIGHT", defaults.month_weight)),
            week_weight=float(env.get("KLINE_WEEK_WEIGHT", defaults.week_weight)),
            day_weight=float(env.get("KLINE_DAY_WEIGHT", defaults.day_weight)),
            default_top_k=int(env.get("KLINE_DEFAULT_TOP_K", defaults.default_top_k)),
            active_symbols=tuple(
                symbol.strip()
                for symbol in env.get("KLINE_ACTIVE_SYMBOLS", "").split(",")
                if symbol.strip()
            ),
        )

    @property
    def chart_size(self) -> tuple[int, int]:
        return self.chart_width, self.chart_height

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        return values


DEFAULT_CONFIG = ProjectConfig()
