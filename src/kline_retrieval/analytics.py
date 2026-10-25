"""Forward-return labels and distribution statistics for retrieval matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import ceil, exp, pi, sqrt
from statistics import fmean, median
from typing import Iterable, Sequence

from .charts import GrayscaleImage, build_future_chart
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import InsufficientFutureError, MarketData
from .retrieval import SimilarityMatch


class ReturnAnalysisError(ValueError):
    """Raised when return labels or analysis parameters are invalid."""


@dataclass(frozen=True, slots=True)
class ForwardReturns:
    symbol: str
    date: date
    base_close: float
    values: tuple[tuple[int, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "base_close": self.base_close,
            "returns": {str(horizon): value for horizon, value in self.values},
        }

    def for_horizon(self, horizon: int) -> float:
        for current_horizon, value in self.values:
            if current_horizon == horizon:
                return value
        raise KeyError(horizon)


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    match: SimilarityMatch
    forward_returns: ForwardReturns
    future_chart: GrayscaleImage
    similarity_weight: float = 1.0

    def as_dict(self) -> dict[str, object]:
        sample = self.match.sample
        return {
            "rank": self.match.rank,
            "symbol": sample.symbol,
            "date": sample.date.isoformat(),
            "distance": self.match.distance,
            "similarity_weight": self.similarity_weight,
            "monthly_distance": self.match.monthly_distance,
            "weekly_distance": self.match.weekly_distance,
            "daily_distance": self.match.daily_distance,
            "returns": self.forward_returns.as_dict()["returns"],
        }


@dataclass(frozen=True, slots=True)
class Histogram:
    edges: tuple[float, ...]
    counts: tuple[int, ...]

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        return {"edges": list(self.edges), "counts": list(self.counts)}


@dataclass(frozen=True, slots=True)
class KDECurve:
    x: tuple[float, ...]
    density: tuple[float, ...]

    def as_dict(self) -> dict[str, list[float]]:
        return {"x": list(self.x), "density": list(self.density)}


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    horizon: int
    values: tuple[float, ...]
    mean: float | None
    weighted_mean: float | None
    median: float | None
    win_rate: float | None
    weighted_win_rate: float | None
    weighted_std: float | None
    effective_sample_size: float | None
    standard_error: float | None
    signal_strength: float | None
    lower_bound: float | None
    upper_bound: float | None
    minimum: float | None
    maximum: float | None
    quantiles: tuple[tuple[str, float], ...]
    histogram: Histogram
    kde: KDECurve

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "count": len(self.values),
            "mean": self.mean,
            "weighted_mean": self.weighted_mean,
            "median": self.median,
            "win_rate": self.win_rate,
            "weighted_win_rate": self.weighted_win_rate,
            "weighted_std": self.weighted_std,
            "effective_sample_size": self.effective_sample_size,
            "standard_error": self.standard_error,
            "signal_strength": self.signal_strength,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "quantiles": {name: value for name, value in self.quantiles},
            "histogram": self.histogram.as_dict(),
            "kde": self.kde.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReturnAnalysis:
    outcomes: tuple[MatchOutcome, ...]
    statistics: tuple[ReturnStatistics, ...]

    def for_horizon(self, horizon: int) -> ReturnStatistics:
        for statistic in self.statistics:
            if statistic.horizon == horizon:
                return statistic
        raise KeyError(horizon)

    def as_dict(self) -> dict[str, object]:
        return {
            "matches": [outcome.as_dict() for outcome in self.outcomes],
            "statistics": {
                str(statistic.horizon): statistic.as_dict()
                for statistic in self.statistics
            },
        }


def calculate_forward_returns(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    horizons: Sequence[int] = DEFAULT_CONFIG.future_horizons,
) -> ForwardReturns:
    """Calculate close-to-close forward returns by trading-day horizon."""

    normalized_horizons = _normalize_horizons(horizons)
    base_bar = market_data.get_bar(symbol, trading_date)
    try:
        future_bars = market_data.get_future(symbol, trading_date, max(normalized_horizons))
    except InsufficientFutureError as exc:
        raise ReturnAnalysisError(str(exc)) from exc

    values = tuple(
        (
            horizon,
            future_bars[horizon - 1].close / base_bar.close - 1.0,
        )
        for horizon in normalized_horizons
    )
    return ForwardReturns(
        symbol=symbol,
        date=base_bar.date,
        base_close=base_bar.close,
        values=values,
    )


def build_match_outcomes(
    market_data: MarketData,
    matches: Sequence[SimilarityMatch],
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    horizons: Sequence[int] | None = None,
) -> tuple[MatchOutcome, ...]:
    """Attach forward-return labels and future charts to ranked matches."""

    requested_horizons = config.future_horizons if horizons is None else horizons
    weights = similarity_weights(matches, config.similarity_temperature)
    outcomes: list[MatchOutcome] = []
    for match, weight in zip(matches, weights):
        sample = match.sample
        try:
            returns = calculate_forward_returns(
                market_data,
                sample.symbol,
                sample.date,
                requested_horizons,
            )
            future_chart = build_future_chart(
                market_data,
                sample.symbol,
                sample.date,
                config,
            )
        except (ReturnAnalysisError, InsufficientFutureError) as exc:
            raise ReturnAnalysisError(
                f"无法为 {sample.symbol} {sample.date.isoformat()} 计算完整未来标签: {exc}"
            ) from exc
        outcomes.append(
            MatchOutcome(
                match=match,
                forward_returns=returns,
                future_chart=future_chart,
                similarity_weight=weight,
            )
        )
    return tuple(outcomes)


def analyze_matches(
    market_data: MarketData,
    matches: Sequence[SimilarityMatch],
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    horizons: Sequence[int] | None = None,
) -> ReturnAnalysis:
    """Build match outcomes and distribution statistics for each horizon."""

    requested_horizons = config.future_horizons if horizons is None else _normalize_horizons(horizons)
    outcomes = build_match_outcomes(
        market_data,
        matches,
        config=config,
        horizons=requested_horizons,
    )
    statistics = tuple(
        summarize_returns(
            requested_horizon,
            [outcome.forward_returns.for_horizon(requested_horizon) for outcome in outcomes],
            weights=[outcome.similarity_weight for outcome in outcomes],
        )
        for requested_horizon in requested_horizons
    )
    return ReturnAnalysis(outcomes=outcomes, statistics=statistics)


def summarize_returns(
    horizon: int,
    values: Iterable[float],
    *,
    weights: Iterable[float] | None = None,
) -> ReturnStatistics:
    """Summarize one return horizon, including histogram and Gaussian KDE."""

    normalized_horizon = _normalize_horizons((horizon,))[0]
    raw_values = tuple(float(value) for value in values)
    raw_weights = (
        tuple(float(weight) for weight in weights)
        if weights is not None
        else tuple(1.0 for _ in raw_values)
    )
    if len(raw_values) != len(raw_weights):
        raise ReturnAnalysisError("收益值和相似度权重数量不一致")
    if not raw_values:
        return ReturnStatistics(
            horizon=normalized_horizon,
            values=(),
            mean=None,
            weighted_mean=None,
            median=None,
            win_rate=None,
            weighted_win_rate=None,
            weighted_std=None,
            effective_sample_size=None,
            standard_error=None,
            signal_strength=None,
            lower_bound=None,
            upper_bound=None,
            minimum=None,
            maximum=None,
            quantiles=(),
            histogram=Histogram(edges=(), counts=()),
            kde=KDECurve(x=(), density=()),
        )
    if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
        raise ReturnAnalysisError("相似度权重必须非负且至少有一个正权重")
    ordered_pairs = tuple(sorted(zip(raw_values, raw_weights), key=lambda pair: pair[0]))
    ordered = tuple(value for value, _ in ordered_pairs)
    ordered_weights = tuple(weight for _, weight in ordered_pairs)
    quantiles = tuple(
        (name, _quantile(ordered, probability))
        for name, probability in (("q10", 0.10), ("q25", 0.25), ("q75", 0.75), ("q90", 0.90))
    )
    weight_sum = sum(ordered_weights)
    weighted_mean = sum(value * weight for value, weight in zip(ordered, ordered_weights)) / weight_sum
    weighted_variance = sum(
        weight * (value - weighted_mean) ** 2
        for value, weight in zip(ordered, ordered_weights)
    ) / weight_sum
    weighted_std = sqrt(max(0.0, weighted_variance))
    effective_sample_size = weight_sum * weight_sum / sum(weight * weight for weight in ordered_weights)
    standard_error = weighted_std / sqrt(effective_sample_size)
    signal_strength = weighted_mean / weighted_std if weighted_std > 0 else None
    margin = 1.96 * standard_error
    return ReturnStatistics(
        horizon=normalized_horizon,
        values=ordered,
        mean=fmean(ordered),
        weighted_mean=weighted_mean,
        median=median(ordered),
        win_rate=sum(value > 0 for value in ordered) / len(ordered),
        weighted_win_rate=sum(
            weight for value, weight in zip(ordered, ordered_weights) if value > 0
        ) / weight_sum,
        weighted_std=weighted_std,
        effective_sample_size=effective_sample_size,
        standard_error=standard_error,
        signal_strength=signal_strength,
        lower_bound=weighted_mean - margin,
        upper_bound=weighted_mean + margin,
        minimum=ordered[0],
        maximum=ordered[-1],
        quantiles=quantiles,
        histogram=_histogram(ordered),
        kde=_kde(ordered),
    )


def similarity_weights(
    matches: Sequence[SimilarityMatch], temperature: float,
) -> tuple[float, ...]:
    """Convert ranked distances into normalized exponential similarity weights."""

    if temperature <= 0:
        raise ReturnAnalysisError("similarity_temperature 必须大于 0")
    if not matches:
        return ()
    raw = tuple(exp(-max(0.0, match.distance) / temperature) for match in matches)
    total = sum(raw)
    if total <= 0:
        return tuple(1.0 / len(raw) for _ in raw)
    return tuple(value / total for value in raw)


def _histogram(values: Sequence[float]) -> Histogram:
    if len(values) == 1 or values[0] == values[-1]:
        value = values[0]
        return Histogram(edges=(value - 0.5, value + 0.5), counts=(len(values),))

    bin_count = max(5, min(20, ceil(sqrt(len(values)))))
    minimum, maximum = values[0], values[-1]
    width = (maximum - minimum) / bin_count
    edges = tuple(minimum + index * width for index in range(bin_count + 1))
    counts = [0 for _ in range(bin_count)]
    for value in values:
        index = min(bin_count - 1, int((value - minimum) / width))
        counts[index] += 1
    return Histogram(edges=edges, counts=tuple(counts))


def _kde(values: Sequence[float], points: int = 64) -> KDECurve:
    if len(values) == 1 or values[0] == values[-1]:
        value = values[0]
        return KDECurve(x=(value,), density=(1.0,))

    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = sqrt(variance)
    bandwidth = 1.06 * standard_deviation * len(values) ** (-0.2)
    if bandwidth <= 0:
        bandwidth = (values[-1] - values[0]) / 10.0
    padding = max(bandwidth * 2.0, (values[-1] - values[0]) * 0.05)
    start = values[0] - padding
    end = values[-1] + padding
    grid = tuple(start + index * (end - start) / (points - 1) for index in range(points))
    scale = 1.0 / (len(values) * bandwidth * sqrt(2.0 * pi))
    density = tuple(
        scale
        * sum(exp(-0.5 * ((point - value) / bandwidth) ** 2) for value in values)
        for point in grid
    )
    return KDECurve(x=grid, density=density)


def _normalize_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(horizon) for horizon in horizons)
    if not normalized or any(horizon <= 0 for horizon in normalized):
        raise ReturnAnalysisError("收益周期必须包含正整数")
    if tuple(sorted(set(normalized))) != normalized:
        raise ReturnAnalysisError("收益周期必须按升序排列且不能重复")
    return normalized


def _quantile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight
