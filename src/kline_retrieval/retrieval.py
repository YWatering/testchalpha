"""ColProj embeddings and multi-scale similarity search."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from math import sqrt
from time import perf_counter
from typing import Protocol, Sequence

from .charts import QueryCharts, build_query_charts
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData


class Embedder(Protocol):
    """Embedding contract shared by ColProj and future CNN models."""

    dimension: int

    def encode(self, image: object) -> tuple[float, ...]:
        ...


class RetrievalError(ValueError):
    """Raised for invalid embedding or search parameters."""


class ColumnProjectionEmbedder:
    """Encode each image column as a normalized vertical ink trajectory.

    For each column, black-pixel intensity is used as the weight of each row.
    The weighted row centroid is converted to an upward-positive value, empty
    columns are linearly interpolated, then the trajectory is centered and
    L2-normalized. The result is a deterministic 36-dimensional shape vector.
    """

    dimension = 36

    def encode(self, image: object) -> tuple[float, ...]:
        width = getattr(image, "width", None)
        height = getattr(image, "height", None)
        pixels = getattr(image, "pixels", None)
        if width != self.dimension or height is None or pixels is None:
            raise RetrievalError("ColProj 需要宽度为 36 的灰度图像")
        if height < 2 or len(pixels) != height:
            raise RetrievalError("图像高度必须至少为 2")

        trajectory: list[float | None] = []
        for x in range(width):
            ink_weights = [255 - int(pixels[y][x]) for y in range(height)]
            total_ink = sum(ink_weights)
            if total_ink == 0:
                trajectory.append(None)
                continue
            centroid = sum(y * ink_weights[y] for y in range(height)) / total_ink
            trajectory.append(1.0 - centroid / (height - 1))

        values = _interpolate_empty_columns(trajectory)
        mean = sum(values) / len(values)
        centered = [value - mean for value in values]
        norm = sqrt(sum(value * value for value in centered))
        if norm == 0:
            return tuple(0.0 for _ in centered)
        return tuple(value / norm for value in centered)


@dataclass(frozen=True, slots=True)
class SampleEmbeddings:
    monthly: tuple[float, ...]
    weekly: tuple[float, ...]
    daily: tuple[float, ...]

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        return {
            "monthly": self.monthly,
            "weekly": self.weekly,
            "daily": self.daily,
        }


@dataclass(frozen=True, slots=True)
class RetrievalSample:
    symbol: str
    date: date
    embeddings: SampleEmbeddings


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    rank: int
    sample: RetrievalSample
    distance: float
    monthly_distance: float
    weekly_distance: float
    daily_distance: float


class SampleEmbeddingCache:
    """Bounded per-dataset LRU cache for historical sample embeddings."""

    def __init__(self, max_items: int = 20_000) -> None:
        if max_items <= 0:
            raise ValueError("max_items 必须大于 0")
        self.max_items = max_items
        self._items: OrderedDict[tuple[object, ...], RetrievalSample] = OrderedDict()

    def get(
        self,
        market_data: MarketData,
        symbol: str,
        trading_date: date,
        config: ProjectConfig,
        embedder: Embedder | None,
    ) -> tuple[RetrievalSample, bool]:
        model = embedder or ColumnProjectionEmbedder()
        key = (market_data, symbol, trading_date, config, type(model), id(model) if embedder else None)
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached, True
        sample = build_sample_embedding(
            market_data,
            symbol,
            trading_date,
            config=config,
            embedder=model,
        )
        self._items[key] = sample
        if len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return sample, False


def embed_query_charts(
    charts: QueryCharts,
    embedder: Embedder | None = None,
) -> SampleEmbeddings:
    model = embedder or ColumnProjectionEmbedder()
    return SampleEmbeddings(
        monthly=model.encode(charts.monthly),
        weekly=model.encode(charts.weekly),
        daily=model.encode(charts.daily),
    )


def build_sample_embedding(
    market_data: MarketData,
    symbol: str,
    trading_date: date | str,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    embedder: Embedder | None = None,
) -> RetrievalSample:
    """Create retrieval features without requiring a future chart."""

    parsed_date = _as_date(trading_date)
    charts = build_query_charts(market_data, symbol, parsed_date, config)
    return RetrievalSample(
        symbol=symbol,
        date=parsed_date,
        embeddings=embed_query_charts(charts, embedder),
    )


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise RetrievalError("余弦距离要求两个非空且等长向量")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 and right_norm == 0:
        return 0.0
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def weighted_distance(
    query: SampleEmbeddings,
    candidate: SampleEmbeddings,
    *,
    month_weight: float = 1.0,
    week_weight: float = 1.0,
    day_weight: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return total and per-scale cosine distances."""

    if day_weight != 1.0:
        raise RetrievalError("日线权重必须固定为 1.0")
    if month_weight < 0 or week_weight < 0 or month_weight + week_weight + day_weight <= 0:
        raise RetrievalError("月线和周线权重不能为负，且总权重必须大于 0")

    monthly = cosine_distance(query.monthly, candidate.monthly)
    weekly = cosine_distance(query.weekly, candidate.weekly)
    daily = cosine_distance(query.daily, candidate.daily)
    total = (
        month_weight * monthly + week_weight * weekly + day_weight * daily
    ) / (month_weight + week_weight + day_weight)
    return total, monthly, weekly, daily


def search_similar_samples(
    query: RetrievalSample,
    candidates: Sequence[RetrievalSample],
    *,
    month_weight: float = 1.0,
    week_weight: float = 1.0,
    day_weight: float = 1.0,
    top_k: int = 20,
) -> list[SimilarityMatch]:
    """Rank historical candidates by weighted multi-scale distance."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise RetrievalError("top_k 必须是大于 0 的整数")

    scored: list[tuple[float, str, date, float, float, float, RetrievalSample]] = []
    for candidate in candidates:
        if candidate.symbol == query.symbol and candidate.date == query.date:
            continue
        total, monthly, weekly, daily = weighted_distance(
            query.embeddings,
            candidate.embeddings,
            month_weight=month_weight,
            week_weight=week_weight,
            day_weight=day_weight,
        )
        scored.append((total, candidate.symbol, candidate.date, monthly, weekly, daily, candidate))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        SimilarityMatch(
            rank=index,
            sample=item[6],
            distance=item[0],
            monthly_distance=item[3],
            weekly_distance=item[4],
            daily_distance=item[5],
        )
        for index, item in enumerate(scored[:top_k], 1)
    ]


def search_market_data(
    market_data: MarketData,
    query_symbol: str,
    query_date: date | str,
    *,
    symbols: Sequence[str] | None = None,
    config: ProjectConfig = DEFAULT_CONFIG,
    embedder: Embedder | None = None,
    month_weight: float = 1.0,
    week_weight: float = 1.0,
    top_k: int = 20,
    embargo_trading_days: int = 0,
    embedding_cache: SampleEmbeddingCache | None = None,
    diagnostics: dict[str, int | float] | None = None,
) -> list[SimilarityMatch]:
    """Build valid historical candidates and return their ranked matches."""

    if embargo_trading_days < 0:
        raise RetrievalError("embargo_trading_days 不能小于 0")
    parsed_query_date = _as_date(query_date)
    started = perf_counter()
    cache_hits = 0
    cache_misses = 0
    embed_started = perf_counter()
    if embedding_cache is None:
        query = build_sample_embedding(
            market_data, query_symbol, parsed_query_date, config=config, embedder=embedder
        )
    else:
        query, hit = embedding_cache.get(
            market_data, query_symbol, parsed_query_date, config, embedder
        )
        cache_hits += int(hit)
        cache_misses += int(not hit)
    candidate_symbols = tuple(symbols) if symbols is not None else market_data.symbols
    candidates: list[RetrievalSample] = []
    history_count = config.monthly_period * config.daily_window
    future_count = max(config.future_horizons)
    query_dates = market_data.dates(query_symbol)
    query_index = query_dates.index(parsed_query_date)
    candidate_count = 0
    for symbol in candidate_symbols:
        for candidate_index, candidate_date in enumerate(market_data.dates(symbol)):
            if candidate_date >= parsed_query_date:
                continue
            if symbol == query_symbol:
                if query_index - candidate_index <= embargo_trading_days:
                    continue
            validation = market_data.validate_sample(
                symbol,
                candidate_date,
                history_count=history_count,
                future_count=future_count,
            )
            if not validation.valid:
                continue
            candidate_count += 1
            if embedding_cache is None:
                sample = build_sample_embedding(
                    market_data, symbol, candidate_date, config=config, embedder=embedder
                )
                cache_misses += 1
            else:
                sample, hit = embedding_cache.get(
                    market_data, symbol, candidate_date, config, embedder
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
            candidates.append(sample)

    embedding_seconds = perf_counter() - embed_started
    scoring_started = perf_counter()

    matches = search_similar_samples(
        query,
        candidates,
        month_weight=month_weight,
        week_weight=week_weight,
        top_k=top_k,
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "candidate_count": candidate_count,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "embedding_ms": round(embedding_seconds * 1000, 3),
                "scoring_ms": round((perf_counter() - scoring_started) * 1000, 3),
                "total_ms": round((perf_counter() - started) * 1000, 3),
            }
        )
    return matches


def _interpolate_empty_columns(values: Sequence[float | None]) -> list[float]:
    filled = list(values)
    known = [index for index, value in enumerate(filled) if value is not None]
    if not known:
        return [0.0 for _ in filled]
    for index in range(0, known[0]):
        filled[index] = filled[known[0]]
    for index in range(known[-1] + 1, len(filled)):
        filled[index] = filled[known[-1]]
    for left, right in zip(known, known[1:]):
        left_value = filled[left]
        right_value = filled[right]
        assert left_value is not None and right_value is not None
        span = right - left
        for index in range(left + 1, right):
            ratio = (index - left) / span
            filled[index] = left_value + ratio * (right_value - left_value)
    return [float(value) for value in filled]


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])
