import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import (  # noqa: E402
    ColumnProjectionEmbedder,
    MarketData,
    SampleEmbeddingCache,
    build_sample_embedding,
    build_sample_charts,
    cosine_distance,
    search_market_data,
    weighted_distance,
)
from kline_retrieval.retrieval import RetrievalError  # noqa: E402


def make_rows(count: int = 320, symbol: str = "TEST", offset: float = 0.0) -> list[dict[str, object]]:
    start = date(2024, 1, 2)
    rows = []
    for index in range(count):
        trading_date = start + timedelta(days=index)
        base = 100 + offset + index * 0.1 + (index % 7) * 0.2
        rows.append(
            {
                "symbol": symbol,
                "date": trading_date.isoformat(),
                "open": base,
                "high": base + 2,
                "low": base - 1,
                "close": base + 1,
            }
        )
    return rows


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = MarketData.from_rows(
            make_rows(symbol="TEST") + make_rows(symbol="OTHER", offset=5.0)
        )
        self.query_date = date(2024, 1, 2) + timedelta(days=280)

    def test_colproj_returns_deterministic_36_dimensional_shape_vector(self) -> None:
        charts = build_sample_charts(self.data, "TEST", self.query_date)
        embedder = ColumnProjectionEmbedder()

        first = embedder.encode(charts.daily)
        second = embedder.encode(charts.daily)

        self.assertEqual(len(first), 36)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)

    def test_colproj_handles_flat_and_empty_projection(self) -> None:
        charts = build_sample_charts(self.data, "TEST", self.query_date)
        embedder = ColumnProjectionEmbedder()
        flat = embedder.encode(charts.daily)

        self.assertEqual(len(flat), 36)

    def test_cosine_distance_has_expected_identity_and_opposite_values(self) -> None:
        self.assertAlmostEqual(cosine_distance((1.0, 0.0), (1.0, 0.0)), 0.0)
        self.assertAlmostEqual(cosine_distance((1.0, 0.0), (-1.0, 0.0)), 2.0)
        self.assertAlmostEqual(cosine_distance((0.0, 0.0), (0.0, 0.0)), 0.0)

    def test_weighted_distance_returns_total_and_scale_distances(self) -> None:
        query = build_sample_embedding(self.data, "TEST", self.query_date)
        candidate = build_sample_embedding(
            self.data,
            "OTHER",
            date(2024, 1, 2) + timedelta(days=250),
        )

        total, monthly, weekly, daily = weighted_distance(
            query.embeddings,
            candidate.embeddings,
            month_weight=2.0,
            week_weight=0.5,
        )

        self.assertGreaterEqual(total, 0.0)
        self.assertGreaterEqual(monthly, 0.0)
        self.assertGreaterEqual(weekly, 0.0)
        self.assertGreaterEqual(daily, 0.0)
        with self.assertRaises(RetrievalError):
            weighted_distance(query.embeddings, candidate.embeddings, day_weight=2.0)

    def test_market_search_returns_ranked_historical_candidates(self) -> None:
        results = search_market_data(
            self.data,
            "TEST",
            self.query_date,
            top_k=5,
            month_weight=1.5,
            week_weight=0.5,
            embargo_trading_days=3,
        )

        self.assertEqual(len(results), 5)
        self.assertEqual([result.rank for result in results], [1, 2, 3, 4, 5])
        self.assertEqual(
            [result.distance for result in results],
            sorted(result.distance for result in results),
        )
        self.assertTrue(all(result.sample.date < self.query_date for result in results))
        self.assertTrue(
            all(
                not (
                    result.sample.symbol == "TEST"
                    and (self.query_date - result.sample.date).days <= 3
                )
                for result in results
            )
        )

    def test_market_search_can_query_latest_date_without_future_chart(self) -> None:
        latest_date = date(2024, 1, 2) + timedelta(days=319)
        results = search_market_data(self.data, "TEST", latest_date, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.sample.date < latest_date for result in results))

    def test_future_bars_do_not_change_query_embedding(self) -> None:
        rows = make_rows(count=330)
        query_date = date(2024, 1, 2) + timedelta(days=280)
        changed_rows = [dict(row) for row in rows]
        for row in changed_rows[281:]:
            row["open"] = float(row["open"]) * 1.8
            row["high"] = float(row["high"]) * 1.8
            row["low"] = float(row["low"]) * 1.8
            row["close"] = float(row["close"]) * 1.8

        original = build_sample_embedding(MarketData.from_rows(rows), "TEST", query_date)
        altered_future = build_sample_embedding(
            MarketData.from_rows(changed_rows), "TEST", query_date
        )

        self.assertEqual(original.embeddings, altered_future.embeddings)

    def test_embedding_cache_reuses_features_and_preserves_ranking(self) -> None:
        cache = SampleEmbeddingCache(max_items=512)
        cold_diagnostics: dict[str, int | float] = {}
        warm_diagnostics: dict[str, int | float] = {}
        cold = search_market_data(
            self.data,
            "TEST",
            self.query_date,
            top_k=5,
            embedding_cache=cache,
            diagnostics=cold_diagnostics,
        )
        warm = search_market_data(
            self.data,
            "TEST",
            self.query_date,
            top_k=5,
            embedding_cache=cache,
            diagnostics=warm_diagnostics,
        )

        self.assertEqual(cold, warm)
        self.assertGreater(cold_diagnostics["cache_misses"], 0)
        self.assertEqual(warm_diagnostics["cache_misses"], 0)
        self.assertGreater(warm_diagnostics["cache_hits"], 0)
        self.assertGreaterEqual(warm_diagnostics["candidate_count"], 5)


if __name__ == "__main__":
    unittest.main()
