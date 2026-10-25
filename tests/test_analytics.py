import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import (  # noqa: E402
    MarketData,
    analyze_matches,
    calculate_forward_returns,
    search_market_data,
    summarize_returns,
)
from kline_retrieval.analytics import ReturnAnalysisError  # noqa: E402


def make_rows(count: int = 520, symbol: str = "TEST", offset: float = 0.0) -> list[dict[str, object]]:
    start = date(2024, 1, 2)
    rows = []
    for index in range(count):
        trading_date = start + timedelta(days=index)
        close = 100 + offset + index * 0.2
        rows.append(
            {
                "symbol": symbol,
                "date": trading_date.isoformat(),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            }
        )
    return rows


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = MarketData.from_rows(
            make_rows(symbol="TEST") + make_rows(symbol="OTHER", offset=5.0)
        )
        self.query_date = date(2024, 1, 2) + timedelta(days=450)

    def test_calculate_forward_returns_uses_close_to_close_formula(self) -> None:
        result = calculate_forward_returns(self.data, "TEST", self.query_date)

        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.date, self.query_date)
        self.assertAlmostEqual(result.base_close, 190.0)
        self.assertAlmostEqual(result.for_horizon(1), 0.2 / 190.0)
        self.assertEqual(tuple(horizon for horizon, _ in result.values), (1, 5, 10, 20))

    def test_calculate_forward_returns_rejects_incomplete_future_window(self) -> None:
        latest_date = date(2024, 1, 2) + timedelta(days=519)
        with self.assertRaises(ReturnAnalysisError):
            calculate_forward_returns(self.data, "TEST", latest_date)

    def test_summary_handles_values_and_empty_values(self) -> None:
        summary = summarize_returns(5, (-0.1, 0.0, 0.2, 0.3))

        self.assertAlmostEqual(summary.mean, 0.1)
        self.assertAlmostEqual(summary.weighted_mean, 0.1)
        self.assertAlmostEqual(summary.median, 0.1)
        self.assertAlmostEqual(summary.win_rate, 0.5)
        self.assertEqual(sum(summary.histogram.counts), 4)
        self.assertEqual(len(summary.kde.x), 64)
        self.assertEqual(len(summary.kde.density), 64)

        weighted = summarize_returns(5, (-0.1, 0.2), weights=(0.8, 0.2))
        self.assertAlmostEqual(weighted.weighted_mean, -0.04)
        self.assertLess(weighted.effective_sample_size, 2.0)

        empty = summarize_returns(5, ())
        self.assertIsNone(empty.mean)
        self.assertEqual(empty.histogram.counts, ())
        self.assertEqual(empty.kde.x, ())

    def test_analyze_matches_attaches_returns_future_charts_and_statistics(self) -> None:
        matches = search_market_data(self.data, "TEST", self.query_date, top_k=5)
        analysis = analyze_matches(self.data, matches)

        self.assertEqual(len(analysis.outcomes), 5)
        self.assertEqual(len(analysis.statistics), 4)
        self.assertEqual(analysis.for_horizon(20).horizon, 20)
        self.assertTrue(all(outcome.future_chart.black_pixel_count > 0 for outcome in analysis.outcomes))
        self.assertEqual(len(analysis.as_dict()["matches"]), 5)


if __name__ == "__main__":
    unittest.main()
