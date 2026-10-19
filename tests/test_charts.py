import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import (  # noqa: E402
    ChartError,
    MarketData,
    aggregate_ohlc,
    build_sample_bars,
    build_sample_charts,
    render_ohlc,
    save_sample_charts,
)
from kline_retrieval.market_data import MarketBar  # noqa: E402


def make_rows(count: int = 300, symbol: str = "TEST") -> list[dict[str, object]]:
    start = date(2024, 1, 2)
    rows = []
    for offset in range(count):
        trading_date = start + timedelta(days=offset)
        base = 100 + offset * 0.1
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


class ChartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = MarketData.from_rows(make_rows())
        self.query_date = date(2024, 1, 2) + timedelta(days=250)

    def test_aggregate_ohlc_uses_first_open_last_close_and_extremes(self) -> None:
        bars = [
            MarketBar("TEST", date(2024, 1, day), 10 + day, 20 + day, 5 + day, 15 + day)
            for day in range(1, 5)
        ]

        aggregated = aggregate_ohlc(bars, 2)

        self.assertEqual(len(aggregated), 2)
        self.assertEqual(aggregated[0].open, 11)
        self.assertEqual(aggregated[0].high, 22)
        self.assertEqual(aggregated[0].low, 6)
        self.assertEqual(aggregated[0].close, 17)
        self.assertEqual(aggregated[0].start_date, date(2024, 1, 1))
        self.assertEqual(aggregated[0].end_date, date(2024, 1, 2))

    def test_aggregate_rejects_partial_group(self) -> None:
        bars = self.data.get_history("TEST", self.query_date, 11)
        with self.assertRaises(ChartError):
            aggregate_ohlc(bars, 5)

    def test_build_sample_bars_has_three_query_scales_and_future_chart(self) -> None:
        sample = build_sample_bars(self.data, "TEST", self.query_date)

        self.assertEqual(len(sample.monthly), 12)
        self.assertEqual(len(sample.weekly), 12)
        self.assertEqual(len(sample.daily), 12)
        self.assertEqual(len(sample.future_12d), 12)
        self.assertEqual(sample.monthly[-1].end_date, self.query_date)
        self.assertEqual(sample.future_12d[0].start_date, self.query_date + timedelta(days=1))

    def test_render_is_fixed_size_black_and_white_png(self) -> None:
        sample = build_sample_bars(self.data, "TEST", self.query_date)
        image = render_ohlc(sample.daily)

        self.assertEqual((image.width, image.height), (36, 36))
        self.assertGreater(image.black_pixel_count, 0)
        self.assertLess(image.black_pixel_count, 36 * 36)
        self.assertTrue(image.to_png_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(set(pixel for row in image.pixels for pixel in row) <= {0, 255})

    def test_build_and_save_sample_charts_is_deterministic(self) -> None:
        first = build_sample_charts(self.data, "TEST", self.query_date)
        second = build_sample_charts(self.data, "TEST", self.query_date)

        self.assertEqual(first, second)
        with TemporaryDirectory() as directory:
            paths = save_sample_charts(first, directory, prefix="test")
            self.assertEqual(set(paths), {"monthly", "weekly", "daily", "future_12d"})
            self.assertTrue(all(path.is_file() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
