import sqlite3
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.market_data import MarketBar, MarketData  # noqa: E402
from kline_retrieval.price_reconstruction import (  # noqa: E402
    materialize_reconstructed_prices,
    reconstruct_symbol_bars,
)
from kline_retrieval.sqlite_store import SCHEMA  # noqa: E402


class PriceReconstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = (
            MarketBar(
                "000001", date(2024, 1, day), open_price, high, low, close,
                return_rate=rate,
            )
            for day, open_price, high, low, close, rate in (
                (2, 98.0, 102.0, 97.0, 100.0, 0.04),
                (3, 103.0, 107.0, 101.0, 105.0, -0.1),
                (4, 94.0, 96.0, 90.0, 94.5, 0.05),
            )
        )
        self.bars = tuple(self.bars)

    def test_reconstructs_close_backward_from_latest_anchor_and_scales_ohlc(self) -> None:
        result = reconstruct_symbol_bars(self.bars)

        self.assertEqual(result.quality.status, "reconstructed")
        self.assertEqual(result.quality.extreme_scale_count, 0)
        self.assertIsNotNone(result.quality.scale_min)
        self.assertIsNotNone(result.quality.scale_max)
        self.assertAlmostEqual(result.bars[0].close, 100.0)
        self.assertAlmostEqual(result.bars[1].close, 90.0)
        self.assertAlmostEqual(result.bars[2].close, 94.5)
        self.assertAlmostEqual(result.bars[0].open, 98.0)
        self.assertAlmostEqual(result.bars[0].high, 102.0)
        self.assertAlmostEqual(result.bars[0].low, 97.0)
        self.assertEqual(result.bars[0].raw_close, 100.0)
        self.assertEqual(result.bars[0].price_basis, "reconstructed")
        for bar in result.bars:
            self.assertLessEqual(bar.low, min(bar.open, bar.close))
            self.assertGreaterEqual(bar.high, max(bar.open, bar.close))

    def test_missing_used_return_marks_symbol_unavailable_without_changing_raw_prices(self) -> None:
        broken = (
            self.bars[0],
            MarketBar("000001", date(2024, 1, 3), 103, 107, 101, 105, return_rate=None),
            self.bars[2],
        )

        result = reconstruct_symbol_bars(broken)

        self.assertEqual(result.quality.status, "raw_fallback_missing_or_invalid_return")
        self.assertEqual(result.quality.missing_return_count, 1)
        self.assertEqual(tuple(bar.close for bar in result.bars), (100.0, 105, 94.5))

    def test_materializes_new_columns_without_overwriting_raw_ohlc(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.executescript(SCHEMA)
                connection.executemany(
                    """INSERT INTO daily_bars
                       (symbol,date,open,high,low,close,volume,return_rate,pre_close)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        ("000001", "2024-01-02", 98, 102, 97, 100, 1000, 0.04, 96.15),
                        ("000001", "2024-01-03", 103, 107, 101, 105, 1000, -0.1, 100),
                        ("000001", "2024-01-04", 94, 96, 90, 94.5, 1000, 0.05, 105),
                    ],
                )
                connection.execute("INSERT INTO dataset_meta VALUES ('source_hash', 'fixture-hash')")

            report = materialize_reconstructed_prices(database)
            self.assertEqual(report["status_counts"], {"reconstructed": 1})
            market = MarketData.from_sqlite(database)
            self.assertEqual(market.price_basis, "reconstructed")
            self.assertNotEqual(market.source_hash, "fixture-hash")
            self.assertEqual(MarketData.from_sqlite(database, price_basis="raw").source_hash, "fixture-hash")
            first = market.get_bar("000001", "2024-01-02")
            self.assertAlmostEqual(first.close, 100.0)
            self.assertAlmostEqual(first.raw_close, 100.0)
            with sqlite3.connect(database) as connection:
                raw, rebuilt = connection.execute(
                    "SELECT close, reconstructed_close FROM daily_bars WHERE date='2024-01-03'"
                ).fetchone()
            self.assertEqual(raw, 105.0)
            self.assertAlmostEqual(rebuilt, 90.0)


if __name__ == "__main__":
    unittest.main()
