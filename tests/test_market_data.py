import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import (  # noqa: E402
    DataValidationError,
    MarketData,
    UnknownTradingDateError,
)
from kline_retrieval.market_data import InsufficientHistoryError, InsufficientFutureError  # noqa: E402


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
                "volume": 1000 + offset,
            }
        )
    return rows


class MarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = MarketData.from_rows(make_rows())

    def test_symbols_and_dates_are_sorted(self) -> None:
        self.assertEqual(self.data.symbols, ("TEST",))
        self.assertEqual(self.data.dates("TEST")[0], date(2024, 1, 2))
        self.assertEqual(len(self.data.dates("TEST")), 300)

    def test_csv_fixture_loads_and_records_source_hash(self) -> None:
        fixture = ROOT / "data" / "sample_market_data.csv"
        loaded = MarketData.from_csv(fixture)

        self.assertEqual(loaded.symbols, ("DEMO",))
        self.assertEqual(len(loaded.dates("DEMO")), 12)
        self.assertIsNotNone(loaded.source_hash)

    def test_history_is_inclusive_and_future_is_exclusive(self) -> None:
        query_date = date(2024, 1, 2) + timedelta(days=250)
        history = self.data.get_history("TEST", query_date, 240)
        future = self.data.get_future("TEST", query_date, 21)

        self.assertEqual(len(history), 240)
        self.assertEqual(history[-1].date, query_date)
        self.assertEqual(len(future), 21)
        self.assertEqual(future[0].date, query_date + timedelta(days=1))

    def test_sample_validation_reports_available_observations(self) -> None:
        query_date = date(2024, 1, 2) + timedelta(days=250)
        result = self.data.validate_sample("TEST", query_date)

        self.assertTrue(result.valid)
        self.assertEqual(result.history_available, 251)
        self.assertEqual(result.future_available, 49)
        self.assertEqual(result.reasons, ())

    def test_sample_validation_reports_insufficient_windows(self) -> None:
        result = self.data.validate_sample("TEST", date(2024, 1, 10))

        self.assertFalse(result.valid)
        self.assertEqual(result.history_available, 9)
        self.assertIn("历史数据不足", result.reasons[0])

    def test_insufficient_windows_raise_specific_errors(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            self.data.get_history("TEST", date(2024, 1, 10), 240)

        with self.assertRaises(InsufficientFutureError):
            self.data.get_future("TEST", date(2024, 10, 25), 21)

    def test_unknown_trading_date_is_rejected(self) -> None:
        with self.assertRaises(UnknownTradingDateError):
            self.data.get_bar("TEST", "2024-01-01")

    def test_duplicate_rows_are_rejected(self) -> None:
        rows = make_rows(2)
        rows.append(rows[0])
        with self.assertRaises(DataValidationError):
            MarketData.from_rows(rows)

    def test_invalid_ohlc_is_rejected(self) -> None:
        rows = make_rows(1)
        rows[0]["high"] = 90
        with self.assertRaises(DataValidationError):
            MarketData.from_rows(rows)

    def test_missing_required_field_is_rejected(self) -> None:
        rows = make_rows(1)
        del rows[0]["close"]
        with self.assertRaises(DataValidationError):
            MarketData.from_rows(rows)


if __name__ == "__main__":
    unittest.main()
