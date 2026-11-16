import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.stage15_etf_benchmark import (  # noqa: E402
    ETFBenchmarkError,
    fetch_etf_daily,
    load_etf_snapshot,
    normalize_etf_rows,
    validate_snapshot_manifest,
    write_etf_snapshot,
)


class Stage15ETFBenchmarkTests(unittest.TestCase):
    def test_normalize_etf_rows_maps_akshare_columns(self) -> None:
        rows = normalize_etf_rows(
            [
                {
                    "日期": "2025-01-06",
                    "开盘": 3.9,
                    "收盘": 4.0,
                    "最高": 4.1,
                    "最低": 3.8,
                    "成交量": 1000,
                    "成交额": 4000,
                }
            ],
            "510300",
        )

        self.assertEqual(rows[0]["symbol"], "510300")
        self.assertEqual(rows[0]["date"], "2025-01-06")
        self.assertEqual(rows[0]["adjust"], "raw")

    def test_snapshot_round_trip_sets_previous_close(self) -> None:
        rows = normalize_etf_rows(
            [
                {"日期": "2025-01-06", "开盘": 3.9, "收盘": 4.0, "最高": 4.1, "最低": 3.8},
                {"日期": "2025-01-07", "开盘": 4.0, "收盘": 4.1, "最高": 4.2, "最低": 3.9},
            ],
            "510300",
        )
        with tempfile.TemporaryDirectory() as directory:
            bars_path = Path(directory) / "bars.csv"
            manifest_path = Path(directory) / "manifest.json"
            write_etf_snapshot(
                rows,
                bars_path,
                manifest_path,
                symbol="510300",
                name="沪深300ETF",
                provider_version="test",
                endpoint="eastmoney",
                requested_start=date(2025, 1, 1),
                requested_end=date(2025, 1, 31),
            )
            bars, sessions, source_hash = load_etf_snapshot(bars_path, "510300")

        self.assertEqual(sessions, (date(2025, 1, 6), date(2025, 1, 7)))
        self.assertIsNone(bars[("510300", date(2025, 1, 6))].pre_close)
        self.assertEqual(bars[("510300", date(2025, 1, 7))].pre_close, 4.0)
        self.assertEqual(len(source_hash), 64)

    def test_invalid_ohlc_is_rejected(self) -> None:
        with self.assertRaises(ETFBenchmarkError):
            normalize_etf_rows(
                [
                    {
                        "日期": "2025-01-06",
                        "开盘": 4.2,
                        "收盘": 4.0,
                        "最高": 4.1,
                        "最低": 3.8,
                    }
                ],
                "510300",
            )

    def test_fetch_retries_transient_failure(self) -> None:
        class Frame:
            empty = False

            def to_dict(self, orient: str) -> list[dict[str, object]]:
                self.orient = orient
                return [
                    {
                        "日期": "2025-01-06",
                        "开盘": 3.9,
                        "收盘": 4.0,
                        "最高": 4.1,
                        "最低": 3.8,
                    }
                ]

        attempts = 0

        def fetcher(**_: object) -> Frame:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary")
            return Frame()

        rows, _ = fetch_etf_daily(
            "510300",
            date(2025, 1, 1),
            date(2025, 1, 31),
            fetcher=fetcher,
            retries=2,
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(len(rows), 1)

    def test_normalize_sina_columns(self) -> None:
        rows = normalize_etf_rows(
            [
                {
                    "date": "2025-01-06",
                    "open": 3.9,
                    "close": 4.0,
                    "high": 4.1,
                    "low": 3.8,
                    "volume": 1000,
                }
            ],
            "510300",
            endpoint="fund_etf_hist_sina",
        )

        self.assertEqual(rows[0]["endpoint"], "fund_etf_hist_sina")
        self.assertEqual(rows[0]["amount"], 0)

    def test_snapshot_manifest_detects_tampering(self) -> None:
        rows = normalize_etf_rows(
            [
                {"日期": "2025-01-06", "开盘": 3.9, "收盘": 4.0, "最高": 4.1, "最低": 3.8}
            ],
            "510300",
        )
        with tempfile.TemporaryDirectory() as directory:
            bars_path = Path(directory) / "bars.csv"
            manifest_path = Path(directory) / "manifest.json"
            write_etf_snapshot(
                rows,
                bars_path,
                manifest_path,
                symbol="510300",
                name="沪深300ETF",
                provider_version="test",
                endpoint="eastmoney",
                requested_start=date(2025, 1, 1),
                requested_end=date(2025, 1, 31),
            )
            bars_path.write_text("changed", encoding="utf-8")

            with self.assertRaises(ETFBenchmarkError):
                validate_snapshot_manifest(bars_path, manifest_path, "510300")


if __name__ == "__main__":
    unittest.main()
