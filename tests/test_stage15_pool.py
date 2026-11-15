import sqlite3
import sys
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.sqlite_store import SCHEMA  # noqa: E402
from kline_retrieval.stage15_pool import (  # noqa: E402
    StockPoolRules,
    allocate_proportional_quotas,
    inspect_index_coverage,
    select_stock_pool,
)


class Stage15StockPoolTests(unittest.TestCase):
    def test_quota_allocation_uses_largest_remainders(self) -> None:
        self.assertEqual(
            allocate_proportional_quotas({"A": 60, "B": 30, "C": 10}, 10),
            {"A": 6, "B": 3, "C": 1},
        )

    def test_selection_uses_only_build_date_information(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.executescript(SCHEMA)
                connection.executemany(
                    """
                    INSERT INTO instruments (symbol, name, exchange, board, status)
                    VALUES (?, ?, ?, ?, 'listed')
                    """,
                    [
                        ("600001", "沪一", "SSE", "MAIN"),
                        ("600002", "沪二", "SSE", "MAIN"),
                        ("600003", "沪三", "SSE", "MAIN"),
                        ("300001", "创一", "SZSE", "CHINEXT"),
                        ("300002", "创二", "SZSE", "CHINEXT"),
                        ("300003", "创三", "SZSE", "CHINEXT"),
                    ],
                )
                dates = [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                ]
                amounts = {
                    "600001": 600.0,
                    "600002": 500.0,
                    "600003": 100.0,
                    "300001": 400.0,
                    "300002": 300.0,
                    "300003": 50.0,
                }
                rows = []
                for symbol, amount in amounts.items():
                    for trading_date in dates:
                        rows.append(
                            (
                                symbol,
                                trading_date,
                                10.0,
                                11.0,
                                9.0,
                                10.5,
                                100.0,
                                amount,
                            )
                        )
                connection.executemany(
                    """
                    INSERT INTO daily_bars
                        (symbol, date, open, high, low, close, volume, amount)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.execute(
                    """
                    INSERT INTO daily_bars
                        (symbol, date, open, high, low, close, volume, amount)
                    VALUES ('600003', '2024-01-10', 10, 11, 9, 10.5, 100, 1000000)
                    """
                )
                connection.executemany(
                    "INSERT INTO dataset_meta (key, value) VALUES (?, ?)",
                    [("source_hash", "fixture"), ("price_basis", "fixture")],
                )
                connection.commit()

            result = select_stock_pool(
                database,
                StockPoolRules(
                    build_date="2024-01-09",
                    pool_size=4,
                    lookback_sessions=5,
                    min_history_sessions=5,
                    min_history_calendar_days=0,
                    min_recent_sessions=5,
                    min_liquid_sessions=5,
                    validation_start="2024-01-10",
                    validation_end="2024-01-20",
                    test_start="2024-01-21",
                    test_end="2024-01-31",
                ),
            )

            symbols = {item["symbol"] for item in result["rows"]}
            self.assertEqual(symbols, {"600001", "600002", "300001", "300002"})
            self.assertEqual(
                result["manifest"]["stratum_quotas"],
                {"SSE_MAIN": 2, "SZSE_CHINEXT": 2},
            )
            self.assertEqual(result["manifest"]["selection_data_end_date"], "2024-01-09")

    def test_index_coverage_reports_missing_symbols(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard.sqlite3"
            with sqlite3.connect(shard) as connection:
                connection.executescript(
                    """
                    CREATE TABLE feature_versions (
                        version_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE samples (
                        version_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO feature_versions VALUES ('v1', 'ready', '2024-01-01')"
                )
                connection.executemany(
                    "INSERT INTO samples VALUES ('v1', ?, ?)",
                    [("600001", "2024-01-02"), ("600001", "2024-01-03")],
                )
                connection.commit()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"indexes": ["shard.sqlite3"]}),
                encoding="utf-8",
            )

            coverage = inspect_index_coverage(manifest, ["600001", "600002"])

            self.assertEqual(coverage["covered_symbol_count"], 1)
            self.assertEqual(coverage["missing_symbols"], ["600002"])
            self.assertEqual(coverage["minimum_sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
