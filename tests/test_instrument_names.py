import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.instrument_names import sync_instrument_names  # noqa: E402
from kline_retrieval.sqlite_store import SCHEMA  # noqa: E402


class InstrumentNameTests(unittest.TestCase):
    def test_sync_updates_database_names_and_keeps_fallbacks(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.executescript(SCHEMA)
                connection.executemany(
                    "INSERT INTO instruments(symbol, name, exchange, board, status) VALUES (?, ?, ?, ?, ?)",
                    [
                        ("000333", "000333", "sz", "main", "listed"),
                        ("000002", "000002", "sz", "main", "listed"),
                    ],
                )

            report = sync_instrument_names(
                database,
                fetcher=lambda **_: {"000002": "万科A"},
            )

            with sqlite3.connect(database) as connection:
                rows = dict(connection.execute("SELECT symbol, name FROM instruments"))
            self.assertEqual(rows["000333"], "美的集团")
            self.assertEqual(rows["000002"], "万科A")
            self.assertEqual(report["updated_symbols"], 2)


if __name__ == "__main__":
    unittest.main()
