import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.config import ProjectConfig  # noqa: E402
from kline_retrieval.batch_index_cli import build_next_batch  # noqa: E402
from kline_retrieval.sqlite_store import write_snapshot  # noqa: E402
from kline_retrieval.universe import (  # noqa: E402
    coverage_report,
    prepare_universe_batches,
    read_universe_symbols,
)


class UniverseTests(unittest.TestCase):
    def test_prepare_batches_tracks_valid_samples_and_filters(self) -> None:
        rows = []
        for symbol, offset in (("000002", 0), ("300750", 5)):
            for index in range(12):
                close = 100 + offset + index
                rows.append(
                    {
                        "symbol": symbol,
                        "date": (date(2024, 1, 2) + timedelta(days=index)).isoformat(),
                        "open": close - 0.5,
                        "high": close + 1,
                        "low": close - 1,
                        "close": close,
                    }
                )
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            write_snapshot(
                database,
                rows,
                source_hash="fixture-source",
                manifest={"provider": "fixture"},
                instruments=(
                    {"symbol": "000002", "name": "万科A", "exchange": "SZSE", "board": "MAIN"},
                    {"symbol": "300750", "name": "宁德时代", "exchange": "SZSE", "board": "CHINEXT"},
                ),
            )
            config = ProjectConfig(
                monthly_period=2,
                weekly_period=1,
                daily_window=2,
                future_chart_window=2,
                chart_width=64,
                chart_height=64,
                future_horizons=(1, 2),
            )
            report = prepare_universe_batches(database, config=config, batch_size=1)
            self.assertEqual(report["total_symbols"], 2)
            self.assertEqual(report["ready_symbols"], 2)
            self.assertEqual(report["coverage_ratio"], 1.0)
            self.assertEqual(report["batches"], {"pending": 2})
            filtered, total = read_universe_symbols(database, exchange="SZSE", board="CHINEXT")
            self.assertEqual(total, 1)
            self.assertEqual(filtered[0]["name"], "宁德时代")
            self.assertEqual(coverage_report(database)["latest_trading_date"], "2024-01-13")

    def test_next_batch_builds_shard_and_manifest(self) -> None:
        rows = []
        for index in range(30):
            close = 100 + index
            rows.append(
                {
                    "symbol": "000002",
                    "date": (date(2024, 1, 2) + timedelta(days=index)).isoformat(),
                    "open": close - 0.5,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                }
            )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "market.sqlite3"
            write_snapshot(
                database,
                rows,
                source_hash="fixture-source",
                manifest={"provider": "fixture"},
                instruments=({"symbol": "000002", "name": "万科A", "exchange": "SZSE", "board": "MAIN"},),
            )
            config = ProjectConfig(
                monthly_period=2,
                weekly_period=1,
                daily_window=2,
                future_chart_window=2,
                chart_width=64,
                chart_height=64,
                future_horizons=(1, 5, 10, 20),
            )
            result = build_next_batch(
                database,
                root / "indexes",
                root / "published.json",
                config=config,
                batch_size=1,
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["ready_shards"], 1)
            self.assertTrue((root / "published.json").is_file())


if __name__ == "__main__":
    unittest.main()
