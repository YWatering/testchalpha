import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval import (  # noqa: E402
    CompositeOfflineFeatureIndex,
    MarketData,
    OfflineFeatureIndex,
    ProjectConfig,
    build_colproj_index,
    build_sample_embedding,
    search_market_data,
)
from kline_retrieval.api import create_app  # noqa: E402
from kline_retrieval.sqlite_store import write_snapshot  # noqa: E402


def make_rows(count: int = 520, symbol: str = "000002", offset: float = 0.0) -> list[dict[str, object]]:
    start = date(2024, 1, 2)
    rows = []
    for index in range(count):
        close = 100 + offset + index * 0.1 + (index % 7) * 0.2
        rows.append(
            {
                "symbol": symbol,
                "date": (start + timedelta(days=index)).isoformat(),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            }
        )
    return rows


class OfflineIndexTests(unittest.TestCase):
    def test_build_and_search_index_matches_request_scan(self) -> None:
        rows = make_rows(symbol="000002") + make_rows(symbol="000004", offset=5.0)
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            index_path = Path(directory) / "colproj.sqlite3"
            write_snapshot(
                database,
                rows,
                source_hash="fixture-source",
                manifest={"provider": "fixture"},
            )
            config = ProjectConfig(active_symbols=("000002", "000004"))
            result = build_colproj_index(database, index_path, config=config)
            market = MarketData.from_sqlite(database, symbols=config.active_symbols)
            query_date = market.dates("000002")[-21]
            query = build_sample_embedding(market, "000002", query_date, config=config)
            indexed = OfflineFeatureIndex(index_path)
            with patch("kline_retrieval.offline_index.INDEX_SCAN_CHUNK_SIZE", 3):
                indexed_matches, diagnostics = indexed.search(
                    query,
                    market,
                    top_k=5,
                    embargo_trading_days=3,
                )
            scanned = search_market_data(
                market,
                "000002",
                query_date,
                config=config,
                top_k=5,
                embargo_trading_days=3,
            )

            self.assertEqual(result.sample_count, 202)
            self.assertEqual(result.inserted_count, 202)
            self.assertEqual(
                [(item.sample.symbol, item.sample.date) for item in indexed_matches],
                [(item.sample.symbol, item.sample.date) for item in scanned],
            )
            self.assertEqual(diagnostics["index_candidate_count"], 197)
            self.assertTrue(indexed.is_compatible(market.source_hash, config))
            with __import__("sqlite3").connect(index_path) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM return_labels").fetchone()[0],
                    202,
                )
                self.assertEqual(
                    connection.execute("SELECT status FROM feature_versions").fetchone()[0],
                    "ready",
                )

            api_config = ProjectConfig(
                data_path=database,
                active_symbols=config.active_symbols,
                index_path=index_path,
            )
            client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
                create_app(config=api_config)
            )
            self.assertEqual(
                client.get("/api/retrieval/config").json()["index_mode"],
                "offline_colproj",
            )
            response = client.post(
                "/api/retrieval/search",
                json={
                    "symbol": "000002",
                    "date": query_date.isoformat(),
                    "top_k": 3,
                    "embargo_trading_days": 3,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["performance"]["search_mode"], "offline_colproj"
            )

    def test_composite_index_merges_shards_and_reranks_globally(self) -> None:
        rows = make_rows(symbol="000002") + make_rows(symbol="000004", offset=5.0)
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            first_path = Path(directory) / "colproj-1.sqlite3"
            second_path = Path(directory) / "colproj-2.sqlite3"
            write_snapshot(database, rows, source_hash="fixture-source", manifest={"provider": "fixture"})
            config = ProjectConfig(active_symbols=("000002", "000004"))
            build_colproj_index(database, first_path, symbols=("000002",), config=config)
            build_colproj_index(database, second_path, symbols=("000004",), config=config)
            market = MarketData.from_sqlite(database, symbols=config.active_symbols)
            query_date = market.dates("000002")[-21]
            query = build_sample_embedding(market, "000002", query_date, config=config)
            composite = CompositeOfflineFeatureIndex((first_path, second_path))
            matches, diagnostics = composite.search(query, market, top_k=5, embargo_trading_days=3)

            self.assertTrue(composite.is_compatible(market.source_hash, config))
            self.assertEqual(diagnostics["index_shard_count"], 2)
            self.assertEqual(diagnostics["index_candidate_count"], 197)
            self.assertEqual(len(matches), 5)

    def test_api_loads_top_k_match_data_outside_runtime_symbol_subset(self) -> None:
        rows = make_rows(symbol="000002") + make_rows(symbol="000004", offset=5.0)
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            index_path = Path(directory) / "colproj.sqlite3"
            write_snapshot(
                database,
                rows,
                source_hash="fixture-source",
                manifest={"provider": "fixture"},
            )
            full_config = ProjectConfig(active_symbols=("000002", "000004"))
            build_colproj_index(database, index_path, config=full_config)
            query_date = (date(2024, 1, 2) + timedelta(days=499)).isoformat()
            runtime_config = ProjectConfig(
                active_symbols=("000002",),
                data_path=database,
                index_path=index_path,
            )
            client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
                create_app(config=runtime_config)
            )
            response = client.post(
                "/api/retrieval/search",
                json={"symbol": "000002", "date": query_date, "top_k": 5},
            )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["performance"]["index_candidate_count"], 100)


if __name__ == "__main__":
    unittest.main()
