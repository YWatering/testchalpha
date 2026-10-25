import sys
import hashlib
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from kline_retrieval.api import app as configured_app, create_app  # noqa: E402
from kline_retrieval.config import ProjectConfig  # noqa: E402
from kline_retrieval.sqlite_store import write_snapshot  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(
            create_app(
                ROOT / "data" / "demo_market_data.csv",
                config=ProjectConfig(
                    daily_window=12,
                    future_chart_window=12,
                    chart_width=64,
                    chart_height=64,
                    future_horizons=(1, 5, 10, 12, 21),
                ),
            )
        )

    def test_health_symbols_and_dates_endpoints(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["dataset"]["provider"], "synthetic")

        symbols = self.client.get("/api/symbols")
        self.assertEqual(symbols.status_code, 200)
        self.assertEqual(symbols.json()["symbols"], ["ALPHA", "BETA", "DEMO"])

        dates = self.client.get("/api/dates", params={"symbol": "DEMO"})
        self.assertEqual(dates.status_code, 200)
        self.assertEqual(len(dates.json()["dates"]), 360)

    def test_health_reports_valid_dataset_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "a_share.csv"
            shutil.copyfile(ROOT / "data" / "demo_market_data.csv", data_path)
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            data_path.with_name("a_share.manifest.json").write_text(
                json.dumps(
                    {
                        "provider": "akshare",
                        "adjust": "qfq",
                        "data_start_date": "2023-01-01",
                        "data_end_date": "2024-01-01",
                        "sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            client = TestClient(create_app(data_path))

            dataset = client.get("/api/health").json()["dataset"]

        self.assertEqual(dataset["provider"], "akshare")
        self.assertEqual(dataset["adjust"], "qfq")

    def test_sqlite_dataset_reload_is_visible_without_restarting_app(self) -> None:
        rows = [
            {
                "symbol": "000001",
                "date": "2024-01-02",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000,
            }
        ]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            write_snapshot(database, rows, source_hash="snapshot-1", manifest={})
            client = TestClient(create_app(database))
            first = client.get("/api/health").json()

            rows[0]["close"] = 10.8
            rows[0]["high"] = 11.0
            write_snapshot(database, rows, source_hash="snapshot-2", manifest={})
            second = client.get("/api/health").json()

        self.assertEqual(first["storage"], "sqlite")
        self.assertEqual(first["source_hash"], "snapshot-1")
        self.assertEqual(second["source_hash"], "snapshot-2")

    def test_search_endpoint_returns_charts_matches_and_statistics(self) -> None:
        dates = self.client.get("/api/dates", params={"symbol": "DEMO"}).json()["dates"]
        response = self.client.post(
            "/api/retrieval/search",
            json={
                "symbol": "DEMO",
                "date": dates[-1],
                "month_weight": 1.5,
                "week_weight": 0.5,
                "top_k": 3,
                "embargo_trading_days": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload["query"]["charts"]), {"monthly", "weekly", "daily"})
        self.assertEqual(len(payload["matches"]), 3)
        self.assertEqual(set(payload["statistics"]), {"1", "5", "10", "12", "21"})
        self.assertTrue(payload["matches"][0]["future_chart"].startswith("data:image/png;base64,"))
        self.assertGreater(payload["performance"]["candidate_count"], 3)
        self.assertIn("scoring_ms", payload["performance"])
        self.assertGreater(payload["performance"]["request_ms"], 0)

        second = self.client.post(
            "/api/retrieval/search",
            json={
                "symbol": "DEMO",
                "date": dates[-1],
                "month_weight": 1.5,
                "week_weight": 0.5,
                "top_k": 3,
                "embargo_trading_days": 3,
            },
        ).json()
        self.assertEqual(
            [(row["symbol"], row["date"], row["distance"]) for row in payload["matches"]],
            [(row["symbol"], row["date"], row["distance"]) for row in second["matches"]],
        )
        self.assertEqual(second["performance"]["cache_misses"], 0)

    def test_search_endpoint_rejects_non_fixed_day_weight(self) -> None:
        response = self.client.post(
            "/api/retrieval/search",
            json={
                "symbol": "DEMO",
                "date": "2024-01-03",
                "day_weight": 2,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_static_homepage_is_available(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Pattern Atlas", response.text)

    def test_missing_configured_dataset_returns_service_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            missing_data_path = Path(directory) / "not-downloaded.csv"
            with patch.dict("os.environ", {"KLINE_DATA_PATH": str(missing_data_path)}):
                response = TestClient(configured_app).get("/api/health")

        self.assertEqual(response.status_code, 503)
        self.assertIn("无法加载配置的行情数据", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
