import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from kline_retrieval.api import create_app  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app(ROOT / "data" / "demo_market_data.csv"))

    def test_health_symbols_and_dates_endpoints(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        symbols = self.client.get("/api/symbols")
        self.assertEqual(symbols.status_code, 200)
        self.assertEqual(symbols.json()["symbols"], ["ALPHA", "BETA", "DEMO"])

        dates = self.client.get("/api/dates", params={"symbol": "DEMO"})
        self.assertEqual(dates.status_code, 200)
        self.assertEqual(len(dates.json()["dates"]), 360)

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


if __name__ == "__main__":
    unittest.main()
