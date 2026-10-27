import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from kline_retrieval.cnn import (  # noqa: E402
    CNN_MODEL_VERSION,
    CNNEmbedder,
    CNNTrainingConfig,
    _build_network,
    build_split_manifest,
    train_cnn,
)
from kline_retrieval.config import ProjectConfig  # noqa: E402
from kline_retrieval.market_data import MarketData  # noqa: E402
from kline_retrieval.offline_index import OfflineFeatureIndex, build_colproj_index  # noqa: E402
from kline_retrieval.sqlite_store import write_snapshot  # noqa: E402
from kline_retrieval.cnn_index import build_cnn_shard  # noqa: E402
from kline_retrieval.api import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def make_rows(symbol: str, count: int = 720, offset: float = 0.0) -> list[dict[str, object]]:
    start = date(2020, 1, 2)
    rows = []
    for index in range(count):
        close = 100 + offset + index * 0.05 + (index % 11) * 0.1
        rows.append(
            {
                "symbol": symbol,
                "date": (start + timedelta(days=index)).isoformat(),
                "open": close - 0.3,
                "high": close + 0.8,
                "low": close - 0.9,
                "close": close,
            }
        )
    return rows


@unittest.skipUnless(torch is not None, "CNN tests require optional torch dependency")
class CNNTests(unittest.TestCase):
    def test_time_split_and_checkpoint_encode_three_scales(self) -> None:
        market = MarketData.from_rows(make_rows("000002") + make_rows("600519", offset=5.0))
        config = ProjectConfig(active_symbols=("000002", "600519"))
        manifest = build_split_manifest(
            market,
            config.active_symbols,
            config=config,
            max_per_symbol_split=2,
        )

        train_dates = [item["date"] for item in manifest["splits"]["train"]]
        validation_dates = [item["date"] for item in manifest["splits"]["validation"]]
        test_dates = [item["date"] for item in manifest["splits"]["test"]]
        self.assertLess(max(train_dates), min(validation_dates))
        self.assertLess(max(validation_dates), min(test_dates))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cnn.pt"
            result = train_cnn(
                market,
                manifest,
                checkpoint,
                config=CNNTrainingConfig(epochs=1, batch_size=2, device="cpu"),
            )
            self.assertEqual(result["train_samples"], 4)
            embedder = CNNEmbedder(checkpoint, device="cpu")
            embedding = embedder.encode_query_charts(
                __import__("kline_retrieval.charts", fromlist=["build_query_charts"]).build_query_charts(
                    market, "000002", date.fromisoformat(test_dates[0]), config
                )
            )
            self.assertEqual(len(embedding.monthly), 128)
            self.assertEqual(len(embedding.weekly), 128)
            self.assertEqual(len(embedding.daily), 128)

    def test_full_cnn_shard_build_is_compatible_resumable_and_searchable(self) -> None:
        config = ProjectConfig(
            monthly_period=1,
            weekly_period=1,
            daily_window=1,
            future_chart_window=20,
            chart_width=64,
            chart_height=64,
        )
        rows = make_rows("000002", count=70)
        market = MarketData.from_rows(rows)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "market.sqlite3"
            colproj = root / "colproj.sqlite3"
            cnn_path = root / "cnn.sqlite3"
            checkpoint = root / "cnn.pt"
            write_snapshot(database, rows, source_hash="test-market-v1", manifest={})
            torch.save(
                {
                    "model_state": _build_network().state_dict(),
                    "model_version": CNN_MODEL_VERSION,
                    "training_config": {"embedding_dim": 128},
                    "device": "cpu",
                },
                checkpoint,
            )
            build_colproj_index(database, colproj, config=config)

            first = build_cnn_shard(
                database, colproj, checkpoint, cnn_path,
                config=config, batch_size=16, device="cpu",
            )
            second = build_cnn_shard(
                database, colproj, checkpoint, cnn_path,
                config=config, batch_size=16, device="cpu",
            )

            index = OfflineFeatureIndex(cnn_path)
            self.assertEqual(first["status"], "ready")
            self.assertEqual(index.model, "cnn")
            self.assertEqual(index.dimension, 128)
            self.assertTrue(index.is_compatible("test-market-v1", config))
            self.assertGreater(first["sample_count"], 0)
            self.assertEqual(second["inserted_count"], 0)
            self.assertEqual(second["resumed_count"], first["sample_count"])

            client = TestClient(
                create_app(
                    database,
                    config=ProjectConfig(
                        monthly_period=1,
                        weekly_period=1,
                        daily_window=1,
                        future_chart_window=20,
                        chart_width=64,
                        chart_height=64,
                        index_path=colproj,
                        cnn_index_path=cnn_path,
                        cnn_checkpoint_path=checkpoint,
                    ),
                )
            )
            retrieval_config = client.get("/api/retrieval/config").json()
            response = client.post(
                "/api/retrieval/search",
                json={
                    "symbol": "000002",
                    "date": rows[-1]["date"],
                    "model": "cnn",
                    "top_k": 3,
                },
            )

        self.assertIn("cnn", retrieval_config["models"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"]["name"], "cnn")
        self.assertEqual(response.json()["performance"]["search_mode"], "offline_cnn")


if __name__ == "__main__":
    unittest.main()
