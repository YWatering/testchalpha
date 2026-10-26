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
    CNNEmbedder,
    CNNTrainingConfig,
    build_split_manifest,
    train_cnn,
)
from kline_retrieval.config import ProjectConfig  # noqa: E402
from kline_retrieval.market_data import MarketData  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
