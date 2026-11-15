import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.stage15_store import Stage15PredictionStore  # noqa: E402


class Stage15PredictionStoreTests(unittest.TestCase):
    def test_run_resumes_and_query_write_is_replaceable(self) -> None:
        with TemporaryDirectory() as directory:
            store = Stage15PredictionStore(Path(directory) / "predictions.sqlite3")
            arguments = {
                "model": "colproj",
                "index_version": "ann-v1",
                "source_hash": "source-v1",
                "parameters": {"top_k": 20},
                "scope": {"start": "2024-01-01", "end": "2024-12-31"},
                "target_count": 2,
            }
            run_key = store.begin_run(**arguments)
            prediction = {
                "symbol": "600001",
                "signal_date": "2024-01-05",
                "status": "ok",
                "predicted_return": 0.01,
            }
            matches = [
                {
                    "query_symbol": "600001",
                    "query_date": "2024-01-05",
                    "rank": 1,
                    "candidate_symbol": "000001",
                }
            ]

            store.put_query(run_key, prediction, matches)
            store.finish_run(run_key, status="partial")
            resumed_key = store.begin_run(**arguments)

            self.assertEqual(resumed_key, run_key)
            self.assertEqual(
                store.completed_keys(run_key),
                {("600001", date(2024, 1, 5))},
            )
            self.assertEqual(store.run_status(run_key)["completed_count"], 1)

            prediction["predicted_return"] = 0.02
            matches[0]["candidate_symbol"] = "000002"
            store.put_query(run_key, prediction, matches)
            self.assertEqual(len(store.load_predictions(run_key)), 1)
            self.assertEqual(store.load_predictions(run_key)[0]["predicted_return"], 0.02)
            self.assertEqual(store.load_matches(run_key)[0]["candidate_symbol"], "000002")

    def test_parameter_change_creates_a_new_run(self) -> None:
        with TemporaryDirectory() as directory:
            store = Stage15PredictionStore(Path(directory) / "predictions.sqlite3")
            common = {
                "model": "colproj",
                "index_version": "ann-v1",
                "source_hash": "source-v1",
                "scope": {"start": "2024-01-01"},
                "target_count": 1,
            }

            first = store.begin_run(parameters={"top_k": 20}, **common)
            second = store.begin_run(parameters={"top_k": 30}, **common)

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
