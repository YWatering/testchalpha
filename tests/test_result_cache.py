import sys
import sqlite3
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.result_cache import RetrievalResultCache  # noqa: E402
from kline_retrieval.retrieval import RetrievalSample, SampleEmbeddings, SimilarityMatch  # noqa: E402


class RetrievalResultCacheTests(unittest.TestCase):
    def test_versioned_parameters_round_trip_and_scope_count(self) -> None:
        match = SimilarityMatch(
            rank=1,
            sample=RetrievalSample("000001", date(2024, 1, 2), SampleEmbeddings((1.0,), (2.0,), (3.0,))),
            distance=0.25,
            monthly_distance=0.1,
            weekly_distance=0.2,
            daily_distance=0.3,
        )
        parameters = {"month_weight": 1.0, "week_weight": 1.0, "day_weight": 1.0, "top_k": 1}

        with TemporaryDirectory() as directory:
            cache = RetrievalResultCache(Path(directory) / "results.sqlite3")
            cache.put(
                [match], model="colproj", index_version="index-v1", source_hash="source-v1",
                symbol="000002", query_date=date(2024, 2, 1), parameters=parameters,
            )
            cached = cache.get(
                model="colproj", index_version="index-v1", source_hash="source-v1",
                symbol="000002", query_date=date(2024, 2, 1), parameters=parameters,
            )

            self.assertEqual(len(cached or []), 1)
            self.assertEqual(cached[0].sample.date, date(2024, 1, 2))
            self.assertEqual(cached[0].distance, 0.25)
            self.assertIsNone(
                cache.get(
                    model="colproj", index_version="index-v2", source_hash="source-v1",
                    symbol="000002", query_date=date(2024, 2, 1), parameters=parameters,
                )
            )
            self.assertIsNone(
                cache.get(
                    model="colproj", index_version="index-v1", source_hash="source-v1",
                    symbol="000002", query_date=date(2024, 2, 1),
                    parameters={**parameters, "week_weight": 2.0},
                )
            )
            self.assertEqual(
                cache.count_scope(
                    model="colproj", index_version="index-v1", source_hash="source-v1",
                    parameters=parameters,
                    date_ranges={"000002": (date(2024, 2, 1), date(2024, 2, 1))},
                ),
                1,
            )
            job_key = cache.begin_job(
                model="colproj", index_version="index-v1", source_hash="source-v1",
                parameters=parameters, target_count=1,
                scope={"symbols": ["000002"], "date_from": "2024-02-01"},
            )
            cache.update_job(job_key, 1, status="complete")
            self.assertEqual(
                cache.status()["jobs"][0]["scope"]["symbols"], ["000002"]
            )
            with sqlite3.connect(Path(directory) / "results.sqlite3") as connection:
                storage_type, payload_bytes = connection.execute(
                    "SELECT typeof(matches_json), length(matches_json) FROM retrieval_results"
                ).fetchone()
            self.assertEqual(storage_type, "blob")
            self.assertLess(payload_bytes, 128)


if __name__ == "__main__":
    unittest.main()
