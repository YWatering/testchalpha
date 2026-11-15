import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.ann_index import (  # noqa: E402
    _batch_cosine_distances,
    _select_temporally_eligible_candidates,
)


class ANNIndexTests(unittest.TestCase):
    def test_batch_cosine_distances_matches_cosine_definition(self) -> None:
        query = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        candidates = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )

        distances = _batch_cosine_distances(query, float(np.linalg.norm(query)), candidates)

        np.testing.assert_allclose(distances, (0.0, 1.0, 1.0), rtol=0.0, atol=1e-6)

    def test_temporal_prefilter_runs_before_global_candidate_budget(self) -> None:
        query_date = date(2024, 1, 5)
        positions = {
            date(2024, 1, 2): 0,
            date(2024, 1, 3): 1,
            date(2024, 1, 4): 2,
            query_date: 3,
        }
        selected = _select_temporally_eligible_candidates(
            [
                (0.99, 0, 1),
                (0.98, 0, 2),
                (0.97, 1, 3),
                (0.96, 1, 4),
            ],
            {
                (0, 1): ("OTHER", "2024-01-08"),
                (0, 2): ("QUERY", "2024-01-04"),
                (1, 3): ("OTHER", "2024-01-03"),
                (1, 4): ("OTHER", "2024-01-02"),
            },
            query_symbol="QUERY",
            query_date=query_date,
            query_positions=positions,
            query_index=3,
            embargo_trading_days=1,
            budget=2,
        )

        self.assertEqual(selected, {1: {3, 4}})

    def test_temporal_prefilter_can_restrict_candidate_dates(self) -> None:
        query_date = date(2024, 1, 12)
        selected = _select_temporally_eligible_candidates(
            [(0.99, 0, 1), (0.98, 0, 2), (0.97, 0, 3)],
            {
                (0, 1): ("OTHER", "2024-01-04"),
                (0, 2): ("OTHER", "2024-01-05"),
                (0, 3): ("OTHER", "2024-01-08"),
            },
            query_symbol="QUERY",
            query_date=query_date,
            query_positions={query_date: 0},
            query_index=0,
            embargo_trading_days=0,
            eligible_candidate_dates={date(2024, 1, 5)},
            budget=2,
        )

        self.assertEqual(selected, {0: {2}})


if __name__ == "__main__":
    unittest.main()
