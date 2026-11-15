import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.ann_index import _batch_cosine_distances  # noqa: E402


class ANNIndexTests(unittest.TestCase):
    def test_batch_cosine_distances_matches_cosine_definition(self) -> None:
        query = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        candidates = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )

        distances = _batch_cosine_distances(query, float(np.linalg.norm(query)), candidates)

        np.testing.assert_allclose(distances, (0.0, 1.0, 1.0), rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
