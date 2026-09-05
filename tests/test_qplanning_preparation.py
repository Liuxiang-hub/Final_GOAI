import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.qplanning.planner import q_weighted_chunk, validate_lingbot_candidates
from scripts.qplanning.rollout_manifest import validate_manifest


class QPlanningPreparationTest(unittest.TestCase):
    def test_q_weighting_prefers_higher_value(self):
        chunks = np.stack((np.zeros((50, 14)), np.ones((50, 14))), axis=0)
        selected, weights = q_weighted_chunk(chunks, np.array([0.0, 2.0]), temperature=0.5)
        self.assertGreater(weights[1], 0.98)
        self.assertTrue(np.all(selected > 0.98))

    def test_elite_mask(self):
        chunks = np.arange(3, dtype=np.float32)[:, None, None] * np.ones((3, 50, 14))
        _, weights = q_weighted_chunk(chunks, np.array([0.0, 1.0, 2.0]), n_elites=2)
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_candidate_contract_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            validate_lingbot_candidates(np.zeros((4, 15, 14)))

    def test_manifest_contract(self):
        record = {
            "episode_id": "baseline-fill-0001",
            "task": "fill_pen_holder",
            "success": False,
            "frames": 250,
            "source": "baseline_rollout",
            "data_path": "episodes/baseline-fill-0001",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(validate_manifest(path), 1)


if __name__ == "__main__":
    unittest.main()
