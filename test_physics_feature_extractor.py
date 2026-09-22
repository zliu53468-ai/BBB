import unittest

import numpy as np
from xgboost import XGBRegressor

from physics_feature_extractor import (
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    OfflineBaccaratSimulator,
    PhysicsFeatureExtractor,
    history_to_vector,
    prepare_xgboost_input,
)


class PhysicsFeatureExtractorTests(unittest.TestCase):
    def test_history_vector_shape_and_determinism(self):
        a = history_to_vector("BPTBBP")
        b = history_to_vector(["B", "P", "T", "B", "B", "P"])
        self.assertEqual(a.shape, (HISTORY_INPUT_DIM,))
        self.assertTrue(np.array_equal(a, b))

    def test_simulator_targets_are_physically_consistent(self):
        data = OfflineBaccaratSimulator(
            random_state=1234,
            max_hands_per_shoe=5,
        ).generate(3)
        self.assertEqual(data.y.shape[1], PHYSICS_DIM)

        for row in data.y:
            self.assertAlmostEqual(float(row[:3].sum()), 1.0, places=6)
            self.assertAlmostEqual(float(row[3:13].sum()), 1.0, places=6)
            self.assertAlmostEqual(float(row[13:23].sum()), 1.0, places=6)
            self.assertAlmostEqual(float(row[23:26].sum()), 1.0, places=6)
            self.assertGreaterEqual(float(row[26:39].sum()), 4.0)
            self.assertLessEqual(float(row[26:39].sum()), 6.0)
            self.assertAlmostEqual(float(row[39:43].sum()), 1.0, places=6)

    def test_prepare_xgboost_input_preserves_original_7d(self):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2,
            max_depth=2,
            n_jobs=1,
            tree_method="hist",
            multi_strategy="one_output_per_tree",
            verbosity=0,
        )
        extractor = PhysicsFeatureExtractor(model=model, random_state=9)
        data = OfflineBaccaratSimulator(
            random_state=9,
            max_hands_per_shoe=4,
        ).generate(3)
        extractor.fit(data.x, data.y)

        original = np.asarray([0.51, 12, 60, 0.8, 0.5, 2, 1], dtype=np.float32)
        merged = prepare_xgboost_input(
            0.51,
            original,
            "BPPBT",
            extractor=extractor,
        )
        self.assertEqual(merged.shape, (1 + 7 + PHYSICS_DIM,))
        self.assertAlmostEqual(float(merged[0]), 0.51, places=6)
        self.assertTrue(np.array_equal(merged[1:8], original))


if __name__ == "__main__":
    unittest.main()
