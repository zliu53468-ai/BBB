import unittest
import numpy as np

from physics_feature_extractor import PHYSICS_DIM
from physics_posterior_mcmc import PhysicsPosteriorExtractor
from xgb_global_probability import (
    GLOBAL_DIM,
    apply_dynamic_clip,
    prepare_inverted_xgboost_input,
)


class FakePhysics:
    def predict_features(self, history):
        x = np.zeros(PHYSICS_DIM, dtype=np.float32)
        x[23:26] = [0.46, 0.45, 0.09]
        x[43] = 120.0
        x[44] = 0.39
        x[45] = 0.31
        return x


class InvertedGlobalTests(unittest.TestCase):
    def test_prepare_inverted_input_is_56d(self):
        original = np.asarray([0.52, 15, 60, 0.75, 0.5, 2, 1], dtype=np.float32)
        x = prepare_inverted_xgboost_input(
            0.52, original, "BPPBTBBP", extractor=FakePhysics()
        )
        self.assertEqual(x.shape, (GLOBAL_DIM,))
        self.assertEqual(GLOBAL_DIM, 56)
        self.assertAlmostEqual(float(x[0]), 0.52, places=6)
        self.assertTrue(np.array_equal(x[1:8], original))

    def test_dynamic_clip_stays_probability(self):
        original = np.asarray([0.52, 15, 60, 0.75, 0.5, 2, 1], dtype=np.float32)
        x = prepare_inverted_xgboost_input(
            0.52, original, "BPPBTBBP", extractor=FakePhysics()
        )
        self.assertGreaterEqual(apply_dynamic_clip(0.99, x), 0.0)
        self.assertLessEqual(apply_dynamic_clip(0.99, x), 1.0)
        self.assertGreaterEqual(apply_dynamic_clip(0.01, x), 0.0)
        self.assertLessEqual(apply_dynamic_clip(0.01, x), 1.0)

    def test_posterior_summary_schema(self):
        posterior = PhysicsPosteriorExtractor(
            n_particles=32,
            rejuvenation_swaps=1,
            random_state=1234,
        )
        s = posterior.summarize_next_hand("BPTBP", rollouts_per_particle=1)
        self.assertEqual(s.physics_48d.shape, (PHYSICS_DIM,))
        self.assertGreaterEqual(s.physics_p_b, 0.0)
        self.assertLessEqual(s.physics_p_b, 1.0)
        self.assertGreater(s.ess, 0.0)
        self.assertEqual(s.n_particles, 32)


if __name__ == "__main__":
    unittest.main()
