import unittest
import numpy as np

from physics_feature_extractor import (
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    OfflineBaccaratSimulator,
    history_to_vector,
    prepare_xgboost_input,
)


class FakePhysics:
    def predict_features(self, history):
        return np.linspace(0.0, 1.0, PHYSICS_DIM, dtype=np.float32)


class PhysicsFeatureExtractorTests(unittest.TestCase):
    def test_history_vector_shape_and_determinism(self):
        a=history_to_vector("BPTBBP")
        b=history_to_vector(["B","P","T","B","B","P"])
        self.assertEqual(a.shape,(HISTORY_INPUT_DIM,))
        self.assertTrue(np.array_equal(a,b))

    def test_simulator_targets_are_physically_consistent(self):
        data=OfflineBaccaratSimulator(random_state=1234,max_hands_per_shoe=6).generate(3)
        self.assertEqual(data.y.shape[1],PHYSICS_DIM)
        for row in data.y:
            self.assertAlmostEqual(float(row[:3].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[3:13].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[13:23].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[23:26].sum()),1.0,places=6)
            self.assertGreaterEqual(float(row[26:39].sum()),4.0)
            self.assertLessEqual(float(row[26:39].sum()),6.0)
            self.assertAlmostEqual(float(row[39:43].sum()),1.0,places=6)
            self.assertGreaterEqual(float(row[43]),0.0)
            self.assertLessEqual(float(row[43]),416.0)

    def test_prepare_xgboost_input_preserves_original_7d(self):
        original=np.asarray([0.51,12,60,0.8,0.5,2,1],dtype=np.float32)
        merged=prepare_xgboost_input(0.51,original,"BPPBT",extractor=FakePhysics())
        self.assertEqual(merged.shape,(56,))
        self.assertAlmostEqual(float(merged[0]),0.51,places=6)
        self.assertTrue(np.array_equal(merged[1:8],original))
        self.assertEqual(merged[8:].shape,(PHYSICS_DIM,))


if __name__=="__main__":
    unittest.main()
