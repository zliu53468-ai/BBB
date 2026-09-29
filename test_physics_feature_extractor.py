import unittest
import numpy as np

from physics_feature_extractor import (
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    OfflineBaccaratSimulator,
    augment_213d,
    history_to_vector,
    prepare_xgboost_input,
    sanitize_physics_prediction,
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

    def test_213d_augmentation_preserves_shape_and_targets(self):
        x=np.zeros((6,HISTORY_INPUT_DIM),dtype=np.float32)
        y=np.zeros((6,PHYSICS_DIM),dtype=np.float32)
        y[:,0]=1.0
        y[np.arange(6),23+(np.arange(6)%3)]=1.0
        augmented_x,augmented_y=augment_213d(x,y,ratio=.5,random_state=7)
        self.assertEqual(augmented_x.shape,(9,HISTORY_INPUT_DIM))
        self.assertEqual(augmented_y.shape,(9,PHYSICS_DIM))
        self.assertTrue(np.array_equal(augmented_x[:6],x))
        self.assertTrue(np.array_equal(augmented_y[:6],y))

    def test_calibrated_physics_probability_blocks_stay_normalized(self):
        raw=np.linspace(-1.0,2.0,PHYSICS_DIM,dtype=np.float32)
        output=sanitize_physics_prediction(raw,{"card_count":.8,"winner":1.2})
        for block in (output[:3],output[3:13],output[13:23],output[23:26],output[39:43]):
            self.assertAlmostEqual(float(np.sum(block)),1.0,places=6)


if __name__=="__main__":
    unittest.main()
