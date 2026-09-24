import unittest

import numpy as np

import xgb_final_probability as final
from physics_feature_extractor import PHYSICS_DIM


class FakeClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, banker_probability: float):
        self.banker_probability = banker_probability
        self.seen = None

    def predict_proba(self, features):
        self.seen = np.asarray(features, dtype=np.float32)
        return np.asarray([[1.0 - self.banker_probability, self.banker_probability]], dtype=float)


class FinalProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.core = 0.53
        self.original = np.asarray([0.53, 15, 60, 0.75, 0.5, 2, 1], dtype=np.float32)
        self.physics = np.zeros(PHYSICS_DIM, dtype=np.float32)
        self.physics[:3] = [0.2, 0.5, 0.3]
        self.physics[26:39] = np.arange(1, 14, dtype=np.float32) / 10.0
        self.physics[39:43] = [0.1, 0.2, 0.3, 0.4]

    def test_bridge_flattens_to_one_56d_matrix(self):
        matrix = final.build_56d_feature_matrix(self.core, self.original, self.physics)
        self.assertEqual(matrix.shape, (1, 56))
        self.assertAlmostEqual(float(matrix[0, 0]), self.core, places=6)
        self.assertTrue(np.array_equal(matrix[0, 1:8], self.original))
        self.assertTrue(np.array_equal(matrix[0, 8:], self.physics))

    def test_xgboost_probability_is_direct_and_bounded(self):
        model = FakeClassifier(0.87)
        prediction = final.predict_final_probability(
            self.core,
            self.original,
            self.physics,
            xgboost_model=model,
        )
        result = final.predict_final_result(
            self.core,
            self.original,
            self.physics,
            xgboost_model=model,
        )
        self.assertAlmostEqual(prediction["final_p_b"], 0.60, places=6)
        self.assertAlmostEqual(result["raw_p_b"], 0.87, places=6)
        self.assertAlmostEqual(result["final_p_b"], 0.60, places=6)
        self.assertEqual(result["direction"], "B")
        self.assertEqual(model.seen.shape, (1, 56))

    def test_physics_forecast_is_unpacked_with_expected_suit_consumption(self):
        forecast = final.unpack_physics_forecast(self.physics)
        self.assertAlmostEqual(
            forecast["next_card_count_probabilities"]["4_cards"], 0.2, places=6
        )
        self.assertAlmostEqual(
            forecast["next_card_count_probabilities"]["5_cards"], 0.5, places=6
        )
        self.assertAlmostEqual(
            forecast["next_card_count_probabilities"]["6_cards"], 0.3, places=6
        )
        self.assertAlmostEqual(forecast["expected_next_card_count"], 5.1, places=6)
        self.assertAlmostEqual(forecast["next_rank_expected_consumption"]["A"], 0.1, places=6)
        self.assertAlmostEqual(forecast["next_rank_expected_consumption"]["K"], 1.3, places=6)
        self.assertAlmostEqual(forecast["next_suit_consumption_ratios"]["spades"], 0.1, places=6)
        self.assertAlmostEqual(forecast["next_suit_expected_consumption"]["clubs"], 2.04, places=6)

    def test_training_labels_are_absolute_banker_player_values(self):
        common = {
            "core_p_b": self.core,
            "round_index": 15,
            "estimated_total_hands": 60,
            "remaining_ratio": 0.75,
            "sx_markov_p_same": 0.5,
            "stage": 2,
            "depth": 1,
            "physics_48d": self.physics.tolist(),
        }
        x, y = final.make_training_arrays([
            {**common, "actual_outcome": "B"},
            {**common, "actual_outcome": "P"},
            {**common, "actual_outcome": "T"},
        ])
        self.assertEqual(x.shape, (2, 56))
        self.assertTrue(np.array_equal(y, np.asarray([1, 0], dtype=np.int8)))

    def test_classifier_is_binary_logistic(self):
        class CapturingClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        original = final.XGBClassifier
        final.XGBClassifier = CapturingClassifier
        try:
            model = final.build_xgboost_classifier()
        finally:
            final.XGBClassifier = original
        self.assertEqual(model.kwargs["objective"], "binary:logistic")
        self.assertEqual(model.kwargs["eval_metric"], "logloss")


if __name__ == "__main__":
    unittest.main()
