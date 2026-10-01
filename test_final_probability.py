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


class VectorClassifier:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        probability=np.clip(np.asarray(features,dtype=float)[:,0],1e-4,1-1e-4)
        return np.column_stack((1.0-probability,probability))


class FinalProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.core = 0.53
        self.original = np.asarray([0.53, 15, 60, 0.75, 0.5, 2, 1], dtype=np.float32)
        self.physics = np.zeros(PHYSICS_DIM, dtype=np.float32)
        self.physics[:3] = [0.2, 0.5, 0.3]
        self.physics[23:26] = [0.4586, 0.4462, 0.0952]
        self.physics[26:39] = np.arange(1, 14, dtype=np.float32) / 10.0
        self.physics[39:43] = [0.1, 0.2, 0.3, 0.4]

    def test_bridge_flattens_to_one_57d_matrix(self):
        matrix = final.build_56d_feature_matrix(self.core, self.original, self.physics)
        self.assertEqual(matrix.shape, (1, 57))
        self.assertAlmostEqual(float(matrix[0, 0]), self.core, places=6)
        self.assertAlmostEqual(float(matrix[0, 1]), (15 / 70.0) ** 3, places=6)
        self.assertTrue(np.array_equal(matrix[0, 2:8], self.original[1:]))
        self.assertTrue(np.array_equal(matrix[0, 8:56], self.physics))
        self.assertAlmostEqual(float(matrix[0, 56]), final.physics_noise_score(self.physics, self.original[1]), places=6)
        legacy=final.legacy_feature_matrix(matrix)
        self.assertEqual(legacy.shape,(1,57))
        self.assertAlmostEqual(float(legacy[0,-1]),0.0,places=6)

    def test_physics_noise_score_tracks_predictive_uncertainty(self):
        uncertain = self.physics.copy()
        uncertain[:3] = 1.0 / 3.0
        uncertain[3:13] = 0.1
        uncertain[13:23] = 0.1
        uncertain[23:26] = 1.0 / 3.0
        confident = uncertain.copy()
        confident[:3] = [1.0, 0.0, 0.0]
        confident[3:13] = [1.0] + [0.0] * 9
        confident[13:23] = [1.0] + [0.0] * 9
        confident[23:26] = [1.0, 0.0, 0.0]
        self.assertGreater(final.physics_noise_score(uncertain), final.physics_noise_score(confident))
        self.assertLess(abs(final.physics_noise_score(uncertain, 20) - 0.5), abs(final.physics_noise_score(uncertain, 55) - 0.5))

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
        self.assertAlmostEqual(prediction["final_p_b"], 0.55, places=6)
        self.assertAlmostEqual(result["raw_p_b"], 0.87, places=6)
        self.assertAlmostEqual(result["final_p_b"], 0.55, places=6)
        self.assertEqual(result["direction"], "B")
        self.assertEqual(result["final_direction"], "莊 B")
        self.assertGreater(result["ev_banker"], result["ev_player"])
        self.assertGreater(result["ev_banker"], 0.0)
        self.assertEqual(model.seen.shape, (1, 57))

    def test_player_ev_uses_binary_complement(self):
        result = final.predict_final_probability(
            self.core, self.original, self.physics, xgboost_model=FakeClassifier(0.48)
        )
        self.assertAlmostEqual(result["p_player"], 0.52, places=6)
        self.assertAlmostEqual(result["ev_player"], 0.04, places=6)
        self.assertAlmostEqual(result["min_ev"], 0.02, places=6)
        self.assertAlmostEqual(result["confidence"], 0.02, places=6)
        self.assertEqual(result["final_direction"], "閒 P")

    def test_dynamic_ev_thresholds(self):
        cases = ((15, 0.494, 0.020, "觀望 Skip"), (45, 0.494, 0.010, "閒 P"), (55, 0.497, 0.005, "閒 P"))
        for round_index, p_banker, min_ev, direction in cases:
            original = self.original.copy(); original[1] = round_index
            result = final.predict_final_probability(self.core, original, self.physics, xgboost_model=FakeClassifier(p_banker))
            self.assertAlmostEqual(result["min_ev"], min_ev, places=6)
            self.assertEqual(result["final_direction"], direction)

    def test_noise_separated_soft_policy_only_relaxes_clean_middle_and_late_rounds(self):
        early=final.decision_policy_value(30,.50,final.DEFAULT_MIN_EV,enabled=True)
        middle=final.decision_policy_value(45,.50,final.DEFAULT_MIN_EV,enabled=True)
        late=final.decision_policy_value(55,.50,final.DEFAULT_MIN_EV,enabled=True)
        noisy_late=final.decision_policy_value(55,1.0,final.DEFAULT_MIN_EV,enabled=True)
        self.assertTrue(np.allclose(early,(.020,.020,0.0)))
        self.assertTrue(np.allclose(middle,(.009,.008,.001)))
        self.assertTrue(np.allclose(late,(.004,.0025,.0015)))
        self.assertTrue(np.allclose(noisy_late,(.006,.006,0.0)))

    def test_soft_transition_uses_small_confidence_floor(self):
        model=FakeClassifier(.4955)
        model.bbb_decision_policy_={"enabled":True}
        original=self.original.copy();original[1]=45
        result=final.predict_final_probability(self.core,original,self.physics,xgboost_model=model)
        self.assertEqual(result["direction"],"P")
        self.assertAlmostEqual(result["min_ev"],.009,places=6)
        self.assertAlmostEqual(result["activation_ev"],.008,places=6)
        self.assertAlmostEqual(result["confidence"],final.MIN_SOFT_CONFIDENCE,places=6)

    def test_dynamic_bounds_expand_only_for_clean_late_physics(self):
        early = final.predict_final_probability(self.core, self.original, self.physics, xgboost_model=FakeClassifier(0.90))
        late_original = self.original.copy(); late_original[1] = 55
        late = final.predict_final_probability(self.core, late_original, self.physics, xgboost_model=FakeClassifier(0.90))
        noisy = self.physics.copy()
        noisy[:3] = 1.0 / 3.0
        noisy[3:13] = 0.1
        noisy[13:23] = 0.1
        noisy[23:26] = 1.0 / 3.0
        late_noisy = final.predict_final_probability(self.core, late_original, noisy, xgboost_model=FakeClassifier(0.90))
        self.assertAlmostEqual(early["final_p_b"], 0.55, places=6)
        self.assertAlmostEqual(late["final_p_b"], 0.65, places=6)
        self.assertAlmostEqual(late_noisy["final_p_b"], 0.60, places=6)

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

    def test_physics_integrity_is_reported_without_changing_features(self):
        consistent = self.physics.copy()
        consistent[26:39] = 5.1 / 13.0
        report = final.physics_integrity_report(consistent)
        self.assertTrue(report["valid"])
        self.assertTrue(report["checks"]["card_count_distribution"])
        self.assertTrue(report["checks"]["suit_ratio_distribution"])
        self.assertTrue(report["checks"]["rank_consumption_total"])
        self.assertAlmostEqual(report["expected_next_card_count"], 5.1, places=6)

        inconsistent = final.physics_integrity_report(self.physics)
        self.assertFalse(inconsistent["checks"]["rank_consumption_total"])

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
        self.assertEqual(x.shape, (2, 57))
        self.assertTrue(np.array_equal(y, np.asarray([1, 0], dtype=np.int8)))

    def test_training_prefers_captured_57d_snapshot(self):
        snapshot = final.build_56d_feature_matrix(self.core, self.original, self.physics)
        x, y = final.make_training_arrays([
            {
                "shoe_id": "shoe-snapshot",
                "features_57d": snapshot.tolist(),
                "actual_b": 1,
            },
        ])
        self.assertTrue(np.array_equal(x, snapshot))
        self.assertTrue(np.array_equal(y, np.asarray([1], dtype=np.int8)))

    def test_shoe_level_validation_holds_out_complete_latest_shoes(self):
        records = [
            {"shoe_id": shoe_id}
            for shoe_id in ("shoe-1", "shoe-1", "shoe-2", "shoe-2", "shoe-3", "shoe-3", "shoe-4", "shoe-4", "shoe-5")
        ]
        mask = final.shoe_level_validation_mask(records, fraction=0.20)
        self.assertTrue(np.array_equal(
            mask,
            np.asarray([False, False, False, False, False, False, False, False, True]),
        ))

    def test_three_way_split_keeps_complete_shoes_and_order(self):
        records = [{"shoe_id": f"shoe-{shoe}"} for shoe in range(10) for _ in range(2)]
        train, calibration, holdout = final.three_way_shoe_masks(
            records, calibration_fraction=0.20, holdout_fraction=0.20
        )
        self.assertFalse(np.any(train & calibration) or np.any(train & holdout) or np.any(calibration & holdout))
        self.assertTrue(np.all(train | calibration | holdout))
        self.assertTrue(np.all(train[:12]))
        self.assertTrue(np.all(calibration[12:16]))
        self.assertTrue(np.all(holdout[16:]))

        fit, tuning = final.nested_tuning_masks(records, train, fraction=0.20)
        self.assertTrue(np.all(fit[:8]))
        self.assertTrue(np.all(tuning[8:12]))
        self.assertFalse(np.any((fit | tuning) & (calibration | holdout)))

    def test_sample_weights_are_finite_and_normalized(self):
        x = np.zeros((6, 57), dtype=np.float32)
        x[:, 2] = [10, 20, 42, 48, 55, 60]
        weights = final.balanced_sample_weights(np.asarray([0, 0, 0, 0, 1, 1]), x)
        self.assertEqual(weights.shape, (6,))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_causal_ema_never_crosses_shoes(self):
        values = final.causal_ema_by_shoe(np.asarray([.8, .2, .4, .6]), ["A", "A", "B", "A"], .10)
        self.assertTrue(np.allclose(values, [.8, .26, .4, .566]))

    def test_smoothing_selection_can_safely_disable_itself(self):
        probability=np.asarray([.9,.1]*20);actual=np.asarray([1,0]*20,dtype=np.int8)
        x=np.zeros((40,57),dtype=np.float32);x[:,2]=np.arange(20,60);x[:,-1]=.5
        tuning=final.optimize_smoothing_and_thresholds(probability,actual,x,["shoe-A"]*40,strengths=[0,.10,.15])
        self.assertEqual(tuning["method"],"causal_ema")
        self.assertEqual(tuning["strength"],0.0)
        self.assertTrue(all("guardrail_passed" in row for row in tuning["candidates"]))

    def test_evaluation_reports_smoothing_before_and_after(self):
        x=np.zeros((20,57),dtype=np.float32);x[:,0]=np.asarray([.55,.45]*10);x[:,2]=np.arange(20,40);x[:,-1]=.5
        y=np.asarray([1,0]*10,dtype=np.int8);shoes=["A"]*10+["B"]*10
        report=final.evaluate(VectorClassifier(),x,y,probability_bounds=final.PROBABILITY_BOUNDS,
                              smoothing_strength=.05,shoe_ids=shoes,bootstrap_samples=10)
        self.assertEqual(set(report["smoothing"]),{"method","strength","guardrail_passed","before","after","delta"})
        self.assertIn("realized_ev_per_bet",report["smoothing"]["before"])
        self.assertIn("skip_rate",report["smoothing"]["after"])
        self.assertIn("guardrail_passed",report["decision_policy"])

    def test_ev_threshold_tuning_keeps_three_stages(self):
        probability = np.asarray([0.45, 0.55, 0.46, 0.54, 0.44, 0.56] * 4)
        actual = np.asarray([0, 1, 0, 1, 0, 1] * 4)
        rounds = np.asarray([20, 25, 45, 48, 55, 60] * 4)
        tuning = final.optimize_ev_thresholds(probability, actual, rounds)
        self.assertEqual(set(tuning["thresholds"]), {"early", "middle", "late"})
        self.assertEqual(set(tuning["stages"]), {"early", "middle", "late"})
        self.assertLessEqual(tuning["skip_rate_delta"], final.MAX_SKIP_RATE_INCREASE + 1e-12)
        self.assertTrue(tuning["skip_constraint"]["passed"])

    def test_ev_tuning_rejects_large_skip_increase(self):
        probability=np.full(100,.489,dtype=float)
        actual=np.asarray(([0]*55)+([1]*45),dtype=np.int8)
        rounds=np.full(100,20,dtype=float)
        tuning=final.optimize_ev_thresholds(probability,actual,rounds)
        self.assertLessEqual(tuning["skip_rate_delta"],.08+1e-12)

    def test_isotonic_calibration_mapping(self):
        calibrated=final.apply_probability_calibration(np.asarray([.25,.50,.75]),{"method":"isotonic","x_thresholds":[0.0,.5,1.0],"y_thresholds":[.1,.45,.9]})
        self.assertTrue(np.allclose(calibrated,[.275,.45,.675]))

    def test_calibration_selects_supported_method(self):
        probability=np.tile(np.linspace(.1,.9,20),30)
        features=np.zeros((len(probability),57),dtype=np.float32);features[:,0]=probability
        labels=(probability>.5).astype(np.int8)
        calibration=final.fit_probability_calibration(VectorClassifier(),features,labels,shoe_ids=[f"shoe-{index//20}" for index in range(len(labels))])
        self.assertIn(calibration["method"],{"platt","isotonic"})
        output=final.apply_probability_calibration(probability,calibration)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.all((output>0)&(output<1)))

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
