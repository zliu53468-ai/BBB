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
        p_b,p_p=float(self.physics[23]),float(self.physics[24])
        ev_b=p_b*.95-p_p;ev_p=p_p-p_b
        self.physics[39:43] = [ev_b, ev_p, ev_b-ev_p, 0.30]

    def test_bridge_flattens_to_one_57d_matrix(self):
        matrix = final.build_56d_feature_matrix(self.core, self.original, self.physics)
        self.assertEqual(matrix.shape, (1, 57))
        self.assertAlmostEqual(float(matrix[0, 0]), self.core, places=6)
        expected_progress=final.effective_shoe_progress(self.original[1],self.physics)
        self.assertAlmostEqual(float(matrix[0, 1]), expected_progress ** 3, places=6)
        self.assertTrue(np.array_equal(matrix[0, 2:8], self.original[1:]))
        self.assertTrue(np.array_equal(matrix[0, 8:56], self.physics))
        self.assertAlmostEqual(float(matrix[0, 56]), final.physics_noise_score(self.physics, self.original[1]), places=6)
        legacy=final.legacy_feature_matrix(matrix)
        self.assertEqual(legacy.shape,(1,57))
        self.assertAlmostEqual(float(legacy[0,-1]),0.0,places=6)

    def test_effective_progress_uses_card_consumption_without_changing_dimension(self):
        low=self.physics.copy();high=self.physics.copy()
        low[43]=40.0;high[43]=250.0
        low_progress=final.effective_shoe_progress(50,low)
        high_progress=final.effective_shoe_progress(50,high)
        self.assertGreater(high_progress,low_progress)
        low_matrix=final.build_56d_feature_matrix(self.core,[self.core,50,60,.2,.5,2,1],low)
        high_matrix=final.build_56d_feature_matrix(self.core,[self.core,50,60,.2,.5,2,1],high)
        self.assertEqual(low_matrix.shape,(1,57));self.assertEqual(high_matrix.shape,(1,57))
        self.assertGreater(high_matrix[0,1],low_matrix[0,1])

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

    def test_strict_selective_policy_removes_soft_entries_and_penalizes_noise(self):
        policy=final.DECISION_POLICY_PROFILES["strict_selective_entry"]
        thresholds={"early":.038,"middle":.024,"late":.018}
        clean=final.decision_policy_value(55,.50,thresholds,enabled=True,policy_config=policy)
        noisy=final.decision_policy_value(55,1.0,thresholds,enabled=True,policy_config=policy)
        self.assertTrue(np.allclose(clean,(.018,.018,0.0)))
        self.assertTrue(np.allclose(noisy,(.024,.024,0.0)))
        model=FakeClassifier(.4955)
        model.bbb_ev_thresholds_=thresholds
        model.bbb_decision_policy_=policy
        original=self.original.copy(); original[1]=45
        result=final.predict_final_probability(self.core,original,self.physics,xgboost_model=model)
        self.assertEqual(result["direction"],"Skip")
        self.assertAlmostEqual(result["min_ev"],.024,places=6)
        self.assertAlmostEqual(result["activation_ev"],.024,places=6)

    def test_confidence_band_is_stage_and_noise_aware(self):
        policy=final.DECISION_POLICY_PROFILES["strict_selective_entry"]
        band,strong=final.confidence_band_arrays(np.asarray([30,45,55,55]),np.asarray([.50,.50,.50,.90]),policy)
        self.assertTrue(np.allclose(band,[.028,.020,.011,.0212]))
        self.assertTrue(np.allclose(strong,[.007,.005,.003,.003]))

    def test_volume_guard_relaxes_only_after_prior_entry_density_is_too_low(self):
        policy=final.DECISION_POLICY_PROFILES["strict_selective_entry"]
        probabilities=np.asarray([.525]*8+[.532],dtype=np.float64)
        actual=np.asarray([1]*9,dtype=np.int8)
        rounds=np.asarray([30]*9,dtype=np.float64)
        realised,wagered=final.decision_returns(probabilities,actual,rounds,{"early":.038,"middle":.024,"late":.018},np.full(9,.5),policy_enabled=True,policy_config=policy,shoe_ids=["A"]*9)
        self.assertFalse(np.any(wagered[:8]))
        self.assertTrue(wagered[8])
        self.assertEqual(final.decision_metrics(realised,wagered)["absolute_correct_bets"],1.0)

    def test_dynamic_bounds_expand_only_for_clean_late_physics(self):
        early = final.predict_final_probability(self.core, self.original, self.physics, xgboost_model=FakeClassifier(0.90))
        late_original = self.original.copy(); late_original[1] = 55
        late = final.predict_final_probability(self.core, late_original, self.physics, xgboost_model=FakeClassifier(0.90))
        noisy = self.physics.copy()
        noisy[:3] = 1.0 / 3.0
        noisy[3:13] = 0.1
        noisy[13:23] = 0.1
        noisy[23:26] = 1.0 / 3.0
        noisy[42] = 1.0
        late_noisy = final.predict_final_probability(self.core, late_original, noisy, xgboost_model=FakeClassifier(0.90))
        self.assertAlmostEqual(early["final_p_b"], 0.55, places=6)
        self.assertAlmostEqual(late["final_p_b"], 0.65, places=6)
        self.assertAlmostEqual(late_noisy["final_p_b"], 0.60, places=6)

    def test_physics_forecast_exposes_pre_core_physical_ev(self):
        forecast = final.unpack_physics_forecast(self.physics)
        self.assertAlmostEqual(forecast["next_card_count_probabilities"]["4_cards"], 0.2, places=6)
        self.assertAlmostEqual(forecast["next_card_count_probabilities"]["5_cards"], 0.5, places=6)
        self.assertAlmostEqual(forecast["next_card_count_probabilities"]["6_cards"], 0.3, places=6)
        self.assertAlmostEqual(forecast["expected_next_card_count"], 5.1, places=6)
        self.assertAlmostEqual(forecast["next_rank_expected_consumption"]["A"], 0.1, places=6)
        self.assertAlmostEqual(forecast["next_rank_expected_consumption"]["K"], 1.3, places=6)
        physical=forecast["physical_ev"]
        self.assertAlmostEqual(physical["physical_ev_banker"],float(self.physics[39]),places=6)
        self.assertAlmostEqual(physical["physical_ev_player"],float(self.physics[40]),places=6)
        self.assertAlmostEqual(physical["physical_ev_gap"],float(self.physics[41]),places=6)
        self.assertAlmostEqual(physical["particle_uncertainty"],.30,places=6)

    def test_physics_integrity_is_reported_without_changing_features(self):
        consistent = self.physics.copy()
        consistent[26:39] = 5.1 / 13.0
        report = final.physics_integrity_report(consistent)
        self.assertTrue(report["valid"])
        self.assertTrue(report["checks"]["card_count_distribution"])
        self.assertTrue(report["checks"]["rank_consumption_total"])
        self.assertTrue(report["checks"]["physical_ev_ranges"])
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

    def test_time_split_rejects_interleaved_or_backwards_shoes(self):
        with self.assertRaisesRegex(ValueError, "reappears"):
            final.three_way_shoe_masks([
                {"shoe_id":"A","round_index":1},{"shoe_id":"B","round_index":1},{"shoe_id":"A","round_index":2},
            ])
        with self.assertRaisesRegex(ValueError, "backwards"):
            final.validate_chronological_shoes([
                {"shoe_id":"A","round_index":2},{"shoe_id":"A","round_index":1},
            ])

    def test_sample_weights_are_finite_and_normalized(self):
        x = np.zeros((6, 57), dtype=np.float32)
        x[:, 2] = [10, 20, 42, 48, 55, 60]
        weights = final.balanced_sample_weights(np.asarray([0, 0, 0, 0, 1, 1]), x)
        self.assertEqual(weights.shape, (6,))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_sample_weights_emphasize_physical_ev_without_label_leakage(self):
        x=np.zeros((4,57),dtype=np.float32);x[:,2]=45;x[:,1]=(45/70)**3;x[:,-1]=.5
        x[:,8+final._PHYSICS_INDEX["particle_uncertainty"]]=.3
        x[:,8+final._PHYSICS_INDEX["physical_ev_banker"]]=[.0,.04,.0,.04]
        weights=final.balanced_sample_weights(np.asarray([0,0,1,1]),x)
        self.assertGreater(weights[1],weights[0]);self.assertGreater(weights[3],weights[2])

    def test_sample_weights_focus_clean_50_to_70_rows_smoothly(self):
        x=np.zeros((5,57),dtype=np.float32);rounds=np.asarray([30,45,55,60,65],dtype=float);x[:,2]=rounds;x[:,1]=(rounds/70)**3;x[:,-1]=.2
        x[:,8+final._PHYSICS_INDEX["particle_uncertainty"]]=.2
        x[:,8+final._PHYSICS_INDEX["physical_ev_banker"]]=.03
        weights=final.balanced_sample_weights(np.asarray([0,1,0,1,0]),x)
        self.assertGreater(weights[2],weights[0]);self.assertGreater(weights[4],weights[1])
        self.assertLess(weights[2],weights[3]);self.assertLess(weights[3],weights[4])

    def test_recalibrated_noise_only_changes_feature_56(self):
        x=np.zeros((2,57),dtype=np.float32);x[:,2]=[45,55];x[:,-1]=[.4,.6]
        records=[{"physics_48d":self.physics.tolist()},{"physics_48d":self.physics.tolist()}]
        calibration={"method":"isotonic","x_thresholds":[0,1],"y_thresholds":[0.1,0.9]}
        out=final.recalibrate_noise_feature(x,records,calibration)
        self.assertTrue(np.array_equal(out[:,:-1],x[:,:-1]))
        self.assertFalse(np.array_equal(out[:,-1],x[:,-1]))

    def test_dynamic_post_clip_ema_never_crosses_shoes(self):
        x=np.zeros((4,57),dtype=np.float32);x[:,2]=20;x[:,-1]=.5
        values=final.dynamic_ema_by_shoe(np.asarray([.55,.45,.48,.52]),x,["A","A","B","A"],final.EMA_PROFILES["balanced"])
        self.assertTrue(np.allclose(values,[.55,.51,.48,.514]))

    def test_dynamic_ema_alpha_is_continuous_and_more_noise_smoothing(self):
        config=final.EMA_PROFILES["balanced"]
        values=[final.dynamic_ema_alpha(r,.5,config) for r in (40,49,50,51,60,70)]
        self.assertTrue(all(a<=b+1e-12 for a,b in zip(values,values[1:])))
        self.assertLess(abs(values[1]-values[2]),.03);self.assertLess(abs(values[2]-values[3]),.03)
        for round_index in (30,45,55,65):
            high=final.dynamic_ema_alpha(round_index,1.0,config);low=final.dynamic_ema_alpha(round_index,0.0,config)
            self.assertLess(high,low);self.assertGreaterEqual(high,.35);self.assertLessEqual(low,.75)

    def test_smoothing_selection_can_safely_disable_itself(self):
        probability=np.asarray([.9,.1]*20);actual=np.asarray([1,0]*20,dtype=np.int8)
        x=np.zeros((40,57),dtype=np.float32);x[:,2]=np.arange(20,60);x[:,-1]=.5
        tuning=final.optimize_smoothing_and_thresholds(probability,actual,x,["shoe-A"]*40,profiles=["off","balanced"])
        self.assertEqual(tuning["method"],"dynamic_post_clip_ema")
        self.assertEqual(tuning["smoothing"]["profile"],"off")
        self.assertTrue(all("guardrail_passed" in row for row in tuning["candidates"]))
        self.assertIn("strict_selective_entry",{row["decision_policy_profile"] for row in tuning["candidates"]})

    def test_evaluation_reports_smoothing_before_and_after(self):
        x=np.zeros((20,57),dtype=np.float32);x[:,0]=np.asarray([.55,.45]*10);x[:,2]=np.arange(20,40);x[:,-1]=.5
        y=np.asarray([1,0]*10,dtype=np.int8);shoes=["A"]*10+["B"]*10
        report=final.evaluate(VectorClassifier(),x,y,probability_bounds=final.PROBABILITY_BOUNDS,
                              smoothing_config=final.EMA_PROFILES["balanced"],shoe_ids=shoes,bootstrap_samples=10)
        self.assertEqual(report["smoothing"]["method"],"dynamic_post_clip_ema")
        self.assertEqual(report["smoothing"]["profile"],"balanced")
        self.assertIn("realized_ev_per_bet",report["smoothing"]["before"])
        self.assertIn("hit_rate",report["smoothing"]["before"])
        self.assertIn("skip_rate",report["smoothing"]["after"])
        self.assertIn("guardrail_passed",report["decision_policy"])
        self.assertEqual(set(report["upgrade_comparison"]),{"guardrail_passed","before","after","delta","absolute_correct_bets_constraint_passed"})
        self.assertIn("absolute_correct_bets",report)
        for stage in ("early","middle","late","late_50_70","late_50_55","late_56_60","late_61_65","late_66_70"):
            self.assertIn("overall_accuracy",report["stage_decision_metrics"][stage])
            self.assertIn("brier",report["stage_decision_metrics"][stage])
            self.assertIn("mean_effective_progress_round",report["stage_decision_metrics"][stage])

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
        self.assertLessEqual(tuning["skip_rate_delta"],.05+1e-12)
        self.assertTrue(tuning["quality_constraint"]["passed"])

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

    def test_calibration_keeps_identity_when_brier_does_not_improve(self):
        probability=np.tile(np.asarray([.4]*5+[.6]*5),60)
        labels=np.tile(np.asarray([0,0,0,1,1,0,0,1,1,1],dtype=np.int8),60)
        features=np.zeros((len(probability),57),dtype=np.float32);features[:,0]=probability
        calibration=final.fit_probability_calibration(VectorClassifier(),features,labels,shoe_ids=[f"shoe-{index//10}" for index in range(len(labels))])
        self.assertEqual(calibration["method"],"identity")
        self.assertIn("identity_brier",calibration["selection"])

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
