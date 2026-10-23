import argparse
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from cardcount_profile import PROFILE, focus_resample_indices, interpolate
from physics_feature_extractor import PhysicsFeatureExtractor
from xgb_final_probability import balanced_sample_weights, train_command, select_xgboost_model, XGB_TUNING_CANDIDATES, evaluate


class MiddleLateTests(unittest.TestCase):
    def test_early_anchors_and_physics_loss_are_preserved(self):
        old=json.loads(Path(__file__).with_name("validation").joinpath("cardcount_moderate_v1_baseline.json").read_text(encoding="utf8"))
        for r in range(1,21):
            for key,anchors in old["anchors"].items():
                expected=float(np.interp(r,[x for x,_ in anchors],[v for _,v in anchors]))
                self.assertAlmostEqual(interpolate(key,r),expected,places=14)
            self.assertAlmostEqual(interpolate("focus_training_weight",r),interpolate("training_weight",r),places=14)
        np.testing.assert_array_equal(PhysicsFeatureExtractor(training_profile="aggressive_cardcount").loss_scale,PhysicsFeatureExtractor(training_profile="middle_late_focus").loss_scale)

    def test_train_only_resampling_is_deterministic_and_never_duplicates_early(self):
        rounds=list(range(1,71))*4
        indices=focus_resample_indices(rounds,123)
        self.assertEqual(indices,focus_resample_indices(rounds,123))
        self.assertEqual(indices[:len(rounds)],list(range(len(rounds))))
        extras=indices[len(rounds):]
        self.assertTrue(all(0<=i<len(rounds) and rounds[i]>20 for i in extras))
        self.assertEqual(sum(21<=rounds[i]<=45 for i in extras),10)
        self.assertEqual(sum(46<=rounds[i]<=70 for i in extras),20)

    def test_focus_search_and_stage_metrics(self):
        x=np.zeros((12,57));x[:,2]=[10,30,60]*4;y=np.array([0,1]*6)
        train=np.array([True]*6+[False]*6);valid=~train
        class Model:
            def fit(self,*args,**kwargs):return self
        p=np.where(y[valid],.8,.2)
        with patch("xgb_final_probability.build_xgboost_classifier",return_value=Model()), patch("xgb_final_probability.bounded_probabilities",return_value=(p,p,p)):
            _,_,reports=select_xgboost_model(x,y,train,valid,training_profile="middle_late_focus",trials=100)
        self.assertEqual(len(reports),len(XGB_TUNING_CANDIDATES)+len(PROFILE["training"]["xgb_candidates"])+2)
        with patch("xgb_final_probability.bounded_probabilities",return_value=(p,p,p)):
            metrics=evaluate(Model(),x[valid],y[valid],probability_bounds=[.4,.6],bootstrap_samples=0)
        stages=metrics["focus_stage_metrics"]
        self.assertEqual(sum(s["rows"] for s in stages.values()),6)
        for stage in stages.values():self.assertIn("brier",stage)
        weights=balanced_sample_weights(y,x,training_profile="middle_late_focus")
        self.assertTrue(np.all((weights>=.35)&(weights<=1.5)))

    def test_focus_cannot_overwrite_production(self):
        with self.assertRaisesRegex(SystemExit,"protected"):
            train_command(argparse.Namespace(training_profile="middle_late_focus",output="final_probability_model.json"))


if __name__=="__main__":unittest.main()
