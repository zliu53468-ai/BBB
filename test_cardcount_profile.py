import argparse
import unittest
from unittest.mock import patch
import numpy as np
from cardcount_profile import PROFILE, interpolate
from physics_feature_extractor import PhysicsFeatureExtractor, PHYSICS_LOSS_WEIGHTS
from xgb_final_probability import balanced_sample_weights, build_parser, train_command, select_xgboost_model, XGB_TUNING_CANDIDATES


class CardcountProfileTests(unittest.TestCase):
    def test_continuous_boundaries(self):
        self.assertEqual(interpolate("decision_alpha", 10), .45)
        self.assertEqual(interpolate("decision_alpha", 30), .55)
        self.assertEqual(interpolate("decision_alpha", 60), .65)
        for boundary in (20, 45):
            self.assertLess(abs(interpolate("decision_alpha", boundary+.001)-interpolate("decision_alpha", boundary-.001)), .001)

    def test_training_weights_use_observable_features(self):
        x=np.zeros((6,57));x[:,2]=[10,30,60,10,30,60]
        # Low uncertainty and equal class counts: later rows receive more emphasis.
        y=np.array([0,0,0,1,1,1])
        w=balanced_sample_weights(y,x,training_profile="aggressive_cardcount")
        self.assertLess(w[0],w[1]);self.assertLess(w[1],w[2])
        altered=[{"actual_outcome":"B"},{"actual_outcome":"P"}]*3
        np.testing.assert_array_equal(w,balanced_sample_weights(y,x,altered,training_profile="aggressive_cardcount"))
        self.assertTrue(np.all((w>=.35)&(w<=1.50)))

    def test_physics_profile_is_opt_in(self):
        default=PhysicsFeatureExtractor();candidate=PhysicsFeatureExtractor(training_profile="aggressive_cardcount")
        np.testing.assert_allclose(default.loss_scale**2,PHYSICS_LOSS_WEIGHTS,rtol=1e-6)
        self.assertEqual(len(candidate.loss_scale),48)
        self.assertAlmostEqual(float(candidate.loss_scale[23]**2),3.50,places=5)
        self.assertLessEqual(float(candidate.loss_scale[26]**2),.56)

    def test_extra_search_candidates_are_used(self):
        x=np.zeros((8,57));x[:,2]=30
        y=np.array([0,1]*4);train=np.array([True]*4+[False]*4);validation=~train
        class Model:
            def fit(self,*args,**kwargs): return self
        probabilities=np.array([.2,.8,.2,.8])
        with patch("xgb_final_probability.build_xgboost_classifier",return_value=Model()), patch("xgb_final_probability.bounded_probabilities",return_value=(probabilities,probabilities,probabilities)):
            _,_,reports=select_xgboost_model(x,y,train,validation,training_profile="aggressive_cardcount",trials=100)
        self.assertEqual(len(reports),len(XGB_TUNING_CANDIDATES)+len(PROFILE["training"]["xgb_candidates"]))

    def test_candidate_cannot_overwrite_production(self):
        args=argparse.Namespace(training_profile="aggressive_cardcount",output="final_probability_model.json")
        with self.assertRaisesRegex(SystemExit,"protected"):
            train_command(args)
        args=build_parser().parse_args(["train","--input","rows.json"])
        self.assertEqual(args.training_profile,"production")
        self.assertEqual(PROFILE["training"]["pattern_gain"],0)


if __name__=="__main__":
    unittest.main()
