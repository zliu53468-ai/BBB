import unittest
from unittest.mock import patch
import xgb_final_probability as final
import numpy as np
from xgb_final_probability import balanced_sample_weights, dynamic_ema_alpha, counting_selection_key, pattern_label, build_parser
from physics_feature_extractor import counting_loss_weights, PHYSICS_LOSS_WEIGHTS, PhysicsFeatureExtractor


class PatternTrainingTests(unittest.TestCase):
    def test_structure_weight_is_symmetric_and_causal(self):
        x=np.zeros((8,57));x[:,2]=30
        x[:,5]=[.5,.5,.5,.5,.95,.95,.05,.05]
        y=np.array([0,1]*4)
        original=balanced_sample_weights(y,x,pattern_weight=0)
        weighted=balanced_sample_weights(y,x,pattern_weight=.15)
        self.assertGreater(weighted[4],weighted[0])
        self.assertEqual(weighted[4],weighted[6])
        self.assertTrue(np.all(np.isfinite(weighted)))
        x[:,2]=3
        np.testing.assert_array_equal(balanced_sample_weights(y,x,pattern_weight=.15),balanced_sample_weights(y,x,pattern_weight=0))
        for value in [-1,.16,2,float('nan')]:
            with self.assertRaises(ValueError):balanced_sample_weights(y,x,pattern_weight=value)

    def test_quality_disables_pattern_weight(self):
        x=np.zeros((4,57));x[:,2]=60;x[:,5]=[.5,.95,.05,.5];x[:,50]=1
        y=np.array([0,1,0,1])
        np.testing.assert_array_equal(balanced_sample_weights(y,x,pattern_weight=.15),balanced_sample_weights(y,x,pattern_weight=0))

    def test_brier_has_strict_priority(self):
        poor={"hit_rate":.1,"realized_ev_per_bet":-1,"realized_ev_per_row":-1}
        good={"hit_rate":.99,"realized_ev_per_bet":1,"realized_ev_per_row":1}
        self.assertLess(counting_selection_key(.2,poor,.05),counting_selection_key(.200001,good,-.5))
        self.assertLess(counting_selection_key(.2,good,0),counting_selection_key(.2,poor,-.5))
        args=build_parser().parse_args(['train','--input','unused.json'])
        self.assertEqual(args.pattern_weight,.05)
        self.assertEqual(args.min_skip_reduction,0)

    def test_patterns_report_history_only(self):
        for history,label in [('BBBB','long_run'),('BPBPBP','single_alternation'),('BBPPBBPP','double_alternation'),('BBBP','turn'),('BTB','other')]:
            self.assertEqual(pattern_label(history),label)

    def test_pattern_segments_are_reports_only(self):
        x=np.zeros((4,57),dtype=np.float32);x[:,2]=60;x[:,1]=.8**3
        y=np.array([1,0,1,0]);prob=np.array([.58,.42,.58,.42])
        with patch.object(final,'bounded_probabilities',return_value=(prob,prob,prob)):
            report=final.evaluate(object(),x,y,probability_bounds=(.4,.6),histories=['BBBB','BPBPBP','BBPPBBPP','BBBP'],bootstrap_samples=5)
        segments=report['pattern_segments_report_only']
        self.assertEqual(segments['long_run']['rows'],1)
        self.assertEqual(segments['double_alternation']['rows'],1)
        self.assertNotIn('pattern_constraint_passed',report)

    def test_counting_emphasis_preserves_topology_and_rank(self):
        np.testing.assert_array_equal(counting_loss_weights(),PHYSICS_LOSS_WEIGHTS)
        weights=counting_loss_weights(1.05,1.02,1.02)
        self.assertAlmostEqual(float(weights[23]),3.25*1.05,places=6)
        np.testing.assert_array_equal(weights[26:],PHYSICS_LOSS_WEIGHTS[26:])
        model=PhysicsFeatureExtractor(winner_scale=1.05)
        self.assertEqual(model.model.hidden_layer_sizes,(64,32))
        self.assertEqual(len(model.loss_scale),48)
        for invalid in [0,1.11,float('nan'),True]:
            with self.assertRaises(ValueError):counting_loss_weights(invalid)

    def test_alpha_anchors_and_invalid_fallback(self):
        baseline=dynamic_ema_alpha(30,.5,{},.125)
        self.assertEqual(dynamic_ema_alpha(30,.5,{'progress_alpha_anchors':[[0,.6],[1,.6]]},.125),.6)
        for invalid in [[],[[1,.6],[0,.6]],[[0,0],[1,1]],[[0,float('nan')],[1,.6]]]:
            self.assertEqual(dynamic_ema_alpha(30,.5,{'progress_alpha_anchors':invalid},.125),baseline)


if __name__=='__main__':unittest.main()
