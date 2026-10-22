import unittest
import numpy as np
from xgb_final_probability import balanced_sample_weights, dynamic_ema_alpha


class PatternTrainingTests(unittest.TestCase):
    def test_structure_weight_is_symmetric_and_causal(self):
        x=np.zeros((8,57));x[:,2]=30
        x[:,5]=[.5,.5,.5,.5,.95,.95,.05,.05]
        y=np.array([0,1]*4)
        original=balanced_sample_weights(y,x,pattern_weight=0)
        weighted=balanced_sample_weights(y,x,pattern_weight=.5)
        self.assertGreater(weighted[4],weighted[0])
        self.assertEqual(weighted[4],weighted[6])
        self.assertTrue(np.all(np.isfinite(weighted)))
        x[:,2]=3
        np.testing.assert_array_equal(balanced_sample_weights(y,x,pattern_weight=.5),balanced_sample_weights(y,x,pattern_weight=0))
        for value in [-1,2,float('nan')]:
            with self.assertRaises(ValueError):balanced_sample_weights(y,x,pattern_weight=value)

    def test_alpha_anchors_and_invalid_fallback(self):
        baseline=dynamic_ema_alpha(30,.5,{},.125)
        self.assertEqual(dynamic_ema_alpha(30,.5,{'progress_alpha_anchors':[[0,.6],[1,.6]]},.125),.6)
        for invalid in [[],[[1,.6],[0,.6]],[[0,0],[1,1]],[[0,float('nan')],[1,.6]]]:
            self.assertEqual(dynamic_ema_alpha(30,.5,{'progress_alpha_anchors':invalid},.125),baseline)


if __name__=='__main__':unittest.main()
