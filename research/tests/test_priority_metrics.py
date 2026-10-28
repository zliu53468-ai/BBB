import unittest
from research.priority_report import runs,streak_report,metrics,blank
class PriorityMetricsTests(unittest.TestCase):
    def test_maximal_and_overlapping_runs(self):
        h=runs([True]*3+[False]*5+[True]*7)
        self.assertEqual(dict(h[True]),{3:1,7:1});self.assertEqual(dict(h[False]),{5:1})
        z=streak_report(h[False],15)
        self.assertEqual(z['thresholds']['3']['maximal_run_count'],1)
        self.assertEqual(z['thresholds']['3']['overlapping_windows'],3)
        self.assertEqual(z['thresholds']['3']['mean_run_length'],5)
    def test_clock_skips_break_but_placed_does_not(self):
        self.assertEqual(max(runs([False,None,False,None,False])[False]),1)
        self.assertEqual(max(runs([False,False,False])[False]),3)
    def test_zero_recommendations_not_fake_perfect(self):
        a=blank();a['hands']=80;z=metrics(a)
        self.assertIsNone(z['hit_on_recommendations_excluding_ties']);self.assertEqual(z['action_rate'],0)
        self.assertIsNone(z['EV_per_recommendation']);self.assertEqual(z['EV_per_opportunity'],0)
