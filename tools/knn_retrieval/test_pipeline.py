import copy
import unittest

from generate_synthetic import generate
from pipeline import aggregate, build_signals, metrics, validate


class PipelineTests(unittest.TestCase):
    def test_prefix_conservation_and_holdout_isolation(self):
        payload=generate(20)
        shoes, groups, split=validate(payload)
        self.assertEqual([len(groups[g]) for g in groups], [14,3,3])
        signals, library=build_signals(shoes,groups,30,500000,20261029)
        self.assertEqual(library['rows'],14*59)
        for s in signals.values():
            self.assertEqual(s['b_count']+s['p_count']+s['tie_count'],30)
            self.assertTrue(all(k.rsplit(':',1)[0] in groups['library'] for k in s['neighbor_keys']))
        changed=copy.deepcopy(payload)
        # Future label changes cannot change lookup keys/distances/neighbors.
        for r in changed['rows']:
            if r['shoe_id'] in groups['validation_test']:r['outcome']='T'
        altered, _, _=validate(changed)
        signals2,_=build_signals(altered,groups,30,500000,20261029)
        self.assertEqual(signals,signals2)
        changed=copy.deepcopy(payload); changed['rows'][1]['remaining_counts_before'][0]+=1
        with self.assertRaisesRegex(ValueError,'conservation'):validate(changed)

    def test_streak_windows_and_zero_bet_semantics(self):
        rows=[dict(direction='B',outcome=o) for o in ['B','B','T','B','B','P','P']]
        m=metrics(rows)
        self.assertEqual(m['max_losses'],2)
        self.assertEqual(m['has_four_streak'],1)
        self.assertEqual(m['standard_hit_rate'],4/6)
        self.assertEqual(m['five_coverage'],1/4)
        self.assertEqual(m['six_coverage'],2/3)
        none=metrics([dict(direction='Skip',outcome='P')]*59)
        self.assertFalse(none['hit_rate_estimable'])
        self.assertEqual(aggregate([none])['pct_shoes_max_losses_le_2'],1)


if __name__=='__main__':unittest.main()
