import unittest
from preflight import audit


class AuditTests(unittest.TestCase):
    def test_absence_is_not_negative_evidence(self):
        r = audit()
        self.assertEqual(r['status'], 'INSUFFICIENT_SAMPLE')
        self.assertIsNone(r['no_increment'])
        self.assertIsNone(r['significance'])
        for row in r['comparisons']:
            self.assertIsNone(row['pct_shoes_max_losses_le_2'])

    def test_split_and_out_of_range(self):
        rows = []
        for s in range(11):
            for i in range(49 if s == 10 else 60):
                rows.append(dict(shoe_id=str(s), round_index=i+1, outcome='B',
                                 shoe_start=f'2026-10-{s+1:02}T00:00:00Z'))
        r = audit(dict(data_kind='real_observations', source='fixture',
                       chronological_order='oldest_to_newest', rows=rows))
        groups = [x['split'] for x in r['split']]
        self.assertEqual(groups.count('library'), 7)
        self.assertEqual(groups.count('validation_fit'), 1)
        self.assertEqual(groups.count('validation_test'), 2)
        self.assertEqual(r['split'][-1]['status'], 'OUT_OF_RANGE_SHOE')
        self.assertIn('MISSING_CAUSAL_COMPOSITION', r['failures'])
        self.assertFalse(r['fitting_executed'])


if __name__ == '__main__':
    unittest.main()
