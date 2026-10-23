"""Workflow checks use fabricated fixtures, never model quality evidence."""
import copy,json,tempfile,unittest
from pathlib import Path
from run_middle_late import split_hands,stage_models,ROOT

class WorkflowTests(unittest.TestCase):
    def payload(self,n=1500):
        return {'data_kind':'real_observations','chronological_order':'oldest_to_newest','rows':[{'shoe_id':str(s),'round_index':r,'outcome':'B' if r%2 else 'P'} for s in range(n) for r in range(1,21)]}
    def test_whole_shoe_split(self):
        dev,hold,split=split_hands(self.payload())
        self.assertEqual(len(split['development_shoes']),1200);self.assertEqual(len(split['outer_holdout_shoes']),300)
        self.assertFalse(set(split['development_shoes'])&set(split['outer_holdout_shoes']))
        self.assertEqual(dev['rows'][-1]['shoe_id'],'1199');self.assertEqual(hold['rows'][0]['shoe_id'],'1200')
    def test_missing_rows_and_synthetic_rejected(self):
        data=self.payload();data['rows'].pop(3)
        with self.assertRaisesRegex(ValueError,'Missing'):split_hands(data)
        data=self.payload();data['data_kind']='synthetic'
        with self.assertRaisesRegex(ValueError,'real_observations'):split_hands(data)
    def test_input_cannot_override_frozen_features(self):
        data=self.payload();data['rows'][0].update(features_57d=[999]*57,physics_48d=[999]*48,core_p_b=.99)
        dev,_,_=split_hands(data)
        self.assertNotIn('features_57d',dev['rows'][0]);self.assertNotIn('physics_48d',dev['rows'][0]);self.assertNotIn('core_p_b',dev['rows'][0])
    def test_preserves_pf_and_policy_in_staged_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=out/'physics_candidate.json';x=out/'xgb_candidate.json'
            p.write_bytes((ROOT/'physics_multitask_model.json').read_bytes())
            v=json.loads((ROOT/'final_probability_model.json').read_text(encoding='utf-8'));v['particle_filter_enabled']=False;v['decision_policy']={};v['calibration']={'method':'identity','marker':'candidate'}
            x.write_text(json.dumps(v),encoding='utf-8');stage_models(ROOT,out,p,x)
            staged=json.loads((out/'final_probability_model.json').read_text(encoding='utf-8'));base=json.loads((ROOT/'final_probability_model.json').read_text(encoding='utf-8'))
            self.assertEqual(staged['particle_filter_enabled'],base['particle_filter_enabled']);self.assertEqual(staged['decision_policy'],base['decision_policy']);self.assertEqual(staged['calibration']['marker'],'candidate')
            v['feature_names']=v['feature_names'][::-1];x.write_text(json.dumps(v),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'57D'):stage_models(ROOT,out,p,x)

if __name__=='__main__':unittest.main()
