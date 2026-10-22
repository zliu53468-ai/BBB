import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
from macro_ema_features import MacroEMA, PROFILES, resolve_config
import xgb_final_probability as final

ROOT=Path(__file__).resolve().parent


class ResponsiveUpgradeTests(unittest.TestCase):
    def test_browser_python_entry_numbers_match(self):
        actual=json.loads(subprocess.check_output(['node','test_responsive_upgrade.js','--parity'],cwd=ROOT))
        config=final.DECISION_POLICY_PROFILES['aggressive_entry_v2']
        for row in actual:
            args=np.array([row['round']]),np.array([row['noise']])
            minimum,activation,soft=final.decision_policy_arrays(*args,final.DEFAULT_MIN_EV,enabled=True,policy_config=config)
            band,_=final.confidence_band_arrays(*args,config)
            self.assertAlmostEqual(row['alpha'],final.dynamic_ema_alpha(row['round'],row['noise'],final.RUNTIME_PROFILES['final_smoothing'],.125,{'physics_uncertainty':.2,'signal_distance':.06}),places=12)
            for key,value in [('min',minimum),('activation',activation),('soft',soft),('band',band)]:
                self.assertAlmostEqual(row[key],float(value[0]),places=12)

    def test_profile_snapshot_parity(self):
        rows=[{'outcome':'B','card_count':4},{'outcome':'P','card_count':6},{'outcome':'T'}, {'outcome':'B'}]*8
        for profile in PROFILES:
            state=MacroEMA('test',profile=profile)
            expected=[state.update(row) for row in rows]
            source="const m=require('./macro_ema.js'),s=m.createState('test',process.argv[1]);console.log(JSON.stringify(JSON.parse(require('fs').readFileSync(0,'utf8')).map(h=>m.update(s,h))));"
            actual=json.loads(subprocess.check_output(['node','-e',source,profile],input=json.dumps(rows).encode(),cwd=ROOT))
            self.assertEqual(actual,expected)

    def test_controlled_signal_reduces_skip_without_chasing_noise(self):
        p=np.full(20,.4905);y=np.array([0,1]*10);rounds=np.arange(10,30);noise=np.full(20,.3)
        config=final.DECISION_POLICY_PROFILES['aggressive_entry_v2']
        _,baseline=final.decision_returns(p,y,rounds,final.DEFAULT_MIN_EV)
        _,candidate=final.decision_returns(p,y,rounds,final.DEFAULT_MIN_EV,noise,policy_enabled=True,policy_config=config)
        self.assertGreater(candidate.sum(),baseline.sum())
        for values,n in [(np.full(20,.5001),noise),(p,np.full(20,.99))]:
            _,bets=final.decision_returns(values,y,rounds,final.DEFAULT_MIN_EV,n,policy_enabled=True,policy_config=config)
            self.assertFalse(bets.any())

    def test_profile_limits_and_cap_match_browser(self):
        config={'profile':'responsive_v1','short_alpha':99,'long_alpha':-1,'max_delta':99,'min_observations':99,'max_age':-1,'progress_start':9,'progress_power':99,'noise_low':.8,'noise_high':.2}
        source="const m=require('./macro_ema.js');console.log(JSON.stringify(m.resolveConfig(JSON.parse(process.argv[1]))));"
        actual=json.loads(subprocess.check_output(['node','-e',source,json.dumps(config)],cwd=ROOT))
        self.assertEqual(actual,resolve_config(config=config))
        self.assertEqual(actual['max_delta'],.045)
        self.assertEqual(actual['short_alpha'],.35)
        self.assertEqual(actual['long_alpha'],.10)

    def test_new_search_ranges_keep_regularization(self):
        ranges={'max_depth':(2,4),'n_estimators':(400,800),'learning_rate':(.015,.03),'min_child_weight':(6,16),'subsample':(.85,.95),'colsample_bytree':(.75,.95),'reg_alpha':(.1,.6),'reg_lambda':(8,18)}
        for candidate in final.XGB_TUNING_CANDIDATES[1:]:
            for key,(low,high) in ranges.items():self.assertTrue(low<=candidate[key]<=high,(key,candidate[key]))


if __name__=='__main__':unittest.main()
