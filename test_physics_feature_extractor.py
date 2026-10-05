import unittest
import numpy as np

from particle_shoe_filter import ParticleShoeTracker, estimate_particle_physics, fuse_particle_physics
from physics_feature_extractor import (
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    OfflineBaccaratSimulator,
    apply_uncertainty_calibration,
    augment_213d,
    fit_uncertainty_calibration,
    history_to_vector,
    physics_uncertainty_proxy,
    prepare_xgboost_input,
    sanitize_physics_prediction,
)


class FakePhysics:
    def predict_features(self, history):
        return np.linspace(0.0, 1.0, PHYSICS_DIM, dtype=np.float32)


class PhysicsFeatureExtractorTests(unittest.TestCase):
    def test_history_vector_shape_and_determinism(self):
        a=history_to_vector("BPTBBP")
        b=history_to_vector(["B","P","T","B","B","P"])
        self.assertEqual(a.shape,(HISTORY_INPUT_DIM,))
        self.assertTrue(np.array_equal(a,b))

    def test_simulator_targets_are_physically_consistent(self):
        data=OfflineBaccaratSimulator(random_state=1234,max_hands_per_shoe=6).generate(3)
        self.assertEqual(data.y.shape[1],PHYSICS_DIM)
        for row in data.y:
            self.assertAlmostEqual(float(row[:3].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[3:13].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[13:23].sum()),1.0,places=6)
            self.assertAlmostEqual(float(row[23:26].sum()),1.0,places=6)
            self.assertGreaterEqual(float(row[26:39].sum()),4.0)
            self.assertLessEqual(float(row[26:39].sum()),6.0)
            p_b,p_p=float(row[23]),float(row[24])
            self.assertAlmostEqual(float(row[39]),p_b*.95-p_p,places=6)
            self.assertAlmostEqual(float(row[40]),p_p-p_b,places=6)
            self.assertAlmostEqual(float(row[41]),float(row[39]-row[40]),places=6)
            self.assertGreaterEqual(float(row[42]),0.0)
            self.assertLessEqual(float(row[42]),1.0)
            self.assertGreaterEqual(float(row[43]),0.0)
            self.assertLessEqual(float(row[43]),416.0)

    def test_prepare_xgboost_input_preserves_original_7d(self):
        original=np.asarray([0.51,12,60,0.8,0.5,2,1],dtype=np.float32)
        merged=prepare_xgboost_input(0.51,original,"BPPBT",extractor=FakePhysics())
        self.assertEqual(merged.shape,(56,))
        self.assertAlmostEqual(float(merged[0]),0.51,places=6)
        self.assertTrue(np.array_equal(merged[1:8],original))
        self.assertEqual(merged[8:].shape,(PHYSICS_DIM,))

    def test_213d_augmentation_preserves_shape_and_targets(self):
        x=np.zeros((6,HISTORY_INPUT_DIM),dtype=np.float32)
        y=np.zeros((6,PHYSICS_DIM),dtype=np.float32)
        y[:,0]=1.0
        y[np.arange(6),23+(np.arange(6)%3)]=1.0
        augmented_x,augmented_y=augment_213d(x,y,ratio=.5,random_state=7)
        self.assertEqual(augmented_x.shape,(9,HISTORY_INPUT_DIM))
        self.assertEqual(augmented_y.shape,(9,PHYSICS_DIM))
        self.assertTrue(np.array_equal(augmented_x[:6],x))
        self.assertTrue(np.array_equal(augmented_y[:6],y))

    def test_calibrated_physics_probability_blocks_stay_normalized(self):
        raw=np.linspace(-1.0,2.0,PHYSICS_DIM,dtype=np.float32)
        output=sanitize_physics_prediction(raw,{"card_count":.8,"winner":1.2})
        for block in (output[:3],output[3:13],output[13:23],output[23:26]):
            self.assertAlmostEqual(float(np.sum(block)),1.0,places=6)
        self.assertGreaterEqual(float(output[39]),-1.0)
        self.assertLessEqual(float(output[39]),.95)
        self.assertGreaterEqual(float(output[42]),0.0)
        self.assertLessEqual(float(output[42]),1.0)

    def test_particle_shoe_estimator_is_deterministic_and_rule_consistent(self):
        history="BPPBTBBPPTBBPPBTBP"
        a=estimate_particle_physics(history)
        b=estimate_particle_physics(history)
        self.assertTrue(np.array_equal(a.physics_48d,b.physics_48d))
        self.assertEqual(a.physics_48d.shape,(PHYSICS_DIM,))
        for block in (a.physics_48d[:3],a.physics_48d[3:13],a.physics_48d[13:23],a.physics_48d[23:26]):
            self.assertAlmostEqual(float(np.sum(block)),1.0,places=6)
        p_b,p_p=float(a.physics_48d[23]),float(a.physics_48d[24])
        self.assertAlmostEqual(float(a.physics_48d[39]),p_b*.95-p_p,places=6)
        self.assertAlmostEqual(float(a.physics_48d[40]),p_p-p_b,places=6)
        self.assertAlmostEqual(float(a.physics_48d[41]),float(a.physics_48d[39]-a.physics_48d[40]),places=6)
        self.assertAlmostEqual(float(a.physics_48d[42]),a.diagnostics["posterior_uncertainty"],places=6)
        expected_cards=float(4*a.physics_48d[0]+5*a.physics_48d[1]+6*a.physics_48d[2])
        self.assertAlmostEqual(float(np.sum(a.physics_48d[26:39])),expected_cards,places=5)
        self.assertGreater(a.diagnostics["expected_consumed_cards"],4*len(history))
        self.assertLess(a.diagnostics["expected_consumed_cards"],6*len(history))
        self.assertGreaterEqual(a.diagnostics["posterior_uncertainty"],0.0)
        self.assertLessEqual(a.diagnostics["posterior_uncertainty"],1.0)

    def test_incremental_particle_tracker_matches_same_prefix_contract(self):
        tracker=ParticleShoeTracker()
        tracker.estimate("BPPB")
        incremental=tracker.estimate("BPPBTBBP")
        fresh=estimate_particle_physics("BPPBTBBP")
        self.assertEqual(incremental.physics_48d.shape,fresh.physics_48d.shape)
        self.assertAlmostEqual(incremental.diagnostics["history_rounds"],8.0,places=6)
        self.assertGreater(incremental.diagnostics["expected_consumed_cards"],32.0)
        self.assertLess(incremental.diagnostics["expected_consumed_cards"],48.0)

    def test_particle_fusion_preserves_48d_contract_and_uses_more_physics_late(self):
        mlp=np.zeros(PHYSICS_DIM,dtype=np.float32)
        mlp[:3]=[.58,.34,.08];mlp[3:13]=.1;mlp[13:23]=.1;mlp[23:26]=[.4586,.4462,.0952]
        mlp[26:39]=5/13;mlp[39:43]=[0.0,0.0,0.0,.5];mlp[43]=0;mlp[44:46]=.5
        early,early_diag=fuse_particle_physics(mlp,"BPPB")
        late,late_diag=fuse_particle_physics(mlp,"BPPBTBBPPTBBPPBTBPBPPBTBBPPTBBPPBTBPBPPBTBBPPTBBPPBTBP")
        self.assertEqual(early.shape,(PHYSICS_DIM,))
        self.assertEqual(late.shape,(PHYSICS_DIM,))
        self.assertGreater(late_diag["fusion_weight"],early_diag["fusion_weight"])
        self.assertAlmostEqual(float(np.sum(late[:3])),1.0,places=6)
        self.assertAlmostEqual(float(np.sum(late[23:26])),1.0,places=6)
        self.assertAlmostEqual(float(late[39]),late_diag["physical_ev_banker"],places=6)
        self.assertAlmostEqual(float(late[40]),late_diag["physical_ev_player"],places=6)
        self.assertAlmostEqual(float(late[41]),late_diag["physical_ev_gap"],places=6)
        self.assertGreaterEqual(float(late[42]),0.0)
        self.assertLessEqual(float(late[42]),1.0)

    def test_uncertainty_calibration_is_bounded_and_monotone(self):
        base=np.zeros((64,PHYSICS_DIM),dtype=np.float32)
        base[:,0]=1.0;base[:,3]=1.0;base[:,13]=1.0;base[:,23]=1.0;base[:,42]=.5
        truth=base.copy()
        for i in range(64):
            base[i,23:26]=[1-i/126.0,i/126.0,0.0]
            truth[i,23:26]=[1.0,0.0,0.0]
        rounds=np.linspace(10,70,64)
        calibration=fit_uncertainty_calibration(base,truth,rounds)
        proxy=np.asarray([physics_uncertainty_proxy(row,rnd) for row,rnd in zip(base,rounds)])
        calibrated=apply_uncertainty_calibration(proxy,calibration)
        self.assertTrue(np.all((calibrated>=0)&(calibrated<=1)))
        order=np.argsort(proxy)
        self.assertTrue(np.all(np.diff(calibrated[order])>=-1e-9))


if __name__=="__main__":
    unittest.main()
