import unittest
import tempfile
import warnings
from pathlib import Path
import numpy as np

from particle_shoe_filter import (
    ParticleShoeTracker,
    _banker_draws as particle_banker_draws,
    early35_evidence_weight,
    early35_physical_ev_reliability,
    estimate_particle_physics,
    fuse_particle_physics,
)
from physics_feature_extractor import (
    AUXILIARY_TARGET_DIM,
    AUXILIARY_TARGET_SLICES,
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    THIRD_CARD_NONE,
    Card,
    HandResult,
    OfflineBaccaratSimulator,
    PhysicsFeatureExtractor,
    _banker_draws as physics_banker_draws,
    apply_uncertainty_calibration,
    augment_213d,
    build_auxiliary_draw_target,
    deal_baccarat_hand,
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
        self.assertEqual(data.x.shape[1],HISTORY_INPUT_DIM)
        self.assertEqual(data.auxiliary_targets.shape[1],AUXILIARY_TARGET_DIM)
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

    def test_draw_auxiliary_targets_follow_standard_baccarat_sequence(self):
        def hand(player_initial,banker_initial,player_third=None,banker_third=None,natural=False):
            cards=tuple(Card(1,0) for _ in range(4+int(player_third is not None)+int(banker_third is not None)))
            return HandResult("B",0,0,cards,player_initial,banker_initial,natural,player_third,banker_third)
        natural_shoe=[Card(8,0),Card(2,0),Card(10,0),Card(3,0),Card(1,0),Card(1,0)]
        natural,_=deal_baccarat_hand(natural_shoe,0)
        self.assertTrue(natural.natural_8_9)
        self.assertEqual(natural.card_count,4)
        self.assertIsNone(natural.player_third_card_value)
        self.assertIsNone(natural.banker_third_card_value)
        natural_target=build_auxiliary_draw_target(natural)
        self.assertEqual(float(natural_target[AUXILIARY_TARGET_SLICES["player_draw"]][0]),0.0)
        self.assertEqual(float(natural_target[AUXILIARY_TARGET_SLICES["banker_draw"]][0]),0.0)
        player_draw_shoe=[Card(2,0),Card(7,0),Card(3,0),Card(10,0),Card(4,0),Card(1,0)]
        player_draw,_=deal_baccarat_hand(player_draw_shoe,0)
        self.assertEqual(player_draw.player_initial_total,5)
        self.assertEqual(player_draw.player_third_card_value,4)
        self.assertEqual(float(build_auxiliary_draw_target(player_draw)[AUXILIARY_TARGET_SLICES["player_draw"]][0]),1.0)
        for total in range(10):
            for third in (None,*range(10)):
                self.assertEqual(particle_banker_draws(total,third),physics_banker_draws(total,third))
        cases=(
            (hand(6,7),0),
            (hand(5,7,3),1),
            (hand(6,5,None,4),2),
            (hand(4,3,9,6),3),
        )
        for sample,draw_class in cases:
            target=build_auxiliary_draw_target(sample)
            self.assertEqual(int(np.argmax(target[AUXILIARY_TARGET_SLICES["draw_consistency"]])),draw_class)
            self.assertEqual(sample.card_count,4+(draw_class in {1,2})+2*(draw_class==3))
        third_target=build_auxiliary_draw_target(hand(5,6,9,None))
        self.assertEqual(int(np.argmax(third_target[AUXILIARY_TARGET_SLICES["player_third_card_value"]])),9)
        self.assertEqual(int(np.argmax(third_target[AUXILIARY_TARGET_SLICES["banker_third_card_value"]])),THIRD_CARD_NONE)
        context=6*11+9
        self.assertEqual(float(third_target[AUXILIARY_TARGET_SLICES["banker_draw_context"].start+context]),-1.0)

    def test_auxiliary_heads_preserve_213d_to_48d_production_contract(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(24,HISTORY_INPUT_DIM)).astype(np.float32)
        y=rng.normal(size=(24,PHYSICS_DIM)).astype(np.float32)
        aux=np.zeros((24,AUXILIARY_TARGET_DIM),dtype=np.float32)
        aux[:,AUXILIARY_TARGET_SLICES["player_draw"]]=np.arange(24,dtype=np.float32).reshape(-1,1)%2
        model=PhysicsFeatureExtractor(random_state=9)
        model.model.set_params(max_iter=1,early_stopping=False,batch_size=24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x,y,auxiliary_targets=aux)
        self.assertEqual(model._decode_scaled(model.model.predict(model.scaler.transform(x))).shape,(24,PHYSICS_DIM))
        with tempfile.TemporaryDirectory() as directory:
            bundle=model.export_browser_bundle(Path(directory)/"physics.json")
        self.assertEqual(bundle["history_input_dim"],HISTORY_INPUT_DIM)
        self.assertEqual(bundle["physics_dim"],PHYSICS_DIM)
        self.assertEqual(len(bundle["intercepts"][-1]),PHYSICS_DIM)

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

        self.assertGreater(late_diag["effective_progress_round"],early_diag["effective_progress_round"])
        self.assertGreater(late_diag["physical_ev_reliability"],early_diag["physical_ev_reliability"])
        self.assertGreater(late_diag["fusion_weight"],early_diag["fusion_weight"])
        self.assertLessEqual(late_diag["fusion_weight"],.58)
        self.assertLessEqual(abs(float(early[39])),abs(float(early_diag["raw_physical_ev_banker"]))+1e-9)

    def test_early35_evidence_is_reliability_aware_and_continuous(self):
        reliable={"recent_ess_ratio":.90,"posterior_uncertainty":.20,"expected_consumed_cards":180.0}
        uncertain={**reliable,"posterior_uncertainty":.80}
        low_ess={**reliable,"recent_ess_ratio":.20}
        weights=[early35_evidence_weight(rounds,reliable) for rounds in (5,15,25,35)]
        self.assertLess(weights[0],weights[1])
        self.assertLess(weights[1],weights[2])
        self.assertLess(weights[2],weights[3])
        self.assertLess(early35_evidence_weight(10,uncertain),early35_evidence_weight(10,reliable))
        self.assertGreater(early35_evidence_weight(10,reliable),early35_evidence_weight(10,low_ess))
        self.assertLess(abs(early35_evidence_weight(21,reliable)-early35_evidence_weight(20,reliable)),.10)
        self.assertLess(abs(early35_evidence_weight(36,reliable)-early35_evidence_weight(35,reliable)),.10)
        self.assertLess(abs(early35_physical_ev_reliability(36,reliable)-early35_physical_ev_reliability(35,reliable)),.10)
        mlp=np.zeros(PHYSICS_DIM,dtype=np.float32);mlp[:3]=[.58,.34,.08];mlp[3:13]=.1;mlp[13:23]=.1;mlp[23:26]=[.4586,.4462,.0952];mlp[26:39]=5/13;mlp[39:43]=[0,0,0,.5];mlp[44:46]=.5
        fused,_=fuse_particle_physics(mlp,"BPPBTBBPPT",early35_version=1)
        self.assertEqual(fused.shape,(PHYSICS_DIM,))

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
    build_auxiliary_draw_target,
    deal_baccarat_hand,
