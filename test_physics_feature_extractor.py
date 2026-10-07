import unittest
import tempfile
import warnings
from pathlib import Path
import numpy as np

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
        self.assertTrue(physics_banker_draws(3,7))
        self.assertFalse(physics_banker_draws(3,8))
        self.assertTrue(physics_banker_draws(5,4))
        self.assertFalse(physics_banker_draws(6,5))
        self.assertTrue(physics_banker_draws(6,6))
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

    def test_direct_physics_inference_preserves_48d_without_particle_filter(self):
        rng=np.random.default_rng(13)
        x=rng.normal(size=(32,HISTORY_INPUT_DIM)).astype(np.float32)
        y=np.zeros((32,PHYSICS_DIM),dtype=np.float32)
        y[:,0]=1.0;y[:,3]=1.0;y[:,13]=1.0;y[:,23]=1.0
        y[:,26:39]=4.0/13.0
        y[:,39]=.95;y[:,40]=-1.0;y[:,41]=1.95;y[:,42]=.5
        y[:,43]=np.linspace(0,120,32);y[:,44:46]=.5
        model=PhysicsFeatureExtractor(random_state=13)
        model.model.set_params(max_iter=1,early_stopping=False,batch_size=32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x,y)
        physics,diagnostics=model.predict_features_with_diagnostics("BPPBTBBP")
        self.assertEqual(physics.shape,(PHYSICS_DIM,))
        self.assertEqual(diagnostics["particle_filter_enabled"],0.0)
        self.assertEqual(diagnostics["physics_direct_version"],1.0)
        self.assertGreaterEqual(float(physics[42]),0.0)
        self.assertLessEqual(float(physics[42]),1.0)
        self.assertAlmostEqual(float(np.sum(physics[:3])),1.0,places=6)
        self.assertAlmostEqual(float(np.sum(physics[23:26])),1.0,places=6)

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
