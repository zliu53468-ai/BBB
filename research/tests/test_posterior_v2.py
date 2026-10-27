"""Physics posterior V2 完整可重現且不依賴正式權重。"""
import unittest
import numpy as np
from research.physics_posterior_v2 import PosteriorTeacher
from research.features import vector

class TestPosteriorTeacher(unittest.TestCase):
    def test_train_label_information_never_used_in_prediction(self):
        x=np.stack([vector("BBBBB"),vector("PPPPP"),vector("BBBBB"),vector("TTBBB")])
        p=np.asarray([[.47,.43,.10],[.44,.46,.10],[.46,.45,.09],[.45,.45,.10]])
        c=np.tile(np.asarray([128/416]+[32/416]*9),(4,1))
        teacher=PosteriorTeacher(context_length=5,prior_strength=200).fit(x,p,c)
        one=teacher.predict(x)
        self.assertEqual(one["p"].shape,(4,3))
        np.testing.assert_allclose(one["p"].sum(axis=1),1,atol=1e-10)
        np.testing.assert_allclose(one["remaining_point_probabilities"].sum(axis=1),1,atol=1e-10)
        np.testing.assert_array_equal(one["p"][0],one["p"][2])
        # 改動測試時未來的 outcome、組成標籤不會改變 predict API；其只有 x 輸入。
        self.assertEqual(teacher.predict_proba(x).shape,(4,3))

    def test_unseen_history_uses_train_prior(self):
        x=np.stack([vector("BBBBB"),vector("PPPPP")])
        p=np.asarray([[.46,.45,.09],[.45,.46,.09]])
        c=np.tile(np.asarray([128/416]+[32/416]*9),(2,1))
        model=PosteriorTeacher(context_length=5,prior_strength=200).fit(x,p,c)
        pred=model.predict(np.stack([vector("BPBPB")]))
        self.assertEqual(int(pred["bucket_train_support"][0]),0)
        np.testing.assert_allclose(pred["p"][0],p.mean(axis=0),atol=1e-12)

    def test_rejects_bad_labels(self):
        x=np.stack([vector("B")])
        with self.assertRaises(ValueError):
            PosteriorTeacher().fit(x,np.zeros((1,3)),np.zeros((1,10)))

if __name__=="__main__":
    unittest.main()
