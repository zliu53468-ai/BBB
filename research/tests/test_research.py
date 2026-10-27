"""python -m unittest discover -s research/tests -v"""
import unittest
import numpy as np
from research.baccarat import INITIAL, _banker_draw, exact_distribution, simulate_shoe, deal
from research.features import vector, DIM
from research.run import split, bootstrap, shoe_rows

class TestBaccarat(unittest.TestCase):
    def test_banker_third_card_rule_all_rows(self):
        for banker_total in range(8):
            for player_third in range(10):
                should=(banker_total<=2 or
                        banker_total==3 and player_third!=8 or
                        banker_total==4 and 2<=player_third<=7 or
                        banker_total==5 and 4<=player_third<=7 or
                        banker_total==6 and 6<=player_third<=7)
                self.assertEqual(bool(_banker_draw(banker_total,player_third)),bool(should))
            self.assertEqual(bool(_banker_draw(banker_total,-1)),banker_total<=5)

    def test_initial_eight_deck_probabilities(self):
        v=np.asarray(exact_distribution(INITIAL))
        self.assertAlmostEqual(sum(v[:3]),1,places=9)
        self.assertAlmostEqual(sum(v[3:6]),1,places=9)
        self.assertAlmostEqual(sum(v[6:16]),1,places=9)
        self.assertAlmostEqual(sum(v[16:26]),1,places=9)
        for got,expected in zip(v[:3],(.4586,.4462,.0952)):
            self.assertLess(abs(got-expected),.002)

    def test_all_zero_cards_always_tie_natural_stand(self):
        v=exact_distribution((6,0,0,0,0,0,0,0,0,0))
        self.assertAlmostEqual(v[2],1,places=10)
        self.assertAlmostEqual(v[3],1,places=10)

    def test_reproducible_shoe_and_cut(self):
        a=list(simulate_shoe(7,seed=20261027))
        b=list(simulate_shoe(7,seed=20261027))
        self.assertEqual(a,b)
        self.assertGreater(len(a),30)
        self.assertEqual([r["round"] for r in a],list(range(1,len(a)+1)))
        self.assertTrue(all(len(r["ground_truth"])==27 for r in a))
        for i,r in enumerate(a):
            self.assertEqual(r["history"],"".join(item["outcome"] for item in a[:i]))
            self.assertEqual(r["remaining_before"]-r["cards"],
                             a[i+1]["remaining_before"] if i+1<len(a) else
                             r["remaining_before"]-r["cards"])

    def test_213D_feature_offset_and_no_future(self):
        x=vector("BPT")
        self.assertEqual(x.shape,(DIM,))
        self.assertEqual(float(x[61*3]),1)
        self.assertEqual(float(x[62*3+1]),1)
        self.assertEqual(float(x[63*3+2]),1)
        self.assertEqual(sum(x[:192]),3)
        with self.assertRaises(ValueError):vector("BP?")

    def test_whole_shoe_hash_split(self):
        self.assertEqual(split(101),split(101))
        self.assertTrue(set(split(i) for i in range(100))=={0,1,2})

    def test_cluster_bootstrap_detects_no_bets(self):
        rows=shoe_rows(np.repeat(np.arange(30),2),
                       np.tile(np.asarray([0,1]),30),
                       np.tile(np.asarray([[.46,.44,.10]]),[60,1]),
                       np.full(60,-1,dtype=np.int8))
        self.assertIsNone(bootstrap(rows))

if __name__=="__main__":
    unittest.main()
