import math
import random
import unittest
from blind_upgrade_reference import exact_initial_matrix, sample_history_consumption, dynamic_threshold


class BlindMath(unittest.TestCase):
    def test_exact_eight_deck_probabilities(self):
        matrix = exact_initial_matrix()
        probs = matrix["outcome_probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1., places=10)
        self.assertAlmostEqual(probs["B"], .458597422632763, places=10)
        self.assertAlmostEqual(probs["T"], .09515596802364, places=10)
        for row in matrix["conditional_card_counts_4_5_6"].values():
            self.assertAlmostEqual(sum(row), 1., places=12)
        self.assertGreater(matrix["conditional_card_counts_4_5_6"]["T"][0], .37)

    def test_consumption_is_legal_and_matches_observation(self):
        rng = random.Random(42)
        counts = [128]+[32]*9
        for actual in "BPT"*80:
            hand = sample_history_consumption(counts, actual, rng)
            self.assertIsNotNone(hand)
            self.assertEqual(hand["outcome"], actual)
            self.assertIn(hand["card_count"], (4, 5, 6))
            self.assertEqual(sum(counts)-sum(hand["remaining"]), hand["card_count"])
            self.assertTrue(all(a >= b >= 0 for a, b in zip(counts, hand["remaining"])))
        self.assertEqual(counts, [128]+[32]*9, "proposal must not mutate parent world")

    def test_depth_and_consensus(self):
        d = dict(particle_count=500, probability_std=.04, probability_mc_se=.003,
                 pre_resample_ess_ratio=.8, ancestor_ratio=.7, conditional_failures=0,
                 forecast_unsupported_particles=0)
        self.assertLess(dynamic_threshold(15, .05, .02, d)["ev"], .05)
        self.assertEqual(dynamic_threshold(16, .05, .02, d)["ev"], .05)
        self.assertAlmostEqual(dynamic_threshold(51, .05, .02, d)["ev"], .05)
        for bad in ({}, {**d, "ancestor_ratio": .01}, {**d, "probability_std": math.nan}):
            self.assertFalse(dynamic_threshold(51, .05, .02, bad)["consensus"])
            self.assertGreater(dynamic_threshold(51, .05, .02, bad)["ev"], .05)


if __name__ == "__main__":
    unittest.main()
