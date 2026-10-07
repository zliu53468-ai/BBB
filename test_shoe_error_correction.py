import unittest

import numpy as np

from particle_shoe_filter import (
    PHYSICS_DIM,
    ParticleShoeTracker,
    _apply_shoe_error_correction,
    _empty_shoe_error_correction,
    fuse_particle_physics,
)


def _snapshot(winner, *, gap=.08, cards=(.25, .50, .25), points=None, expected_cards=5.0):
    point_distribution = list(points or [0.1] * 10)
    return {
        "available": True,
        "winner_distribution": list(winner),
        "cards_distribution": list(cards),
        "player_final_point_distribution": point_distribution,
        "banker_final_point_distribution": point_distribution,
        "expected_cards_consumed": expected_cards,
        "physical_ev_gap": gap,
        "particle_uncertainty": .10,
        "recent_ess_ratio": .90,
    }


class ShoeErrorCorrectionTests(unittest.TestCase):
    def _update(self, prior, actual, *, ess=.9, uncertainty=.1, state=None, memory=()):
        posterior = _snapshot(
            [.20, .70, .10],
            cards=(.20, .50, .30),
            points=[.20] + [.80 / 9] * 9,
            expected_cards=5.2,
        )
        return _apply_shoe_error_correction(
            state or _empty_shoe_error_correction(),
            memory,
            prior_snapshot=prior,
            posterior_hidden_given_actual=posterior,
            actual=actual,
            recent_ess_ratio=ess,
            particle_uncertainty=uncertainty,
        )

    def test_surprise_is_probability_sensitive(self):
        ordinary, _ = self._update(_snapshot([.45, .45, .10]), "P")
        severe, _ = self._update(_snapshot([.72, .20, .08]), "P", ess=.2, uncertainty=.8)
        self.assertLess(ordinary["prediction_surprise"], severe["prediction_surprise"])
        self.assertAlmostEqual(ordinary["actual_probability"], .45, places=6)
        self.assertAlmostEqual(severe["actual_probability"], .20, places=6)

    def test_health_is_conservative_asymmetric_and_recovers(self):
        ordinary, ordinary_memory = self._update(_snapshot([.45, .45, .10]), "P")
        severe, severe_memory = self._update(_snapshot([.72, .20, .08]), "P", ess=.2, uncertainty=.8)
        self.assertGreater(ordinary["shoe_posterior_health"], .90)
        self.assertLess(severe["shoe_posterior_health"], ordinary["shoe_posterior_health"])

        current, memory = severe, severe_memory
        for _ in range(5):
            current, memory = self._update(
                _snapshot([.72, .20, .08]), "P", ess=.2, uncertainty=.8, state=current, memory=memory,
            )
        self.assertLess(current["shoe_posterior_health"], severe["shoe_posterior_health"])
        self.assertGreater(current["shoe_posterior_health"], .30)

        before_recovery = current["shoe_posterior_health"]
        for _ in range(8):
            current, memory = self._update(
                _snapshot([.50, .45, .05]), "B", ess=1.0, uncertainty=0.0, state=current, memory=memory,
            )
        self.assertGreater(current["shoe_posterior_health"], before_recovery)
        self.assertLessEqual(len(memory), 10)

    def test_only_health_changes_and_ev_direction_is_never_flipped(self):
        state, _ = self._update(_snapshot([.72, .20, .08], gap=.20), "P", ess=.2, uncertainty=.8)
        self.assertTrue(state["particle_observation_reused"])
        self.assertFalse(state["particle_posterior_reweighted"])
        self.assertGreaterEqual(state["posterior_reliability_multiplier"], .90)
        self.assertGreaterEqual(state["draw_reliability_multiplier"], .90)
        self.assertGreaterEqual(state["physical_ev_reliability_multiplier"], .80)
        self.assertLessEqual(state["physical_ev_reliability_multiplier"], 1.0)
        raw_ev = -.12
        self.assertLess(raw_ev * state["physical_ev_reliability_multiplier"], 0.0)

    def test_tracker_resets_and_replays_chronologically_without_dimension_change(self):
        tracker = ParticleShoeTracker()
        estimate = tracker.estimate("BPPBTBBPPTBBPPBTBP")
        self.assertEqual(estimate.physics_48d.shape, (PHYSICS_DIM,))
        self.assertEqual(estimate.diagnostics["history_rounds"], 18.0)
        self.assertTrue(estimate.diagnostics["pre_hand_snapshot"]["available"])
        self.assertLessEqual(estimate.diagnostics["error_memory_size"], 10.0)
        self.assertTrue(estimate.diagnostics["shoe_error_correction"]["particle_observation_reused"])
        tracker.reset()
        reset = tracker.estimate("").diagnostics
        self.assertEqual(reset["shoe_posterior_health"], 1.0)
        self.assertEqual(reset["draw_state_health"], 1.0)
        self.assertEqual(reset["physical_ev_health"], 1.0)

    def test_fusion_keeps_48d_and_only_shrinks_reliability(self):
        mlp = np.zeros(PHYSICS_DIM, dtype=np.float32)
        mlp[:3] = [.58, .34, .08]
        mlp[3:13] = .1
        mlp[13:23] = .1
        mlp[23:26] = [.4586, .4462, .0952]
        mlp[26:39] = 5 / 13
        mlp[39:43] = [0.0, 0.0, 0.0, .5]
        mlp[44:46] = .5
        fused, diagnostics = fuse_particle_physics(
            mlp, "BPPBTBBPPTBBPPBTBP", shoe_error_correction_version=1,
        )
        self.assertEqual(fused.shape, (PHYSICS_DIM,))
        self.assertGreaterEqual(diagnostics["physical_ev_reliability_multiplier"], .80)
        self.assertLessEqual(diagnostics["physical_ev_reliability_multiplier"], 1.0)
        raw = diagnostics["raw_physical_ev_banker"]
        if abs(raw) > 1e-12:
            self.assertEqual(np.sign(fused[39]), np.sign(raw))
        legacy, legacy_diagnostics = fuse_particle_physics(
            mlp, "BPPBTBBPPTBBPPBTBP", shoe_error_correction_version=0,
        )
        self.assertEqual(legacy.shape, (PHYSICS_DIM,))
        self.assertEqual(legacy_diagnostics["shoe_error_correction_version"], 0.0)


if __name__ == "__main__":
    unittest.main()
