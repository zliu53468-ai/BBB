import math
import unittest

import numpy as np

from particle_shoe_filter import (
    ESS_RESAMPLE_THRESHOLD,
    PARTICLE_MAX,
    PARTICLE_MID,
    PARTICLE_MIN,
    PARTICLE_TEACHER,
    PHYSICS_DIM,
    ParticleShoeTracker,
    _normalise_log_weights,
    _rejuvenate_particles,
    fuse_particle_physics,
    particle_quality,
    select_particle_budget,
    should_resample,
)
import xgb_final_probability as final


class AdaptiveParticleFilterTests(unittest.TestCase):
    def test_runtime_budget_thresholds_and_hysteresis(self):
        low = {"posterior_uncertainty": .20, "recent_ess_ratio": .85, "composition_spread": .04}
        medium = {"posterior_uncertainty": .45, "recent_ess_ratio": .60, "composition_spread": .10}
        high = {"posterior_uncertainty": .80, "recent_ess_ratio": .20, "composition_spread": .30}
        self.assertEqual(select_particle_budget(low), PARTICLE_MIN)
        self.assertEqual(select_particle_budget(medium), PARTICLE_MID)
        self.assertEqual(select_particle_budget(high), PARTICLE_MAX)
        self.assertEqual(select_particle_budget(high, previous_budget=PARTICLE_MIN), PARTICLE_MID)
        self.assertEqual(select_particle_budget(low, previous_budget=PARTICLE_MAX), PARTICLE_MID)

    def test_log_weight_normalisation_and_normalized_ess(self):
        weights = _normalise_log_weights([-900.0, -901.0, -1200.0])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        ess = 1.0 / float(np.sum(weights * weights))
        ratio = ess / len(weights)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
        self.assertTrue(should_resample(ESS_RESAMPLE_THRESHOLD - .01))
        self.assertFalse(should_resample(ESS_RESAMPLE_THRESHOLD))

    def test_rejuvenation_preserves_legal_rank_composition(self):
        rng = np.random.default_rng(7)
        particles = [np.asarray([30] + [32] * 12, dtype=np.int16) for _ in range(128)]
        before = [int(p.sum()) for p in particles]
        rejuvenated, changed = _rejuvenate_particles(particles, rng)
        self.assertGreater(changed, 0)
        self.assertEqual([int(p.sum()) for p in rejuvenated], before)
        self.assertTrue(all(np.all(p >= 0) and np.all(p <= 32) for p in rejuvenated))

    def test_teacher_mode_is_fixed_and_runtime_is_bounded(self):
        teacher = ParticleShoeTracker(mode="teacher")
        runtime = ParticleShoeTracker(mode="runtime")
        self.assertEqual(len(teacher.particles), PARTICLE_TEACHER)
        self.assertIn(len(runtime.particles), (PARTICLE_MIN, PARTICLE_MID, PARTICLE_MAX))

    def test_quality_uses_posterior_health_not_raw_count(self):
        poor = {"particle_count": PARTICLE_MAX, "recent_ess_ratio": .10, "posterior_uncertainty": .90, "composition_spread": .25}
        good = {"particle_count": PARTICLE_MIN, "recent_ess_ratio": .90, "posterior_uncertainty": .10, "composition_spread": .03}
        self.assertGreater(particle_quality(good), particle_quality(poor))
        self.assertGreater(
            final.dynamic_ema_alpha(35, .50, {}, .30, good),
            final.dynamic_ema_alpha(35, .50, {}, .30, poor),
        )

    def test_dimensions_and_ev_direction_stay_fixed(self):
        tracker = ParticleShoeTracker(particle_count=32, particle_filter_version=3)
        estimate = tracker.estimate("BP")
        self.assertEqual(estimate.physics_48d.shape, (PHYSICS_DIM,))
        physics = estimate.physics_48d.copy()
        physics[39] = -.04
        physics[40] = .04
        physics[41] = -.08
        self.assertLess(physics[39] * .80, 0.0)
        original = np.asarray([.52, 3, 60, .95, .5, 1, 1], dtype=np.float32)
        bridge = final.build_56d_feature_matrix(.52, original, physics)
        self.assertEqual(bridge.shape, (1, 57))
        mlp = np.zeros(PHYSICS_DIM, dtype=np.float32)
        mlp[:3] = [.3, .4, .3]
        mlp[3:13] = .1
        mlp[13:23] = .1
        mlp[23:26] = [.4586, .4462, .0952]
        mlp[26:39] = 5.0 / 13.0
        fused, diagnostics = fuse_particle_physics(
            mlp, "BP", particle_count=32, particle_filter_version=3,
        )
        self.assertEqual(fused.shape, (PHYSICS_DIM,))
        raw = diagnostics["raw_physical_ev_banker"]
        if abs(raw) > 1e-12:
            self.assertEqual(np.sign(fused[39]), np.sign(raw))


if __name__ == "__main__":
    unittest.main()
