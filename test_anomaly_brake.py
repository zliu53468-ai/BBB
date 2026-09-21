#!/usr/bin/env python3
import unittest

import numpy as np

from shoe_particle_filter import new_shoe_particle_filter
from xgb_particle_filter_residual import (
    MODEL_FEATURE_NAMES,
    anomaly_brake_factor,
    apply_anomaly_brake,
)


class AnomalyBrakeTests(unittest.TestCase):
    def test_long_run_break_spikes_anomaly(self) -> None:
        pf = new_shoe_particle_filter()
        pf.recent_outcomes = [1, 1, 1]
        pf.anomaly_score = 0.0

        pf.observe_and_project(
            real_outcome=0,
            core_pb=0.70,
            current_round=4,
        )

        self.assertGreaterEqual(
            pf.current_anomaly_score(),
            0.95,
        )

    def test_stable_extension_stays_low(self) -> None:
        pf = new_shoe_particle_filter()
        pf.recent_outcomes = [1, 1, 1]
        pf.anomaly_score = 0.0

        pf.observe_and_project(
            real_outcome=1,
            core_pb=0.70,
            current_round=4,
        )

        self.assertLess(
            pf.current_anomaly_score(),
            0.35,
        )

    def test_full_brake_zeroes_delta(self) -> None:
        self.assertEqual(
            anomaly_brake_factor(0.95),
            0.0,
        )
        braked = apply_anomaly_brake(
            np.asarray([0.10, -0.10]),
            np.asarray([0.95, 1.0]),
        )
        self.assertTrue(
            np.allclose(braked, 0.0),
        )

    def test_low_anomaly_keeps_delta(self) -> None:
        self.assertEqual(
            anomaly_brake_factor(0.20),
            1.0,
        )

    def test_model_dimension_is_11(self) -> None:
        self.assertEqual(
            len(MODEL_FEATURE_NAMES),
            11,
        )
        self.assertEqual(
            MODEL_FEATURE_NAMES[-1],
            "anomaly_score",
        )


if __name__ == "__main__":
    unittest.main()
