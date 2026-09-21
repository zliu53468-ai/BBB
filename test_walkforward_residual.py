#!/usr/bin/env python3
import unittest

import numpy as np

from xgb_residual_bias import (
    FEATURE_NAMES,
    PreparedRow,
    adaptive_delta_limit,
    build_walk_forward_folds,
    non_tie_arrays,
    prepare_rows,
)


class WalkForwardResidualTests(unittest.TestCase):
    def _row(self, shoe, round_index, outcome, created_at):
        return {
            "shoe_id": shoe,
            "created_at": created_at,
            "history_fingerprint": f"{shoe}-{round_index}",
            "actual_outcome": outcome,
            "core_p_b": 0.52,
            "core_raw_p_b": 0.53,
            "core_logit": float(np.log(0.53 / 0.47)),
            "round_index": round_index,
            "estimated_total_hands": 60,
            "remaining_ratio": max(0, (60 - (round_index - 1)) / 60),
            "sx_markov_p_same": 0.5,
            "stage": 1,
            "depth": 1,
        }

    def test_fixed_7d_schema_is_preserved(self):
        self.assertEqual(
            FEATURE_NAMES,
            (
                "core_p_b",
                "round_index",
                "estimated_total_hands",
                "remaining_ratio",
                "sx_markov_p_same",
                "stage",
                "depth",
            ),
        )

    def test_walk_forward_is_chronological_and_disjoint(self):
        shoes = [f"s{i:02d}" for i in range(14)]
        folds = build_walk_forward_folds(
            shoes,
            min_train_shoes=8,
            calibration_shoes=2,
            test_shoes=2,
            step_shoes=2,
        )
        self.assertGreaterEqual(len(folds), 2)

        prior_test = set()
        for fold in folds:
            train = set(fold.train_shoes)
            cal = set(fold.calibration_shoes)
            test = set(fold.test_shoes)

            self.assertFalse(train & cal)
            self.assertFalse(train & test)
            self.assertFalse(cal & test)
            self.assertFalse(prior_test & test)
            prior_test |= test

            all_positions = {
                shoe: shoes.index(shoe)
                for shoe in train | cal | test
            }
            self.assertLess(
                max(all_positions[s] for s in train),
                min(all_positions[s] for s in cal),
            )
            self.assertLess(
                max(all_positions[s] for s in cal),
                min(all_positions[s] for s in test),
            )

    def test_ties_are_retained_but_excluded_from_primary_target(self):
        records = [
            self._row("shoe_a", 1, "B", 1000),
            self._row("shoe_a", 2, "T", 1001),
            self._row("shoe_a", 3, "P", 1002),
        ]
        rows = prepare_rows(records)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row.outcome for row in rows], ["B", "T", "P"])

        x, y, _, _ = non_tie_arrays(rows)
        self.assertEqual(x.shape, (2, 7))
        self.assertEqual(y.tolist(), [1.0, 0.0])

    def test_missing_shoe_id_is_rejected(self):
        row = self._row("shoe_a", 1, "B", 1000)
        row.pop("shoe_id")
        with self.assertRaises(ValueError):
            prepare_rows([row])

    def test_duplicate_prediction_is_rejected(self):
        row = self._row("shoe_a", 1, "B", 1000)
        duplicate = dict(row)
        with self.assertRaises(ValueError):
            prepare_rows([row, duplicate])

    def test_round_index_must_increase_inside_shoe(self):
        a = self._row("shoe_a", 1, "B", 1000)
        b = self._row("shoe_a", 1, "P", 1001)
        b["history_fingerprint"] = "different"
        with self.assertRaises(ValueError):
            prepare_rows([a, b])

    def test_adaptive_clip_is_stricter_near_half(self):
        limits = adaptive_delta_limit(
            np.asarray([0.50, 0.52, 0.58]),
            min_delta=0.025,
            max_delta=0.10,
            confidence_span=0.08,
        )
        self.assertAlmostEqual(float(limits[0]), 0.025, places=8)
        self.assertGreater(float(limits[1]), float(limits[0]))
        self.assertAlmostEqual(float(limits[2]), 0.10, places=8)


if __name__ == "__main__":
    unittest.main()
