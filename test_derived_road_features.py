import unittest

from derived_road_features import (
    DERIVED_FEATURE_NAMES,
    big_road_continuation_state,
    build_derived_road_features,
    derived_markers,
    recent_continue_rate,
    road_continuation_state,
    survival_continue_prob,
)


class DerivedRoadContinuationFeatureTests(unittest.TestCase):
    def test_standard_marker_generation_is_preserved(self):
        self.assertEqual(derived_markers("BPB", 1), [1])
        self.assertEqual(derived_markers("BPP", 1), [-1])
        self.assertEqual(derived_markers("BBPP", 1), [1])

    def test_ties_do_not_change_derived_roads(self):
        self.assertEqual(derived_markers("BTPB", 1), derived_markers("BPB", 1))

    def test_probability_pairs_sum_to_one(self):
        features = build_derived_road_features("BBPBBBPPBBPBPBB")
        for prefix in ("big_eye", "small_road", "cockroach"):
            self.assertAlmostEqual(
                features[f"{prefix}_p_continue"] + features[f"{prefix}_p_turn"], 1.0
            )
        self.assertAlmostEqual(
            features["big_road_p_continue"] + features["big_road_p_turn"], 1.0
        )
        self.assertAlmostEqual(
            features["derived_p_continue"] + features["derived_p_turn"], 1.0
        )

    def test_continue_and_turn_flags_are_complements_when_available(self):
        state = road_continuation_state("BBPBBBPPBBPBPBB", 1)
        self.assertEqual(state["available"], 1.0)
        self.assertEqual(state["continue_now"] + state["turn_now"], 1.0)

    def test_sparse_history_falls_back_to_neutral(self):
        self.assertEqual(survival_continue_prob([]), 0.5)
        self.assertEqual(recent_continue_rate(["B"]), 0.5)
        features = build_derived_road_features("B")
        self.assertEqual(features["derived_p_continue"], 0.5)
        self.assertEqual(features["derived_p_turn"], 0.5)

    def test_big_road_current_state_detects_continuation_and_turn(self):
        state = big_road_continuation_state("BBBP")
        self.assertEqual(state["turn_now"], 1.0)
        state2 = big_road_continuation_state("BBBPP")
        self.assertEqual(state2["continue_now"], 1.0)

    def test_no_raw_color_or_turn_only_legacy_fields_exported(self):
        features = build_derived_road_features("BBPBBBPPBBP")
        self.assertEqual(set(features), set(DERIVED_FEATURE_NAMES))
        for legacy in (
            "big_eye_color",
            "small_road_color",
            "cockroach_color",
            "big_eye_steps_since_turn",
            "small_road_steps_since_turn",
            "cockroach_steps_since_turn",
            "derived_turn_sync",
        ):
            self.assertNotIn(legacy, features)

    def test_probabilities_are_bounded(self):
        features = build_derived_road_features("BBPBBBPPBBPBPBBPBPBBPP")
        for name, value in features.items():
            if "p_" in name:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
