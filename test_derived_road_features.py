import unittest

from derived_road_features import (
    DERIVED_FEATURE_NAMES,
    big_road_continuation_state,
    build_derived_road_features,
    derived_markers,
    derived_to_bigroad_continue_prob,
    recent_continue_rate,
    road_probability_state,
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
                features[f"{prefix}_p_bigroad_continue"]
                + features[f"{prefix}_p_bigroad_turn"],
                1.0,
            )
        self.assertAlmostEqual(
            features["big_road_p_continue"] + features["big_road_p_turn"], 1.0
        )
        self.assertAlmostEqual(
            features["derived_p_bigroad_continue"]
            + features["derived_p_bigroad_turn"],
            1.0,
        )

    def test_continue_and_turn_flags_are_complements_when_available(self):
        state = road_probability_state("BBPBBBPPBBPBPBB", 1)
        if state["available"]:
            self.assertEqual(state["continue_now"] + state["turn_now"], 1.0)

    def test_sparse_history_falls_back_to_neutral_or_bigroad_base(self):
        self.assertEqual(survival_continue_prob([]), 0.5)
        self.assertEqual(recent_continue_rate(["B"]), 0.5)
        self.assertEqual(derived_to_bigroad_continue_prob("B", 1), 0.5)
        features = build_derived_road_features("B")
        self.assertEqual(features["big_road_p_continue"], 0.5)
        self.assertEqual(features["derived_p_bigroad_continue"], 0.5)

    def test_big_road_current_state_detects_continuation_and_turn(self):
        state = big_road_continuation_state("BBBP")
        self.assertEqual(state["turn_now"], 1.0)
        state2 = big_road_continuation_state("BBBPP")
        self.assertEqual(state2["continue_now"], 1.0)

    def test_no_raw_color_features_exported(self):
        features = build_derived_road_features("BBPBBBPPBBP")
        self.assertEqual(set(features), set(DERIVED_FEATURE_NAMES))
        for legacy in (
            "big_eye_color",
            "small_road_color",
            "cockroach_color",
            "derived_turn_sync",
        ):
            self.assertNotIn(legacy, features)

    def test_probabilities_are_bounded(self):
        features = build_derived_road_features("BBPBBBPPBBPBPBBPBPBBPP")
        for name, value in features.items():
            if "_p_" in name or name.startswith("big_road_p_") or name.startswith("derived_p_"):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_conditional_probability_is_deterministic(self):
        history = "BBPBBBPPBBPBPBBPBPBBPP"
        first = derived_to_bigroad_continue_prob(history, 1)
        second = derived_to_bigroad_continue_prob(history, 1)
        self.assertAlmostEqual(first, second)


if __name__ == "__main__":
    unittest.main()
