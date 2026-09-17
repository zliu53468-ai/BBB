import unittest

from derived_road_features import (
    build_derived_road_features,
    derived_markers,
    steps_since_turn,
    turn_now,
    turn_rate,
)


class DerivedRoadTurnFeatureTests(unittest.TestCase):
    def test_standard_marker_generation_is_preserved(self):
        self.assertEqual(derived_markers("BPB", 1), [1])
        self.assertEqual(derived_markers("BPP", 1), [-1])
        self.assertEqual(derived_markers("BBPP", 1), [1])

    def test_ties_do_not_change_derived_roads(self):
        self.assertEqual(derived_markers("BTPB", 1), derived_markers("BPB", 1))

    def test_turn_now_detects_latest_structural_flip(self):
        self.assertEqual(turn_now([-1, -1, 1]), 1.0)
        self.assertEqual(turn_now([-1, 1, 1]), 0.0)

    def test_steps_since_turn_tracks_current_segment(self):
        self.assertEqual(steps_since_turn([-1, -1, 1]), 1)
        self.assertEqual(steps_since_turn([-1, 1, 1]), 2)
        self.assertEqual(steps_since_turn([1, 1, 1]), 3)

    def test_turn_rate_uses_recent_markers(self):
        self.assertAlmostEqual(turn_rate([-1, 1, -1, -1], 6), 2 / 3)

    def test_features_do_not_export_raw_color(self):
        features = build_derived_road_features("BBPBBB")
        self.assertNotIn("big_eye_color", features)
        self.assertNotIn("small_road_color", features)
        self.assertNotIn("cockroach_color", features)
        self.assertIn("big_eye_turn_now", features)
        self.assertIn("derived_turn_sync", features)

    def test_sync_measures_simultaneous_available_turns(self):
        # BBPBBB produces a fresh turn in both available Big Eye and Small Road.
        features = build_derived_road_features("BBPBBB")
        self.assertEqual(features["big_eye_turn_now"], 1.0)
        self.assertEqual(features["small_road_turn_now"], 1.0)
        self.assertEqual(features["derived_turn_sync"], 1.0)


if __name__ == "__main__":
    unittest.main()
