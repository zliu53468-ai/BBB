import unittest

from derived_road_features import build_derived_road_features, derived_markers


class DerivedRoadFeatureTests(unittest.TestCase):
    def test_big_eye_new_third_column_equal_depth_is_red(self):
        self.assertEqual(derived_markers("BPB", 1), [1])

    def test_big_eye_second_row_with_short_reference_is_blue(self):
        self.assertEqual(derived_markers("BPP", 1), [-1])

    def test_big_eye_second_row_with_equal_or_deeper_reference_is_red(self):
        self.assertEqual(derived_markers("BBPP", 1), [1])

    def test_small_road_starts_later_than_big_eye(self):
        features = build_derived_road_features("BPBP")
        self.assertEqual(features["big_eye_color"], 1.0)
        self.assertEqual(features["small_road_color"], 1.0)
        self.assertEqual(features["cockroach_color"], 0.0)

    def test_ties_do_not_change_derived_roads(self):
        self.assertEqual(derived_markers("BTPB", 1), derived_markers("BPB", 1))

    def test_agreement_is_mean_of_three_current_colors(self):
        features = build_derived_road_features("BPBPB")
        expected = (
            features["big_eye_color"]
            + features["small_road_color"]
            + features["cockroach_color"]
        ) / 3.0
        self.assertAlmostEqual(features["derived_road_agreement"], expected)


if __name__ == "__main__":
    unittest.main()
