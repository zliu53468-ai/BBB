import unittest

import xgb_residual_bias as base
import xgb_residual_bias_17d as ext


class ResidualRoadProbabilitySchemaTests(unittest.TestCase):
    def test_feature_schema_has_23_fields(self):
        self.assertEqual(len(ext.FEATURE_NAMES), 23)
        self.assertEqual(ext.FEATURE_NAMES[:7], ext._BASE_FEATURE_NAMES)

    def test_build_features_contains_continue_turn_probabilities(self):
        row = ext.build_features(core_p_b=0.52, history="BBPBBBPPBBPBPBB").as_dict()
        self.assertEqual(set(row), set(ext.FEATURE_NAMES))
        for name in (
            "big_eye_p_continue",
            "big_eye_p_turn",
            "small_road_p_continue",
            "small_road_p_turn",
            "cockroach_p_continue",
            "cockroach_p_turn",
            "big_road_p_continue",
            "big_road_p_turn",
            "derived_p_continue",
            "derived_p_turn",
        ):
            self.assertIn(name, row)
        self.assertNotIn("big_eye_color", row)
        self.assertNotIn("derived_turn_sync", row)

    def test_old_row_can_be_upgraded_from_history_fingerprint(self):
        old = {
            "core_p_b": 0.49,
            "round_index": 8,
            "estimated_total_hands": 60,
            "remaining_ratio": 0.88,
            "sx_markov_p_same": 0.5,
            "stage": 1,
            "depth": 1,
            "history_fingerprint": "BBPPBPB",
        }
        upgraded = ext.feature_row(old)
        self.assertEqual(len(upgraded), 23)
        self.assertEqual(set(upgraded), set(ext.FEATURE_NAMES))

    def test_install_patches_base_export_schema_without_breaking_build(self):
        ext.install_road_probability_schema()
        self.assertEqual(len(base.FEATURE_NAMES), 23)
        built = base.build_features(core_p_b=0.51, history="BPBPPB").as_dict()
        self.assertEqual(len(built), 23)


if __name__ == "__main__":
    unittest.main()
