import unittest

import xgb_residual_bias as base
import xgb_residual_bias_17d as ext


class Residual17DSchemaTests(unittest.TestCase):
    def test_feature_schema_has_17_fields(self):
        self.assertEqual(len(ext.FEATURE_NAMES), 17)
        self.assertEqual(ext.FEATURE_NAMES[:7], ext._BASE_FEATURE_NAMES)

    def test_build_features_contains_turn_state_not_raw_colors(self):
        row = ext.build_features(core_p_b=0.52, history="BBPBBB").as_dict()
        self.assertEqual(set(row), set(ext.FEATURE_NAMES))
        self.assertIn("big_eye_turn_now", row)
        self.assertIn("small_road_turn_now", row)
        self.assertIn("cockroach_turn_now", row)
        self.assertIn("derived_turn_sync", row)
        self.assertNotIn("big_eye_color", row)
        self.assertNotIn("small_road_color", row)
        self.assertNotIn("cockroach_color", row)

    def test_old_v1_row_can_be_upgraded_from_history_fingerprint(self):
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
        self.assertEqual(len(upgraded), 17)
        self.assertEqual(set(upgraded), set(ext.FEATURE_NAMES))

    def test_install_patches_base_export_schema_without_breaking_build(self):
        ext.install_17d_schema()
        self.assertEqual(len(base.FEATURE_NAMES), 17)
        built = base.build_features(core_p_b=0.51, history="BPBPPB").as_dict()
        self.assertEqual(len(built), 17)


if __name__ == "__main__":
    unittest.main()
