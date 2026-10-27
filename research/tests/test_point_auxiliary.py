import json
import tempfile
import unittest
from pathlib import Path

from research.train_point_auxiliary import FEATURE_NAMES, build_examples, load_observations, train_candidate


def sample_rows(shoes=4, rounds=12):
    rows = []
    for shoe_index in range(shoes):
        shoe = f"shoe-{shoe_index:02d}"
        for round_index in range(1, rounds + 1):
            player = (round_index * 2 + shoe_index) % 10
            banker = (round_index * 3 + shoe_index + 1) % 10
            outcome = "B" if banker > player else "P" if player > banker else "T"
            rows.append({"shoe_id": shoe, "round_index": round_index, "outcome": outcome,
                         "player_score": player, "banker_score": banker,
                         "card_count": 4 + (round_index % 3)})
    return rows


class PointAuxiliaryTests(unittest.TestCase):
    def test_load_and_build_chronological_features(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "observations.json"
            path.write_text(json.dumps({"rows": sample_rows()}), encoding="utf-8")
            shoes = load_observations(path)
        xs, ys, groups = build_examples(shoes)
        self.assertEqual(len(xs), 48)
        self.assertEqual(len(xs[0]), len(FEATURE_NAMES))
        self.assertEqual(ys[0], shoes[groups[0]][0]["outcome"])
        self.assertEqual(xs[0][0], 0.0, "first prediction uses no current-hand score")
        self.assertEqual(xs[1][0], 1 / 60)

    def test_target_hand_scores_do_not_leak_into_its_features(self):
        rows = sample_rows(shoes=3, rounds=8)
        original = {shoe: [dict(row) for row in hands]
                    for shoe, hands in load_grouped(rows).items()}
        changed_rows = [dict(row) for row in rows]
        target = next(row for row in changed_rows if row["shoe_id"] == "shoe-00" and row["round_index"] == 5)
        # Keep the target's B outcome while changing the revealed score pair.
        target["player_score"], target["banker_score"] = 1, 9
        changed = load_grouped(changed_rows)
        x_before, _, groups_before = build_examples(original)
        x_after, _, groups_after = build_examples(changed)
        target_index = groups_before.index("shoe-00") * 8 + 4
        self.assertEqual(groups_before, groups_after)
        self.assertEqual(x_before[target_index], x_after[target_index])

    def test_rejects_inconsistent_score_labels_and_partial_scores(self):
        for mutate, expected in [
            (lambda row: row.update(player_score=9, banker_score=1), "score/outcome mismatch"),
            (lambda row: row.update(player_score=None), "partial score"),
        ]:
            rows = sample_rows(shoes=1, rounds=3)
            mutate(rows[0])
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.json"
                path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    load_observations(path)

    def test_candidate_is_whole_shoe_holdout_and_not_promoted(self):
        try:
            candidate = train_candidate(load_grouped(sample_rows(shoes=5, rounds=14)))
        except RuntimeError as exc:
            if "install research/requirements.txt" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertEqual(candidate["deployment_status"], "review_only_not_promoted")
        self.assertEqual(candidate["split"]["rule"], "stable SHA256 80/20 group holdout")
        self.assertTrue(set(candidate["split"]["train_shoes"]).isdisjoint(candidate["split"]["test_shoes"]))
        self.assertGreater(candidate["metrics"]["test_rows"], 0)
        self.assertEqual(len(candidate["coefficients"][0]), len(FEATURE_NAMES))


def load_grouped(rows):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "observations.json"
        path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return load_observations(path)


if __name__ == "__main__":
    unittest.main()
