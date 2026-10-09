import json
from pathlib import Path
import subprocess
import unittest

from macro_ema_features import MacroEMA, build_features

ROOT = Path(__file__).resolve().parent


class MacroEMATest(unittest.TestCase):
    def test_initialization_missing_tie_and_exact_ema(self):
        state = MacroEMA("a")
        self.assertIsNone(state.snapshot()["Six_Card_Spread"])
        state.update(dict(outcome="B", card_count=6, banker_score=3, player_score=1))
        value = state.update(dict(outcome="P", card_count=4, banker_score=0, player_score=8))
        self.assertAlmostEqual(value["Six_Card_Spread"], -.15)
        self.assertAlmostEqual(value["Low_Score_Spread"], -.15)
        self.assertAlmostEqual(value["Point_Diff_Spread"], .9)
        missing = state.update(dict(outcome="B"))
        self.assertEqual(missing["counts"], value["counts"])
        self.assertEqual(missing["ages"]["Six_Card"], 1)
        tie = state.update(dict(outcome="T"))
        self.assertEqual(tie["counts"]["Low_Score"], 3)
        self.assertEqual(tie["counts"]["Point_Diff"], 2)

    def test_invalid_is_atomic(self):
        state = MacroEMA()
        before = state.snapshot()
        for bad in [dict(outcome="B", player_score=9, banker_score=0), dict(outcome="T", card_count=7), dict(outcome="P", player_score=True)]:
            with self.assertRaises(ValueError):
                state.update(bad)
            self.assertEqual(state.snapshot(), before)

    def test_causal_and_separate_shoes(self):
        rows = [dict(shoe_id="a", round_index=1, outcome="B", card_count=6),
                dict(shoe_id="b", round_index=1, outcome="P", card_count=4),
                dict(shoe_id="a", round_index=2, outcome="T", card_count=5)]
        result = build_features(rows)
        self.assertIsNone(result[0]["Six_Card_Short"])
        self.assertIsNone(result[1]["Six_Card_Short"])
        self.assertEqual(result[2]["Six_Card_Short"], 1)
        changed = [*rows[:-1], {**rows[-1], "outcome": "B", "card_count": 4}]
        self.assertEqual(build_features(changed)[-1]["Six_Card_Spread"], result[-1]["Six_Card_Spread"])
        for bad in [rows + [rows[-1]], [dict(shoe_id="x", round_index=3, outcome="B")]]:
            with self.assertRaises(ValueError):
                build_features(bad)

    def test_browser_parity_all_snapshots(self):
        rows = [dict(outcome="B", banker_score=3, player_score=1, card_count=6),
                dict(outcome="P", banker_score=2, player_score=9, card_count=4),
                dict(outcome="T", banker_score=0, player_score=0, card_count=6),
                dict(outcome="B"), dict(outcome="P", player_score=0), dict(outcome="T")]*15
        state = MacroEMA("parity")
        expected = [state.update(row) for row in rows]
        source = "const fs=require('fs'),m=require('./macro_ema.js'),s=m.createState('parity');console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(h=>m.update(s,h))));"
        actual = json.loads(subprocess.check_output(["node", "-e", source], input=json.dumps(rows).encode(), cwd=ROOT))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
