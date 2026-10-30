"""CPSI / run-length hazard parity and safety regression tests (stdlib only)."""
import json
import math
import subprocess
import unittest

from run_length_hazard import count_survival_hazard, survival_features_16


class TestHazard(unittest.TestCase):
    def test_strict_observed_gate(self):
        for n in range(4):
            self.assertEqual(count_survival_hazard(["B"] * n, 4, 1, "observed")["direction"], "Skip")
        self.assertEqual(count_survival_hazard(["B"] * 4, 3.1, 1, "estimated")["direction"], "Skip")
        aligned = count_survival_hazard(["B"] * 4, 3.1, 1, "observed")
        self.assertEqual(aligned["direction"], "B")
        self.assertEqual(aligned["follow_multiplier"], 1.5)
        broken = count_survival_hazard(["B"] * 5, -3.1, 1, "observed")
        self.assertEqual(broken["direction"], "P")
        self.assertEqual(broken["follow_multiplier"], .4)
        self.assertEqual(broken["break_multiplier"], 2)
        self.assertEqual(count_survival_hazard(["B"] * 4, 0, 1, "observed")["direction"], "Skip")
        self.assertEqual(count_survival_hazard(["B"] * 4, 3.2, .5, "observed")["direction"], "Skip")
        self.assertEqual(count_survival_hazard(["P"] * 5, -3.1, 1, "observed")["direction"], "P")
        self.assertEqual(count_survival_hazard(["P"] * 5, 3.1, 1, "observed")["direction"], "B")

    def test_range_0_stake(self):
        for history in (["B"] * 20, ["P"] * 5, ["B", "P", "B"], ["B"] * 4 + ["T"]):
            for tc in (-8, -3, -1, 0, 1, 3, 8):
                s = count_survival_hazard(history, tc, 1, "observed")
                self.assertTrue(0 <= s["cpsi"] <= 1)
                self.assertTrue(0 <= s["hazard_rate"] <= 1)
                self.assertEqual(s["bet_fraction"], 0)
                self.assertTrue(0 <= s["simulated_bet_fraction"] <= .25)
                values = survival_features_16(s)
                self.assertEqual(len(values), 16)
                self.assertTrue(all(math.isfinite(x) for x in values))

    def test_python_javascript_parity(self):
        samples = [
            {"history": ["B"] * 4, "trueCount": 3.3, "zeroRatio": .9, "countSource": "observed"},
            {"history": ["B"] * 8, "trueCount": -3.1, "zeroRatio": 1, "countSource": "observed"},
            {"history": ["P"] * 7, "trueCount": -3.2, "zeroRatio": 1, "countSource": "observed"},
            {"history": ["B"] * 5, "trueCount": 4, "zeroRatio": .7, "countSource": "estimated"},
            {"history": ["B", "P", "T", "P", "P", "P", "P"], "trueCount": 2.1, "zeroRatio": .8, "countSource": "observed"},
            {"history": [], "trueCount": 0, "zeroRatio": 1, "countSource": "estimated"},
        ]
        js = subprocess.run(
            ["node", "-e", "const M=require('./bayes_markov_runtime.js');"
             "let s='';process.stdin.on('data',d=>s+=d);"
             "process.stdin.on('end',()=>console.log(JSON.stringify(JSON.parse(s).map(x=>M.countSurvivalHazard(x)))))"],
            input=json.dumps(samples), text=True, capture_output=True, check=True,
        )
        expected = json.loads(js.stdout)
        for sample, js_row in zip(samples, expected):
            py_row = count_survival_hazard(
                sample["history"], sample["trueCount"],
                sample["zeroRatio"], sample["countSource"]
            )
            self.assertEqual(set(py_row), set(js_row))
            for key, value in py_row.items():
                if isinstance(value, float) and not isinstance(value, bool):
                    self.assertAlmostEqual(value, js_row[key], places=12, msg=key)
                else:
                    self.assertEqual(value, js_row[key], msg=key)


if __name__ == "__main__":
    unittest.main()
