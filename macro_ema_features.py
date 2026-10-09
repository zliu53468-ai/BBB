#!/usr/bin/env python3
"""Causal macro EMA features, Python standard library only.

python macro_ema_features.py completed_hands.json features.json
Input: ordered list (or {"rows": [...]}) of shoe_id, round_index, outcome,
card_count (4..6), player_score (0..9), banker_score (0..9).
Missing observations stay missing; result alone never supplies points/cards.
Output features are BEFORE the row's outcome, suitable for next-hand training.
"""
import argparse
import json
import math

NAMES = ("Six_Card", "Low_Score", "Point_Diff")


def integer(value, name, low, high):
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not str(value).strip().isascii() or not str(value).strip().isdigit():
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def normalize_hand(hand):
    outcome = str(hand.get("outcome", "")).upper()
    if outcome not in ("B", "P", "T"):
        raise ValueError("outcome must be B/P/T")
    cards = integer(hand.get("card_count"), "card_count", 4, 6)
    player = integer(hand.get("player_score"), "player_score", 0, 9)
    banker = integer(hand.get("banker_score"), "banker_score", 0, 9)
    if player is not None and banker is not None:
        actual = "B" if banker > player else "P" if player > banker else "T"
        if actual != outcome:
            raise ValueError("outcome disagrees with final scores")
    return dict(outcome=outcome, card_count=cards, player_score=player, banker_score=banker)


def raw_features(hand):
    h = normalize_hand(hand)
    winner = h["banker_score"] if h["outcome"] == "B" else h["player_score"]
    return {
        "Six_Card": None if h["card_count"] is None else int(h["card_count"] == 6),
        "Low_Score": 0 if h["outcome"] == "T" else None if winner is None else int(winner <= 3),
        "Point_Diff": None if h["player_score"] is None or h["banker_score"] is None else abs(h["player_score"] - h["banker_score"]),
    }


class MacroEMA:
    def __init__(self, shoe_id=""):
        self.shoe_id, self.rounds = str(shoe_id), 0
        self.ema = {name: dict(short=None, long=None, count=0, last_round=0) for name in NAMES}

    def snapshot(self):
        out = dict(shoe_id=self.shoe_id, rounds=self.rounds, counts={}, ages={})
        for key, e in self.ema.items():
            out[key + "_Short"], out[key + "_Long"] = e["short"], e["long"]
            out[key + "_Spread"] = e["short"] - e["long"] if e["count"] else None
            out["counts"][key] = e["count"]
            out["ages"][key] = self.rounds - e["last_round"] if e["count"] else None
        return out

    def update(self, hand):
        raw = raw_features(hand)  # Atomic validation, matches JS.
        self.rounds += 1
        for key, value in raw.items():
            if value is None:
                continue
            e = self.ema[key]
            e["short"] = .20 * value + .80 * e["short"] if e["count"] else value
            e["long"] = .05 * value + .95 * e["long"] if e["count"] else value
            e["count"] += 1
            e["last_round"] = self.rounds
        return self.snapshot()


def build_features(rows, estimated_total_hands=60):
    """Rows must contain every completed hand, including unpredicted hands/ties.

    Strict sequential indices prevent silently treating sparse prediction logs
    as a complete observation stream. Interleaved shoes maintain separate state.
    """
    total = float(estimated_total_hands)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("estimated_total_hands must be positive and finite")
    states, output = {}, []
    for row in rows:
        if row.get("shoe_id") is None:
            raise ValueError("shoe_id required")
        shoe = str(row["shoe_id"])
        state = states.setdefault(shoe, MacroEMA(shoe))
        index = integer(row.get("round_index"), "round_index", 1, 1000000)
        if index != state.rounds + 1:
            raise ValueError(f"{shoe}: expected consecutive round_index {state.rounds + 1}")
        observed = normalize_hand(row)
        progress = min(1., index / total)
        output.append({**row, **state.snapshot(), "progress": progress,
                       "sample_weight": .8 if progress < .3 else 1.25 if progress > .7 else 1.,
                       "actual_b": 1 if observed["outcome"] == "B" else 0 if observed["outcome"] == "P" else None})
        state.update(observed)  # Only after pre-hand features have been captured.
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--estimated-total-hands", type=float, default=60)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    result = {"feature_timing": "before_current_outcome", "rows": build_features(rows, args.estimated_total_hands)}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
