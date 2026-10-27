"""Offline candidate trainer for exported point-score hand histories.

This tool is intentionally separate from the browser runtime. It reads the
licensed UI's exported ``bgs_observed_hands_*.json`` file, creates
chronological next-hand examples, holds out entire shoes, and writes a
review-only logistic-regression candidate. It never changes deployed weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

OUTCOMES = ("B", "P", "T")
WINDOW_OUTCOMES = (8, 32)
POINT_WINDOW = 32
FEATURE_NAMES = (
    "shoe_progress",
    *(f"outcome_{side}_last{window}" for window in WINDOW_OUTCOMES for side in OUTCOMES),
    "same_transition_last16", "switch_transition_last16", "point_coverage_last32",
    *(f"player_point_{value}_last32" for value in range(10)),
    *(f"banker_point_{value}_last32" for value in range(10)),
    "card_count_4_last32", "card_count_5_last32", "card_count_6_last32",
    "point_diff_signed_mean_last32", "point_diff_abs_mean_last32", "point_diff_std_last32",
    "last_player_point", "last_banker_point", "last_card_count",
)


def _optional_int(value: Any, low: int, high: int, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result != value and str(result) != str(value):
        raise ValueError(f"{field} must be an integer")
    if not low <= result <= high:
        raise ValueError(f"{field} must be in {low}..{high}")
    return result


def load_observations(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read the browser's macro-observation export and validate shoe order."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = raw.get("rows") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must contain a non-empty rows array")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    point_rows = 0
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("each row must be an object")
        shoe = str(item.get("shoe_id") or "").strip()
        outcome = str(item.get("outcome") or "").upper()
        if not shoe or outcome not in OUTCOMES:
            raise ValueError("each row requires shoe_id and B/P/T outcome")
        round_index = _optional_int(item.get("round_index"), 1, 500, "round_index")
        if round_index is None:
            raise ValueError("round_index is required")
        key = (shoe, round_index)
        if key in seen:
            raise ValueError(f"duplicate shoe/round: {shoe}/{round_index}")
        seen.add(key)
        player = _optional_int(item.get("player_score"), 0, 9, "player_score")
        banker = _optional_int(item.get("banker_score"), 0, 9, "banker_score")
        cards = _optional_int(item.get("card_count"), 4, 6, "card_count")
        if (player is None) != (banker is None):
            raise ValueError(f"partial score observation at {shoe}/{round_index}")
        if player is not None:
            point_rows += 1
            expected = "B" if banker > player else "P" if player > banker else "T"
            if outcome != expected:
                raise ValueError(f"score/outcome mismatch at {shoe}/{round_index}")
        grouped.setdefault(shoe, []).append({
            "shoe_id": shoe, "round_index": round_index, "outcome": outcome,
            "player_score": player, "banker_score": banker, "card_count": cards,
        })
    if point_rows == 0:
        raise ValueError("no complete player_score/banker_score observations found")
    for shoe, hands in grouped.items():
        hands.sort(key=lambda row: row["round_index"])
        rounds = [row["round_index"] for row in hands]
        if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
            raise ValueError(f"round_index must be continuous within shoe {shoe}")
    return grouped


def _features(history: list[dict[str, Any]], total_hands: int) -> list[float]:
    values: list[float] = [min(1.0, len(history) / max(1, total_hands))]
    for window in WINDOW_OUTCOMES:
        tail = history[-window:]
        denominator = max(1, len(tail))
        values.extend(sum(row["outcome"] == side for row in tail) / denominator for side in OUTCOMES)
    directional = [row["outcome"] for row in history if row["outcome"] in ("B", "P")][-17:]
    transitions = [directional[i] == directional[i - 1] for i in range(1, len(directional))]
    values.extend((
        sum(transitions) / max(1, len(transitions)),
        sum(not item for item in transitions) / max(1, len(transitions)),
    ))
    tail = history[-POINT_WINDOW:]
    score_rows = [row for row in tail if row["player_score"] is not None]
    values.append(len(score_rows) / max(1, len(tail)))
    denominator = max(1, len(score_rows))
    values.extend(sum(row["player_score"] == point for row in score_rows) / denominator for point in range(10))
    values.extend(sum(row["banker_score"] == point for row in score_rows) / denominator for point in range(10))
    cards = [row["card_count"] for row in tail if row["card_count"] is not None]
    card_denominator = max(1, len(cards))
    values.extend(sum(card == count for card in cards) / card_denominator for count in (4, 5, 6))
    diffs = [row["banker_score"] - row["player_score"] for row in score_rows]
    mean = sum(diffs) / max(1, len(diffs))
    variance = sum((value - mean) ** 2 for value in diffs) / max(1, len(diffs))
    values.extend((mean / 9, sum(abs(value) for value in diffs) / max(1, len(diffs)) / 9,
                   math.sqrt(variance) / 9))
    last_score = score_rows[-1] if score_rows else {}
    values.extend((
        (last_score.get("player_score") or 0) / 9 if score_rows else 0,
        (last_score.get("banker_score") or 0) / 9 if score_rows else 0,
        ((last_score.get("card_count") or 4) - 4) / 2 if last_score.get("card_count") is not None else 0,
    ))
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"feature ABI mismatch: {len(values)} != {len(FEATURE_NAMES)}")
    return values


def build_examples(shoes: dict[str, list[dict[str, Any]]]):
    xs: list[list[float]] = []
    ys: list[str] = []
    groups: list[str] = []
    for shoe, hands in shoes.items():
        for index, hand in enumerate(hands):
            xs.append(_features(hands[:index], total_hands=60))
            ys.append(hand["outcome"])
            groups.append(shoe)
    return xs, ys, groups


def _shoe_holdout(shoe_ids: list[str], fraction: float = 0.2):
    ordered = sorted(shoe_ids, key=lambda shoe: hashlib.sha256(("point-aux-v1:" + shoe).encode()).hexdigest())
    holdout_count = max(1, math.ceil(len(ordered) * fraction))
    if holdout_count >= len(ordered):
        raise ValueError("at least two separate shoes are required")
    return set(ordered[:-holdout_count]), set(ordered[-holdout_count:])


def train_candidate(shoes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Train and evaluate a review-only auxiliary candidate, grouped by shoe."""
    if len(shoes) < 3:
        raise ValueError("at least three shoes are required for a whole-shoe holdout")
    train_shoes, test_shoes = _shoe_holdout(list(shoes))
    xs, ys, groups = build_examples(shoes)
    train_ix = [i for i, shoe in enumerate(groups) if shoe in train_shoes]
    test_ix = [i for i, shoe in enumerate(groups) if shoe in test_shoes]
    if len(train_ix) < 30 or not test_ix:
        raise ValueError("not enough chronological examples for training and holdout")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, log_loss
    except ImportError as exc:
        raise RuntimeError("install research/requirements.txt to train the offline candidate") from exc
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=20261027))
    x_train = [xs[i] for i in train_ix]
    y_train = [ys[i] for i in train_ix]
    x_test = [xs[i] for i in test_ix]
    y_test = [ys[i] for i in test_ix]
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    classes = [str(value) for value in model[-1].classes_]
    aligned = [[float(row[classes.index(side)]) if side in classes else 0.0 for side in OUTCOMES]
               for row in probabilities]
    actual_indices = [OUTCOMES.index(value) for value in y_test]
    clipped_loss = -sum(math.log(max(1e-12, row[actual])) for row, actual in zip(aligned, actual_indices)) / len(actual_indices)
    predictions = [OUTCOMES[max(range(3), key=lambda i: row[i])] for row in aligned]
    ev_actions = []
    for i, row in enumerate(aligned):
        ev_b = 0.95 * row[0] - row[1]
        ev_p = row[1] - row[0]
        selected = "B" if ev_b > ev_p and ev_b > 0 else "P" if ev_p > ev_b and ev_p > 0 else "Skip"
        if selected != "Skip":
            ev_actions.append((selected, y_test[i]))
    model_data = model[-1]
    scaler = model[0]
    return {
        "schema_version": 1,
        "model_type": "offline_point_history_logistic_candidate",
        "deployment_status": "review_only_not_promoted",
        "input_contract": "prior completed B/P/T outcomes, final player/banker points when available, optional card count",
        "feature_names": list(FEATURE_NAMES),
        "classes": classes,
        "standardizer_mean": [float(v) for v in scaler.mean_],
        "standardizer_scale": [float(v) for v in scaler.scale_],
        "coefficients": [[float(v) for v in row] for row in model_data.coef_],
        "intercepts": [float(v) for v in model_data.intercept_],
        "split": {"unit": "whole_shoe", "rule": "stable SHA256 80/20 group holdout",
                  "train_shoes": sorted(train_shoes), "test_shoes": sorted(test_shoes)},
        "metrics": {"train_rows": len(train_ix), "test_rows": len(test_ix),
                    "test_accuracy": float(accuracy_score(y_test, predictions)),
                    "test_log_loss": float(clipped_loss),
                    "positive_ev_action_count": len(ev_actions),
                    "positive_ev_action_hit_rate": (sum(side == outcome for side, outcome in ev_actions) / len(ev_actions)
                                                     if ev_actions else None)},
        "guardrails": ["candidate only; no automatic promotion", "holdout is grouped by shoe",
                       "positive EV estimates are not proof of profitable or improved real-world betting"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="browser export JSON from 匯出開牌資料")
    parser.add_argument("--output", required=True, help="candidate JSON path; use a new research filename")
    args = parser.parse_args(argv)
    result = train_candidate(load_observations(args.input))
    target = Path(args.output)
    if target.name in {"physics_multitask_model.json", "final_probability_model.json"}:
        raise SystemExit("candidate output must not replace a production model")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "metrics": result["metrics"],
                      "deployment_status": result["deployment_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
