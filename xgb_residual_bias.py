#!/usr/bin/env python3
"""BBB XGBoost residual-bias trainer V1.2 (streak-quality policy).

Architecture stays intentionally narrow:

    Frozen 256D/V23 core
      -> existing 7 residual features
      -> XGBRegressor residual
      -> validation-selected delta scale
      -> bounded delta (+/-10%)
      -> validation-selected switch margin
      -> decisive B/P output (no PASS)

The XGB target remains:
    residual = actual_B - core_p_B

V1.2 adds sequence-aware validation. A switch margin creates hysteresis around
50% so the output does not chatter B/P on tiny 49.x/50.x movements. The margin
is not hard-coded: grouped out-of-shoe OOF predictions select it jointly with
delta_scale. Selection is constrained so accuracy/Brier/net-flip do not regress,
then prefers stronger 50-hand table quality and win-streak quality without
worsening loss-streak risk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from xgboost import XGBRegressor

FEATURE_NAMES: tuple[str, ...] = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "sx_markov_p_same",
    "stage",
    "depth",
)
MODEL_TYPE = "xgb_residual_regressor"
MODEL_VERSION = "XGB_RESIDUAL_BIAS_V1_2_STREAK"
BASE_CORE_VERSION = "V23_SHORT_X_DYNAMIC_HAZARD_R1"
SCHEMA_VERSION = 3

DEFAULT_MAX_DELTA = 0.10
DEFAULT_RANDOM_STATE = 20260916
DEFAULT_FOLDS = 5
DEFAULT_TABLE_SIZE = 50
DEFAULT_DELTA_SCALES: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
DEFAULT_SWITCH_MARGINS: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020)


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        values: Iterable[Any] = [x for x in history.upper() if x in {"B", "P", "T"}]
    else:
        values = history
    out: list[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out


def transition_sequence(history: Sequence[str]) -> list[str]:
    values = [x for x in history if x in {"B", "P"}]
    return ["S" if values[i] == values[i - 1] else "X" for i in range(1, len(values))]


def sx_markov_p_same(history: Sequence[str], *, window: int = 24, prior: float = 1.0) -> float:
    tokens = transition_sequence(history)
    if not tokens:
        return 0.5
    current = tokens[-1]
    start = max(0, len(tokens) - 1 - max(2, int(window)))
    same = switch = 0.0
    for i in range(start, len(tokens) - 1):
        if tokens[i] != current:
            continue
        if tokens[i + 1] == "S":
            same += 1.0
        elif tokens[i + 1] == "X":
            switch += 1.0
    return clip((same + prior) / (same + switch + 2.0 * prior))


def current_stage(history: Sequence[str]) -> int:
    values = [x for x in history if x in {"B", "P"}]
    if not values:
        return 0
    side = values[-1]
    n = 1
    for i in range(len(values) - 2, -1, -1):
        if values[i] != side:
            break
        n += 1
    return n


def current_depth(history: Sequence[str]) -> int:
    tokens = transition_sequence(history)
    if not tokens:
        return 0
    token = tokens[-1]
    n = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != token:
            break
        n += 1
    return n


@dataclass(frozen=True)
class ResidualFeatures:
    core_p_b: float
    round_index: float
    estimated_total_hands: float
    remaining_ratio: float
    sx_markov_p_same: float
    stage: float
    depth: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float32)


def build_features(*, core_p_b: float, history: str | Sequence[str], estimated_total_hands: float = 60.0,
                   stage: float | None = None, depth: float | None = None) -> ResidualFeatures:
    seq = normalize_history(history)
    total_hands = clip(float(estimated_total_hands), 40.0, 90.0)
    round_index = float(max(1, min(70, len(seq) + 1)))
    remaining_ratio = clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b)),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        sx_markov_p_same=sx_markov_p_same(seq),
        stage=float(current_stage(seq) if stage is None else stage),
        depth=float(current_depth(seq) if depth is None else depth),
    )


def build_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=240,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=8.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.20,
        reg_lambda=8.0,
        gamma=0.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


class ResidualBiasPredictor:
    def __init__(self, model: XGBRegressor | None = None, *, max_delta: float = DEFAULT_MAX_DELTA,
                 delta_scale: float = 1.0, switch_margin: float = 0.0) -> None:
        self.model = model or build_regressor()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)
        self.delta_scale = clip(float(delta_scale), 0.0, 1.0)
        self.switch_margin = clip(float(switch_margin), 0.0, 0.05)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "ResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        self.model.fit(x, y - core_pb)
        return self

    def predict_delta(self, feature_row: Sequence[float]) -> float:
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        raw = float(self.model.predict(x)[0])
        return clip(raw * self.delta_scale, -self.max_delta, self.max_delta)

    def correct(self, feature_row: Sequence[float], previous_direction: str = "") -> dict[str, Any]:
        x = np.asarray(feature_row, dtype=np.float32)
        core_pb = clip(float(x[FEATURE_NAMES.index("core_p_b")]))
        delta = self.predict_delta(x)
        final_pb = clip(core_pb + delta)
        raw_direction = "B" if final_pb > 0.5 else "P"
        direction = apply_switch_margin_one(final_pb, previous_direction, self.switch_margin)
        return {"core_p_b": core_pb, "delta_scale": self.delta_scale, "switch_margin": self.switch_margin,
                "delta": delta, "final_p_b": final_pb, "raw_direction": raw_direction,
                "direction": direction, "held_by_margin": direction != raw_direction}


def _parse_actual_b(record: Mapping[str, Any]) -> int:
    if "actual_b" in record and record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").upper().strip()
    if actual == "B": return 1
    if actual == "P": return 0
    raise ValueError("training row must contain actual_outcome B/P or actual_b 0/1")


def _feature_row(record: Mapping[str, Any]) -> dict[str, float]:
    if all(name in record for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}
    return build_features(
        core_p_b=float(record.get("core_p_b", record.get("core_pb", 0.5))),
        history=record.get("history") or record.get("history_fingerprint") or "",
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    ).as_dict()


def load_training_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list): raise ValueError("JSON must be a list or {'rows': [...]} bundle")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("training input must be .json or .csv")


def make_training_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, int]]:
    rows: list[tuple[str, float, int, list[float], float, int]] = []
    seen: set[tuple[str, str]] = set()
    stats = {"input_rows": len(records), "duplicates_removed": 0, "invalid_rows": 0, "wrong_core_rows": 0}
    for idx, record in enumerate(records):
        shoe_id = str(record.get("shoe_id") or f"row_{idx}")
        fingerprint = str(record.get("history_fingerprint") or record.get("history") or f"row_{idx}")
        key = (shoe_id, fingerprint)
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        if str(record.get("core_version") or BASE_CORE_VERSION) != BASE_CORE_VERSION:
            stats["wrong_core_rows"] += 1
            continue
        try:
            actual_b = _parse_actual_b(record)
            fm = _feature_row(record)
            vector = [float(fm[name]) for name in FEATURE_NAMES]
            if not all(math.isfinite(v) for v in vector): raise ValueError("non-finite feature")
            core_p_b = clip(fm["core_p_b"])
            round_index = float(fm["round_index"])
            created_at = int(float(record.get("created_at") or idx))
        except (TypeError, ValueError, KeyError):
            stats["invalid_rows"] += 1
            continue
        rows.append((shoe_id, round_index, created_at, vector, float(actual_b) - core_p_b, actual_b))
    if not rows: raise ValueError("no valid B/P training rows")
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    stats["valid_rows"] = len(rows)
    return (np.asarray([r[3] for r in rows], dtype=np.float32),
            np.asarray([r[4] for r in rows], dtype=np.float32),
            np.asarray([r[5] for r in rows], dtype=np.int8),
            [r[0] for r in rows], np.asarray([r[1] for r in rows], dtype=np.float32), stats)


def _shoe_fold(shoe_id: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(str(shoe_id).encode("utf-8")).digest()[:8], "big") % folds


def grouped_fold_ids(shoes: Sequence[str], *, requested_folds: int = DEFAULT_FOLDS) -> tuple[np.ndarray, int]:
    unique = sorted(set(map(str, shoes)))
    folds = min(max(2, int(requested_folds)), len(unique))
    if folds < 2: raise ValueError("need at least two distinct shoe_id values")
    ids = np.asarray([_shoe_fold(str(shoe), folds) for shoe in shoes], dtype=np.int16)
    if len(set(map(int, np.unique(ids)))) < folds:
        mapping = {shoe: i % folds for i, shoe in enumerate(unique)}
        ids = np.asarray([mapping[str(shoe)] for shoe in shoes], dtype=np.int16)
    return ids, folds


def direction_accuracy_from_direction(direction_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean(direction_b.astype(bool) == (actual_b > 0)))


def brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def apply_switch_margin_one(prob_b: float, previous_direction: str, switch_margin: float) -> str:
    p, margin = clip(prob_b), clip(switch_margin, 0.0, 0.05)
    previous = str(previous_direction or "").upper()
    if previous == "B": return "P" if p < 0.5 - margin else "B"
    if previous == "P": return "B" if p > 0.5 + margin else "P"
    return "B" if p > 0.5 else "P"


def apply_switch_margin_sequence(prob_b: np.ndarray, shoes: Sequence[str], round_index: np.ndarray, *, switch_margin: float) -> np.ndarray:
    prob_b, round_index = np.asarray(prob_b, float), np.asarray(round_index, float)
    out = np.zeros(len(prob_b), dtype=np.int8)
    by_shoe: dict[str, list[int]] = {}
    for i, shoe in enumerate(map(str, shoes)): by_shoe.setdefault(shoe, []).append(i)
    for indices in by_shoe.values():
        indices.sort(key=lambda i: (round_index[i], i))
        previous = ""
        for i in indices:
            direction = apply_switch_margin_one(prob_b[i], previous, switch_margin)
            out[i] = 1 if direction == "B" else 0
            previous = direction
    return out


def _run_lengths(values: Sequence[bool], target: bool) -> list[int]:
    runs, n = [], 0
    for value in values:
        if bool(value) == target: n += 1
        elif n: runs.append(n); n = 0
    if n: runs.append(n)
    return runs


def sequence_quality_metrics(predicted_b: np.ndarray, actual_b: np.ndarray, shoes: Sequence[str], round_index: np.ndarray,
                             *, table_size: int = DEFAULT_TABLE_SIZE) -> dict[str, float | int]:
    predicted_b, actual_b, round_index = np.asarray(predicted_b, np.int8), np.asarray(actual_b, np.int8), np.asarray(round_index, float)
    table_size = max(10, int(table_size))
    by_shoe: dict[str, list[int]] = {}
    for i, shoe in enumerate(map(str, shoes)): by_shoe.setdefault(shoe, []).append(i)
    acc, longest_w, longest_l, win3, win4, loss3, all_wr, all_lr = [], [], [], [], [], [], [], []
    for indices in by_shoe.values():
        indices.sort(key=lambda i: (round_index[i], i))
        for start in range(0, len(indices), table_size):
            block = indices[start:start + table_size]
            if len(block) < max(10, table_size // 2): continue
            correct = (predicted_b[block] == actual_b[block]).tolist()
            wr, lr = _run_lengths(correct, True), _run_lengths(correct, False)
            lw, ll = max(wr, default=0), max(lr, default=0)
            acc.append(float(np.mean(correct))); longest_w.append(lw); longest_l.append(ll)
            win3.append(int(lw >= 3)); win4.append(int(lw >= 4)); loss3.append(int(ll >= 3))
            all_wr.extend(wr); all_lr.extend(lr)
    if not acc:
        return {"table_count": 0, "mean_table_accuracy": 0.0, "median_table_accuracy": 0.0,
                "table_ge_52_rate": 0.0, "table_ge_54_rate": 0.0, "table_ge_56_rate": 0.0,
                "win3_table_rate": 0.0, "win4_table_rate": 0.0, "loss3_table_rate": 0.0,
                "mean_longest_win_streak": 0.0, "mean_longest_loss_streak": 0.0,
                "p95_longest_loss_streak": 0.0, "mean_win_run_length": 0.0, "mean_loss_run_length": 0.0,
                "global_max_win_streak": 0, "global_max_loss_streak": 0}
    a = np.asarray(acc, float)
    return {"table_count": int(len(acc)), "mean_table_accuracy": float(np.mean(a)), "median_table_accuracy": float(np.median(a)),
            "table_ge_52_rate": float(np.mean(a >= 0.52)), "table_ge_54_rate": float(np.mean(a >= 0.54)),
            "table_ge_56_rate": float(np.mean(a >= 0.56)), "win3_table_rate": float(np.mean(win3)),
            "win4_table_rate": float(np.mean(win4)), "loss3_table_rate": float(np.mean(loss3)),
            "mean_longest_win_streak": float(np.mean(longest_w)), "mean_longest_loss_streak": float(np.mean(longest_l)),
            "p95_longest_loss_streak": float(np.percentile(longest_l, 95)),
            "mean_win_run_length": float(np.mean(all_wr)) if all_wr else 0.0,
            "mean_loss_run_length": float(np.mean(all_lr)) if all_lr else 0.0,
            "global_max_win_streak": int(max(all_wr, default=0)), "global_max_loss_streak": int(max(all_lr, default=0))}


def evaluate_policy(raw_delta: np.ndarray, x: np.ndarray, actual_b: np.ndarray, shoes: Sequence[str], round_index: np.ndarray,
                    *, max_delta: float, delta_scale: float, switch_margin: float, table_size: int) -> dict[str, Any]:
    delta = np.clip(np.asarray(raw_delta, float) * float(delta_scale), -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)
    core_dir, raw_dir = core_pb > 0.5, final_pb > 0.5
    policy_dir = apply_switch_margin_sequence(final_pb, shoes, round_index, switch_margin=switch_margin).astype(bool)
    core_ok, policy_ok = core_dir == (actual_b > 0), policy_dir == (actual_b > 0)
    flips = core_dir != policy_dir
    rescue, damage = int(np.sum((~core_ok) & policy_ok)), int(np.sum(core_ok & (~policy_ok)))
    core_seq = sequence_quality_metrics(core_dir.astype(np.int8), actual_b, shoes, round_index, table_size=table_size)
    policy_seq = sequence_quality_metrics(policy_dir.astype(np.int8), actual_b, shoes, round_index, table_size=table_size)
    flip_count = int(np.sum(flips)); hold_count = int(np.sum(policy_dir != raw_dir))
    return {"samples": int(len(x)), "core_accuracy": direction_accuracy_from_direction(core_dir, actual_b),
            "corrected_accuracy": direction_accuracy_from_direction(policy_dir, actual_b),
            "accuracy_gain": direction_accuracy_from_direction(policy_dir, actual_b) - direction_accuracy_from_direction(core_dir, actual_b),
            "raw_corrected_accuracy": direction_accuracy_from_direction(raw_dir, actual_b),
            "core_brier": brier(core_pb, actual_b), "corrected_brier": brier(final_pb, actual_b),
            "brier_gain": brier(core_pb, actual_b) - brier(final_pb, actual_b),
            "mean_abs_delta": float(np.mean(np.abs(delta))), "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
            "delta_scale": float(delta_scale), "switch_margin": float(switch_margin), "flip_count": flip_count,
            "flip_rate": float(flip_count / len(x)) if len(x) else 0.0,
            "flip_win_rate": float(np.mean(policy_ok[flips])) if flip_count else 0.0,
            "rescue_count": rescue, "damage_count": damage, "net_flip_gain": rescue - damage,
            "margin_hold_count": hold_count, "margin_hold_rate": float(hold_count / len(x)) if len(x) else 0.0,
            "core_sequence": core_seq, "corrected_sequence": policy_seq}


def candidate_is_safe(m: Mapping[str, Any], *, max_brier_regression: float, max_accuracy_regression: float,
                      min_net_flip_gain: int, max_p95_loss_worsening: float, max_mean_loss_worsening: float) -> bool:
    c, p = m["core_sequence"], m["corrected_sequence"]
    return (float(m["corrected_brier"]) <= float(m["core_brier"]) + max_brier_regression
            and float(m["corrected_accuracy"]) >= float(m["core_accuracy"]) - max_accuracy_regression
            and int(m["net_flip_gain"]) >= min_net_flip_gain
            and float(p["p95_longest_loss_streak"]) <= float(c["p95_longest_loss_streak"]) + max_p95_loss_worsening
            and float(p["mean_longest_loss_streak"]) <= float(c["mean_longest_loss_streak"]) + max_mean_loss_worsening)


def choose_policy(raw_delta: np.ndarray, x: np.ndarray, actual_b: np.ndarray, shoes: Sequence[str], round_index: np.ndarray,
                  *, max_delta: float, scales: Sequence[float], switch_margins: Sequence[float], table_size: int,
                  max_brier_regression: float, max_accuracy_regression: float, min_net_flip_gain: int,
                  max_p95_loss_worsening: float, max_mean_loss_worsening: float) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    candidates = []
    for scale in scales:
        for margin in switch_margins:
            m = evaluate_policy(raw_delta, x, actual_b, shoes, round_index, max_delta=max_delta,
                                delta_scale=clip(scale, 0, 1), switch_margin=clip(margin, 0, 0.05), table_size=table_size)
            m["safe"] = candidate_is_safe(m, max_brier_regression=max_brier_regression,
                                          max_accuracy_regression=max_accuracy_regression,
                                          min_net_flip_gain=min_net_flip_gain,
                                          max_p95_loss_worsening=max_p95_loss_worsening,
                                          max_mean_loss_worsening=max_mean_loss_worsening)
            candidates.append(m)
    safe = [m for m in candidates if m["safe"]]
    pool = safe if safe else candidates
    best = max(pool, key=lambda m: (
        round(float(m["corrected_sequence"]["table_ge_52_rate"]), 12),
        round(float(m["corrected_sequence"]["table_ge_54_rate"]), 12),
        round(float(m["corrected_sequence"]["mean_win_run_length"]), 12),
        round(float(m["corrected_sequence"]["mean_longest_win_streak"]), 12),
        -round(float(m["corrected_sequence"]["loss3_table_rate"]), 12),
        -round(float(m["corrected_sequence"]["p95_longest_loss_streak"]), 12),
        -round(float(m["corrected_sequence"]["mean_longest_loss_streak"]), 12),
        round(float(m["corrected_accuracy"]), 12), -round(float(m["corrected_brier"]), 12),
        -round(float(m["switch_margin"]), 12), -round(float(m["delta_scale"]), 12)))
    return dict(best), candidates, bool(safe)


def grouped_oof_predictions(x: np.ndarray, residual: np.ndarray, shoes: Sequence[str], *, requested_folds: int,
                            random_state: int) -> tuple[np.ndarray, int, list[dict[str, int]]]:
    fold_ids, folds = grouped_fold_ids(shoes, requested_folds=requested_folds)
    raw_oof = np.full(len(x), np.nan, dtype=float); fold_stats = []
    for fold in range(folds):
        validation, train = fold_ids == fold, fold_ids != fold
        if not np.any(validation) or not np.any(train): raise RuntimeError(f"invalid grouped fold {fold}")
        model = build_regressor(random_state=random_state); model.fit(x[train], residual[train])
        raw_oof[validation] = np.asarray(model.predict(x[validation]), dtype=float)
        fold_stats.append({"fold": fold, "train_rows": int(np.sum(train)), "validation_rows": int(np.sum(validation)),
                           "validation_shoes": int(len(set(np.asarray(shoes, object)[validation].tolist())))})
    if not np.all(np.isfinite(raw_oof)): raise RuntimeError("OOF prediction generation left non-finite rows")
    return raw_oof, folds, fold_stats


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node: Mapping[str, Any] = tree
    for _ in range(256):
        if "leaf" in node: return float(node.get("leaf", 0.0))
        split = str(node.get("split", ""))
        if split.startswith("f") and split[1:].isdigit(): index = int(split[1:])
        else:
            try: index = FEATURE_NAMES.index(split)
            except ValueError: index = -1
        value = float(np.float32(vector[index])) if 0 <= index < len(vector) else math.nan
        threshold = float(np.float32(node.get("split_condition", 0.0)))
        next_id = node.get("missing") if not math.isfinite(value) else (node.get("yes") if value < threshold else node.get("no"))
        found = next((c for c in node.get("children") or [] if int(c.get("nodeid", -999)) == int(next_id)), None)
        if found is None: return 0.0
        node = found
    return 0.0


def export_portable_bundle(model: XGBRegressor, *, reference_x: np.ndarray, output_path: Path, max_delta: float,
                           delta_scale: float, switch_margin: float, table_size: int, metrics: Mapping[str, Any],
                           training_rows: int, validation_folds: int, data_stats: Mapping[str, Any]) -> dict[str, Any]:
    trees = [json.loads(text) for text in model.get_booster().get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf(tree, reference) for tree in trees)
    base_score = float(model.predict(reference.reshape(1, -1))[0]) - tree_sum
    for vector in np.asarray(reference_x[:min(64, len(reference_x))], dtype=float):
        portable = base_score + sum(_tree_leaf(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5: raise RuntimeError(f"portable export mismatch: {portable} vs {native}")
    bundle = {"schema_version": SCHEMA_VERSION, "model_type": MODEL_TYPE, "model_version": MODEL_VERSION,
              "base_core_version": BASE_CORE_VERSION, "trained": True, "feature_names": list(FEATURE_NAMES),
              "base_score": float(base_score), "max_delta": float(max_delta), "delta_scale": float(delta_scale),
              "switch_margin": float(switch_margin), "table_size": int(table_size), "trees": trees,
              "training": {"rows": int(training_rows), "target": "actual_B_minus_core_p_B",
                           "decision_rule": "B/P with validation-selected switch hysteresis; no PASS", "no_pass": True,
                           "validation": f"grouped_{validation_folds}_fold_OOF_by_shoe_id",
                           "metrics": dict(metrics), "data_stats": dict(data_stats)}}
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    x, residual, actual_b, shoes, round_index, data_stats = make_training_arrays(records)
    if len(x) < args.min_samples: raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")
    unique_shoes = len(set(shoes))
    if unique_shoes < args.min_shoes: raise SystemExit(f"need at least {args.min_shoes} distinct shoe_id values; got {unique_shoes}")
    raw_oof, folds, fold_stats = grouped_oof_predictions(x, residual, shoes, requested_folds=args.folds, random_state=args.random_state)
    selected, candidates, had_safe = choose_policy(raw_oof, x, actual_b, shoes, round_index,
        max_delta=args.max_delta, scales=args.delta_scales, switch_margins=args.switch_margins,
        table_size=args.table_size, max_brier_regression=args.max_brier_regression,
        max_accuracy_regression=args.max_accuracy_regression, min_net_flip_gain=args.min_net_flip_gain,
        max_p95_loss_worsening=args.max_p95_loss_worsening, max_mean_loss_worsening=args.max_mean_loss_worsening)
    validation = dict(selected)
    if int(validation["corrected_sequence"]["table_count"]) < args.min_tables:
        selected["safe"] = False; validation["safe"] = False; validation["min_tables_gate_failed"] = True
    validation.update({"folds": folds, "unique_shoes": unique_shoes, "fold_stats": fold_stats,
                       "table_size": int(args.table_size), "safe_candidate_found": had_safe,
                       "policy_candidates": candidates})
    accepted = bool(selected.get("safe")) and had_safe
    report = {"model_version": MODEL_VERSION, "base_core_version": BASE_CORE_VERSION, "validation": validation,
              "data_stats": data_stats, "selected_delta_scale": float(selected["delta_scale"]),
              "selected_switch_margin": float(selected["switch_margin"]), "accepted": accepted}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected V1.2 streak policy; use --force only for diagnostics")
    final_model = build_regressor(random_state=args.random_state); final_model.fit(x, residual)
    export_portable_bundle(final_model, reference_x=x, output_path=Path(args.output), max_delta=args.max_delta,
                           delta_scale=float(selected["delta_scale"]), switch_margin=float(selected["switch_margin"]),
                           table_size=args.table_size, metrics=validation, training_rows=len(x),
                           validation_folds=folds, data_stats=data_stats)
    print(f"wrote {args.output}"); return 0


def _parse_grid(text: str, *, upper: float, allow_zero: bool) -> tuple[float, ...]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part: continue
        value = clip(float(part), 0.0, upper)
        if (allow_zero or value > 0) and value not in values: values.append(value)
    if not values: raise argparse.ArgumentTypeError("grid must contain at least one valid value")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost residual bias trainer V1.2 streak quality")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", required=True); train.add_argument("--output", default="residual_bias_model.json")
    train.add_argument("--min-samples", type=int, default=500); train.add_argument("--min-shoes", type=int, default=20)
    train.add_argument("--min-tables", type=int, default=20); train.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train.add_argument("--table-size", type=int, default=DEFAULT_TABLE_SIZE); train.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA)
    train.add_argument("--delta-scales", type=lambda s: _parse_grid(s, upper=1.0, allow_zero=False), default=DEFAULT_DELTA_SCALES)
    train.add_argument("--switch-margins", type=lambda s: _parse_grid(s, upper=0.05, allow_zero=True), default=DEFAULT_SWITCH_MARGINS)
    train.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train.add_argument("--max-brier-regression", type=float, default=0.0); train.add_argument("--max-accuracy-regression", type=float, default=0.0)
    train.add_argument("--min-net-flip-gain", type=int, default=0); train.add_argument("--max-p95-loss-worsening", type=float, default=0.0)
    train.add_argument("--max-mean-loss-worsening", type=float, default=0.0); train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command); return parser


def main() -> int:
    args = build_parser().parse_args(); return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
