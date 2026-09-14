#!/usr/bin/env python3
"""BBB XGBoost residual-bias predictor with dual-window regime damping.

The deterministic 256D/V23 core remains untouched. XGBRegressor learns only
the calibration residual, not the raw B/P class:

    residual = actual_B - core_p_B

Two S/X regime features are added:
- sx_entropy_18: long-window normalized Shannon entropy of the latest 18
  SAME/SWITCH transition states.
- sx_transition_change_6: short-window signed change in X (SWITCH) frequency,
  comparing the latest 3 S/X tokens with the preceding 3.

Production logic:

    delta = clip(xgb_residual, -0.10, +0.10)
    damper = f(sx_entropy_18)
    final_p_B = 0.50 + ((core_p_B - 0.50) + delta) * damper
    direction = "B" if final_p_B > 0.50 else "P"

There is never a PASS state. A continuous bet-weight signal is derived from the
absolute final edge around 0.50. In a high-entropy regime, the damper contracts
that edge toward 0.50, which naturally lowers the sizing signal.
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
    "stage",
    "depth",
    "sx_entropy_18",
    "sx_transition_change_6",
)
MODEL_TYPE = "xgb_residual_regressor"
SCHEMA_VERSION = 3
DEFAULT_MAX_DELTA = 0.10
DEFAULT_RANDOM_STATE = 20260915

DEFAULT_ENTROPY_WINDOW = 18
DEFAULT_ENTROPY_MIN_TOKENS = 6
DEFAULT_TRANSITION_CHANGE_WINDOW = 6
DEFAULT_TRANSITION_CHANGE_MIN_TOKENS = 6
DEFAULT_REGULAR_ENTROPY_MAX = 0.55
DEFAULT_HIGH_ENTROPY_START = 0.85
DEFAULT_HIGH_DAMPER_MAX = 0.40
DEFAULT_HIGH_DAMPER_MIN = 0.20
DEFAULT_BET_WEIGHT_MIN = 0.20
DEFAULT_BET_REFERENCE_EDGE = 0.18


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def clip_signed(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


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


def sx_entropy(history: Sequence[str], *, window: int = DEFAULT_ENTROPY_WINDOW, min_tokens: int = DEFAULT_ENTROPY_MIN_TOKENS) -> float:
    tokens = transition_sequence(history)[-max(2, int(window)) :]
    if len(tokens) < max(2, int(min_tokens)):
        return 0.0
    p_s = sum(token == "S" for token in tokens) / len(tokens)
    p_x = 1.0 - p_s
    entropy = 0.0
    for p in (p_s, p_x):
        if p > 0.0:
            entropy -= p * math.log2(p)
    return clip(entropy)


def sx_transition_change(history: Sequence[str], *, window: int = DEFAULT_TRANSITION_CHANGE_WINDOW, min_tokens: int = DEFAULT_TRANSITION_CHANGE_MIN_TOKENS) -> float:
    width = max(4, int(window))
    tokens = transition_sequence(history)[-width:]
    if len(tokens) < max(4, int(min_tokens)):
        return 0.0
    split = len(tokens) // 2
    older = tokens[:split]
    newer = tokens[split:]
    if not older or not newer:
        return 0.0
    older_x = sum(token == "X" for token in older) / len(older)
    newer_x = sum(token == "X" for token in newer) / len(newer)
    return clip_signed(newer_x - older_x)


def dynamic_damper(entropy_18: float, *, regular_max: float = DEFAULT_REGULAR_ENTROPY_MAX, high_start: float = DEFAULT_HIGH_ENTROPY_START, high_damper_max: float = DEFAULT_HIGH_DAMPER_MAX, high_damper_min: float = DEFAULT_HIGH_DAMPER_MIN) -> float:
    e = clip(entropy_18)
    regular_max = clip(regular_max, 0.0, 0.95)
    high_start = clip(high_start, regular_max + 1e-6, 0.999999)
    high_damper_max = clip(high_damper_max, 0.20, 1.0)
    high_damper_min = clip(high_damper_min, 0.0, high_damper_max)
    if e <= regular_max:
        return 1.0
    if e < high_start:
        t = (e - regular_max) / (high_start - regular_max)
        return clip(1.0 - t * (1.0 - high_damper_max), high_damper_max, 1.0)
    t = (e - high_start) / (1.0 - high_start)
    return clip(high_damper_max - t * (high_damper_max - high_damper_min), high_damper_min, high_damper_max)


def bet_weight_from_probability(final_p_b: float, *, minimum: float = DEFAULT_BET_WEIGHT_MIN, reference_edge: float = DEFAULT_BET_REFERENCE_EDGE) -> tuple[float, str]:
    edge = abs(clip(final_p_b) - 0.5)
    minimum = clip(minimum, 0.0, 1.0)
    reference_edge = max(1e-6, float(reference_edge))
    weight = minimum + (1.0 - minimum) * clip(edge / reference_edge)
    if weight < 0.40:
        tier = "LOW"
    elif weight < 0.70:
        tier = "MEDIUM"
    else:
        tier = "HIGH"
    return float(weight), tier


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
    stage: float
    depth: float
    sx_entropy_18: float
    sx_transition_change_6: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float32)


def build_features(*, core_p_b: float, history: str | Sequence[str], estimated_total_hands: float = 60.0, stage: float | None = None, depth: float | None = None) -> ResidualFeatures:
    seq = normalize_history(history)
    total_hands = clip(float(estimated_total_hands), 40.0, 90.0)
    round_index = float(max(1, min(70, len(seq) + 1)))
    remaining_ratio = clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b), 0.0, 1.0),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        stage=float(current_stage(seq) if stage is None else stage),
        depth=float(current_depth(seq) if depth is None else depth),
        sx_entropy_18=sx_entropy(seq),
        sx_transition_change_6=sx_transition_change(seq),
    )


def build_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", n_estimators=240, max_depth=3, learning_rate=0.03, min_child_weight=8.0, subsample=0.85, colsample_bytree=0.90, reg_alpha=0.20, reg_lambda=8.0, gamma=0.0, random_state=int(random_state), n_jobs=1, tree_method="hist", verbosity=0)


class DualWindowResidualBiasPredictor:
    def __init__(self, model: XGBRegressor | None = None, *, max_delta: float = DEFAULT_MAX_DELTA, regular_entropy_max: float = DEFAULT_REGULAR_ENTROPY_MAX, high_entropy_start: float = DEFAULT_HIGH_ENTROPY_START, high_damper_max: float = DEFAULT_HIGH_DAMPER_MAX, high_damper_min: float = DEFAULT_HIGH_DAMPER_MIN, bet_weight_min: float = DEFAULT_BET_WEIGHT_MIN, bet_reference_edge: float = DEFAULT_BET_REFERENCE_EDGE) -> None:
        self.model = model or build_regressor()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)
        self.regular_entropy_max = float(regular_entropy_max)
        self.high_entropy_start = float(high_entropy_start)
        self.high_damper_max = float(high_damper_max)
        self.high_damper_min = float(high_damper_min)
        self.bet_weight_min = float(bet_weight_min)
        self.bet_reference_edge = float(bet_reference_edge)

    def assemble_features(self, *, core_p_b: float, history: str | Sequence[str], estimated_total_hands: float = 60.0, stage: float | None = None, depth: float | None = None) -> ResidualFeatures:
        return build_features(core_p_b=core_p_b, history=history, estimated_total_hands=estimated_total_hands, stage=stage, depth=depth)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "DualWindowResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        residual = y - core_pb
        self.model.fit(x, residual)
        return self

    def predict_delta(self, feature_row: Sequence[float]) -> float:
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        raw = float(self.model.predict(x)[0])
        return clip(raw, -self.max_delta, self.max_delta)

    def damper(self, entropy_18: float) -> float:
        return dynamic_damper(entropy_18, regular_max=self.regular_entropy_max, high_start=self.high_entropy_start, high_damper_max=self.high_damper_max, high_damper_min=self.high_damper_min)

    def correct(self, feature_row: Sequence[float]) -> dict[str, Any]:
        x = np.asarray(feature_row, dtype=np.float32)
        core_pb = clip(float(x[FEATURE_NAMES.index("core_p_b")]))
        entropy_18 = clip(float(x[FEATURE_NAMES.index("sx_entropy_18")]))
        short_change = clip_signed(float(x[FEATURE_NAMES.index("sx_transition_change_6")]))
        delta = self.predict_delta(x)
        damper = self.damper(entropy_18)
        undamped_pb = clip(core_pb + delta)
        final_pb = clip(0.5 + ((core_pb - 0.5) + delta) * damper)
        bet_weight, bet_tier = bet_weight_from_probability(final_pb, minimum=self.bet_weight_min, reference_edge=self.bet_reference_edge)
        return {"core_p_b": core_pb, "delta": delta, "sx_entropy_18": entropy_18, "sx_transition_change_6": short_change, "damper": damper, "undamped_p_b": undamped_pb, "final_p_b": final_pb, "direction": "B" if final_pb > 0.5 else "P", "bet_weight": bet_weight, "bet_weight_tier": bet_tier}

    def predict_from_context(self, *, core_p_b: float, history: str | Sequence[str], estimated_total_hands: float = 60.0, stage: float | None = None, depth: float | None = None) -> dict[str, Any]:
        features = self.assemble_features(core_p_b=core_p_b, history=history, estimated_total_hands=estimated_total_hands, stage=stage, depth=depth)
        result = self.correct(features.as_vector())
        result["features"] = features.as_dict()
        return result


DynamicResidualBiasPredictor = DualWindowResidualBiasPredictor
ResidualBiasPredictor = DualWindowResidualBiasPredictor


def _parse_actual_b(record: Mapping[str, Any]) -> int:
    if "actual_b" in record and record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").upper().strip()
    if actual == "B":
        return 1
    if actual == "P":
        return 0
    raise ValueError("training row must contain actual_outcome B/P or actual_b 0/1")


def _feature_row(record: Mapping[str, Any]) -> dict[str, float]:
    if all(name in record for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}
    features = build_features(core_p_b=float(record.get("core_p_b", record.get("core_pb", 0.5))), history=record.get("history") or record.get("history_fingerprint") or "", estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0), stage=(float(record["stage"]) if record.get("stage") is not None else None), depth=(float(record["depth"]) if record.get("depth") is not None else None)).as_dict()
    for name in FEATURE_NAMES:
        if name in record and record.get(name) is not None:
            features[name] = float(record[name])
    return features


def load_training_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON training file must be a list or {'rows': [...]} bundle")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("training input must be .json or .csv")


def make_training_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    for idx, record in enumerate(records):
        try:
            actual_b = _parse_actual_b(record)
            row = _feature_row(record)
            vector = [float(row[name]) for name in FEATURE_NAMES]
            if not all(math.isfinite(x) for x in vector):
                continue
            core_p_b = clip(row["core_p_b"])
        except (TypeError, ValueError, KeyError):
            continue
        vectors.append(vector)
        residuals.append(float(actual_b) - core_p_b)
        actuals.append(actual_b)
        shoes.append(str(record.get("shoe_id") or f"row_{idx}"))
    if not vectors:
        raise ValueError("no valid B/P training rows")
    return np.asarray(vectors, dtype=np.float32), np.asarray(residuals, dtype=np.float32), np.asarray(actuals, dtype=np.int8), shoes


def deterministic_validation_mask(shoes: Sequence[str], *, fraction: float = 0.20) -> np.ndarray:
    threshold = int(256 * clip(fraction, 0.05, 0.50))
    result = np.asarray([hashlib.sha256(str(shoe).encode("utf-8")).digest()[0] < threshold for shoe in shoes], dtype=bool)
    if result.all() or (~result).all():
        n = len(result)
        cut = max(1, int(round(n * (1.0 - fraction))))
        result[:] = False
        result[cut:] = True
    return result


def direction_accuracy(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b > 0.5) == (actual_b > 0)))


def brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def _damp_array(core_pb: np.ndarray, delta: np.ndarray, entropy_18: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dampers = np.asarray([dynamic_damper(float(x)) for x in entropy_18], dtype=float)
    final_pb = np.clip(0.5 + ((core_pb - 0.5) + delta) * dampers, 0.0, 1.0)
    return final_pb, dampers


def evaluate(model: XGBRegressor, x: np.ndarray, actual_b: np.ndarray, *, max_delta: float) -> dict[str, float]:
    raw_delta = np.asarray(model.predict(x), dtype=float)
    delta = np.clip(raw_delta, -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    entropy_18 = x[:, FEATURE_NAMES.index("sx_entropy_18")].astype(float)
    short_change = x[:, FEATURE_NAMES.index("sx_transition_change_6")].astype(float)
    undamped_pb = np.clip(core_pb + delta, 0.0, 1.0)
    final_pb, dampers = _damp_array(core_pb, delta, entropy_18)
    return {"samples": float(len(x)), "core_accuracy": direction_accuracy(core_pb, actual_b), "residual_accuracy": direction_accuracy(undamped_pb, actual_b), "corrected_accuracy": direction_accuracy(final_pb, actual_b), "core_brier": brier(core_pb, actual_b), "residual_brier": brier(undamped_pb, actual_b), "corrected_brier": brier(final_pb, actual_b), "mean_abs_delta": float(np.mean(np.abs(delta))), "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0, "mean_entropy_18": float(np.mean(entropy_18)) if len(entropy_18) else 0.0, "mean_abs_transition_change_6": float(np.mean(np.abs(short_change))) if len(short_change) else 0.0, "mean_damper": float(np.mean(dampers)) if len(dampers) else 1.0}


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node: Mapping[str, Any] = tree
    guard = 0
    while guard < 256:
        guard += 1
        if "leaf" in node:
            return float(node.get("leaf", 0.0))
        split = str(node.get("split", ""))
        if split.startswith("f") and split[1:].isdigit():
            index = int(split[1:])
        else:
            try:
                index = FEATURE_NAMES.index(split)
            except ValueError:
                index = -1
        value = float(np.float32(vector[index])) if 0 <= index < len(vector) else math.nan
        split_condition = float(np.float32(node.get("split_condition", 0.0)))
        next_id = node.get("missing") if not math.isfinite(value) else (node.get("yes") if value < split_condition else node.get("no"))
        children = node.get("children") or []
        found = next((child for child in children if int(child.get("nodeid", -999)) == int(next_id)), None)
        if found is None:
            return 0.0
        node = found
    return 0.0


def export_portable_bundle(model: XGBRegressor, *, reference_x: np.ndarray, output_path: Path, max_delta: float, metrics: Mapping[str, Any], training_rows: int) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(text) for text in booster.get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf(tree, reference) for tree in trees)
    model_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = model_reference - tree_sum
    for vector in np.asarray(reference_x[: min(64, len(reference_x))], dtype=float):
        portable = base_score + sum(_tree_leaf(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(f"portable export mismatch: {portable} vs {native}")
    bundle = {"schema_version": SCHEMA_VERSION, "model_type": MODEL_TYPE, "trained": True, "feature_names": list(FEATURE_NAMES), "base_score": float(base_score), "max_delta": float(max_delta), "dual_window": {"long_feature": "sx_entropy_18", "long_window": DEFAULT_ENTROPY_WINDOW, "long_min_tokens": DEFAULT_ENTROPY_MIN_TOKENS, "short_feature": "sx_transition_change_6", "short_window": DEFAULT_TRANSITION_CHANGE_WINDOW, "short_definition": "recent_3_X_rate_minus_previous_3_X_rate"}, "damper": {"feature": "sx_entropy_18", "regular_max": DEFAULT_REGULAR_ENTROPY_MAX, "high_start": DEFAULT_HIGH_ENTROPY_START, "high_damper_max": DEFAULT_HIGH_DAMPER_MAX, "high_damper_min": DEFAULT_HIGH_DAMPER_MIN}, "bet_weight": {"minimum": DEFAULT_BET_WEIGHT_MIN, "reference_edge": DEFAULT_BET_REFERENCE_EDGE}, "trees": trees, "training": {"rows": int(training_rows), "target": "actual_B_minus_core_p_B", "decision_rule": "B if final_p_B > 0.50 else P", "final_probability": "0.50 + ((core_p_B - 0.50) + delta) * damper", "no_pass": True, "metrics": dict(metrics)}}
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    x, residual, actual_b, shoes = make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")
    validation = deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~validation
    model = build_regressor(random_state=args.random_state)
    model.fit(x[train], residual[train])
    validation_metrics = evaluate(model, x[validation], actual_b[validation], max_delta=args.max_delta)
    accepted = validation_metrics["corrected_brier"] <= validation_metrics["core_brier"] + args.max_brier_regression and validation_metrics["corrected_accuracy"] >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    print(json.dumps({"validation": validation_metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected dual-window residual+damper model; use --force only for diagnostics")
    final_model = build_regressor(random_state=args.random_state)
    final_model.fit(x, residual)
    export_portable_bundle(final_model, reference_x=x, output_path=Path(args.output), max_delta=args.max_delta, metrics=validation_metrics, training_rows=len(x))
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost residual bias + dual-window dynamic damper trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train XGBRegressor and export browser model JSON")
    train.add_argument("--input", required=True, help="browser-exported JSON or CSV")
    train.add_argument("--output", default="residual_bias_model.json")
    train.add_argument("--min-samples", type=int, default=500)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA)
    train.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train.add_argument("--max-brier-regression", type=float, default=0.0)
    train.add_argument("--max-accuracy-regression", type=float, default=0.005)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
