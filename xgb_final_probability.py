#!/usr/bin/env python3
"""Direct 56D XGBoost probability layer for the frozen BBB core.

The 256D JavaScript cores remain unchanged.  Their ``Core P(B)`` is only one
input feature here; it is never added to a residual prediction.

Feature order (fixed):
    [Core P(B)] + [progress_w] + [original 7D without duplicate Core P(B)] + [physics 48D] = 56D

Target (fixed):
    actual_B, where Player=0 and Banker=1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

try:  # Keep vector-only tests usable before requirements-xgb.txt is installed.
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean dev envs
    XGBClassifier = None  # type: ignore[assignment,misc]

from physics_feature_extractor import (
    PHYSICS_DIM,
    PHYSICS_FEATURE_NAMES,
    RANK_LABELS,
    SUITS,
    PhysicsFeatureExtractor,
)

MODEL_TYPE = "xgb_final_probability_classifier"
RANDOM_STATE = 20260923
FEATURE_DIM = 56
PROBABILITY_BOUNDS = (0.40, 0.60)
EARLY_PROBABILITY_BOUNDS = (0.45, 0.55)
LATE_PROBABILITY_BOUNDS = (0.35, 0.65)
SNAPSHOT_SCHEMA_VERSION = 6
PHYSICS_PROBABILITY_TOLERANCE = 1e-4
PHYSICS_RANK_CONSUMPTION_TOLERANCE = 0.50
ORIGINAL_7D_FEATURE_NAMES = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "sx_markov_p_same",
    "stage",
    "depth",
)

FEATURE_NAMES: tuple[str, ...] = (
    ("core_p_b_external", "shoe_progress_weight")
    + tuple(f"original7_{name}" for name in ORIGINAL_7D_FEATURE_NAMES[1:])
    + PHYSICS_FEATURE_NAMES
)
assert len(FEATURE_NAMES) == FEATURE_DIM

NEXT_CARD_COUNT_LABELS = ("4", "5", "6")
_NEXT_CARD_COUNT_NAMES = tuple(f"cards_p{label}" for label in NEXT_CARD_COUNT_LABELS)
_NEXT_RANK_CONSUMPTION_NAMES = tuple(f"next_rank_expected_{label}" for label in RANK_LABELS)
_NEXT_SUIT_RATIO_NAMES = tuple(f"next_suit_ratio_{suit}" for suit in SUITS)
_PHYSICS_INDEX = {name: index for index, name in enumerate(PHYSICS_FEATURE_NAMES)}
assert all(name in _PHYSICS_INDEX for name in _NEXT_CARD_COUNT_NAMES + _NEXT_RANK_CONSUMPTION_NAMES + _NEXT_SUIT_RATIO_NAMES)


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


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _probability_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("probability_bounds must contain exactly (min, max)")
    lo, hi = (float(bounds[0]), float(bounds[1]))
    if not (0.0 <= lo <= 0.5 <= hi <= 1.0):
        raise ValueError("probability bounds must satisfy 0 <= min <= .5 <= max <= 1")
    return lo, hi


def _physics_vector(physics_48d: Sequence[float]) -> np.ndarray:
    """Validate the already-produced 48D physical forecast without simulating it."""
    physics = np.asarray(physics_48d, dtype=np.float32).reshape(-1)
    if physics.size != PHYSICS_DIM:
        raise ValueError(f"physics_48d must contain exactly {PHYSICS_DIM} values")
    if not np.all(np.isfinite(physics)):
        raise ValueError("physics_48d must contain only finite values")
    return physics


def unpack_physics_forecast(physics_48d: Sequence[float]) -> dict[str, Any]:
    """Decode next-hand physical estimates embedded in the existing 48D vector.

    This is a pure view of the supplied feature block: it does not invoke, train,
    or alter any simulation/MCMC component.  Suit ratios are also converted to
    expected cards by multiplying them by the decoded next-hand card expectation.
    """
    physics = _physics_vector(physics_48d)
    value = lambda name: float(physics[_PHYSICS_INDEX[name]])

    card_count_probabilities = {
        f"{label}_cards": value(f"cards_p{label}")
        for label in NEXT_CARD_COUNT_LABELS
    }
    expected_next_card_count = sum(
        int(label) * card_count_probabilities[f"{label}_cards"]
        for label in NEXT_CARD_COUNT_LABELS
    )
    rank_expected_consumption = {
        label: value(f"next_rank_expected_{label}")
        for label in RANK_LABELS
    }
    suit_consumption_ratios = {
        suit: value(f"next_suit_ratio_{suit}")
        for suit in SUITS
    }
    suit_expected_consumption = {
        suit: expected_next_card_count * ratio
        for suit, ratio in suit_consumption_ratios.items()
    }
    return {
        "next_card_count_probabilities": card_count_probabilities,
        "expected_next_card_count": float(expected_next_card_count),
        "next_rank_expected_consumption": rank_expected_consumption,
        "next_suit_consumption_ratios": suit_consumption_ratios,
        "next_suit_expected_consumption": suit_expected_consumption,
    }


def physics_integrity_report(physics_48d: Sequence[float]) -> dict[str, Any]:
    """Report whether the supplied 48D physical forecast is self-consistent.

    The bridge deliberately does not alter or simulate the physics block.  This
    report is diagnostic metadata for snapshots and model validation: card-count
    and suit distributions should normalise to one, while the A-K consumption
    total should be close to the decoded expected next-hand card count.
    """
    forecast = unpack_physics_forecast(physics_48d)
    card_probability_sum = float(sum(forecast["next_card_count_probabilities"].values()))
    suit_ratio_sum = float(sum(forecast["next_suit_consumption_ratios"].values()))
    expected_card_count = float(forecast["expected_next_card_count"])
    rank_expected_total = float(sum(forecast["next_rank_expected_consumption"].values()))
    rank_total_gap = abs(rank_expected_total - expected_card_count)
    rank_tolerance = max(
        PHYSICS_RANK_CONSUMPTION_TOLERANCE,
        expected_card_count * 0.10,
    )
    checks = {
        "card_count_distribution": abs(card_probability_sum - 1.0) <= PHYSICS_PROBABILITY_TOLERANCE,
        "suit_ratio_distribution": abs(suit_ratio_sum - 1.0) <= PHYSICS_PROBABILITY_TOLERANCE,
        "rank_consumption_total": rank_total_gap <= rank_tolerance,
    }
    return {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "card_count_probability_sum": card_probability_sum,
        "expected_next_card_count": expected_card_count,
        "rank_expected_consumption_total": rank_expected_total,
        "rank_expected_total_gap": rank_total_gap,
        "rank_expected_total_tolerance": rank_tolerance,
        "suit_ratio_sum": suit_ratio_sum,
    }


def shoe_progress_weight(round_index: float) -> float:
    return float((max(0.0, float(round_index)) / 70.0) ** 2)


def dynamic_probability_bounds(round_index: float) -> tuple[float, float]:
    if float(round_index) <= 40.0:
        return EARLY_PROBABILITY_BOUNDS
    if float(round_index) > 50.0:
        return LATE_PROBABILITY_BOUNDS
    return PROBABILITY_BOUNDS

def build_56d_feature_matrix(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
) -> np.ndarray:
    """Build the direct classifier vector without duplicate Original-7D Core P(B)."""
    core = np.asarray(core_pb, dtype=np.float32).reshape(-1)
    original = np.asarray(original_7d, dtype=np.float32).reshape(-1)
    physics = _physics_vector(physics_48d)

    if core.size != 1:
        raise ValueError("core_pb must contain exactly one probability")
    if not 0.0 <= float(core[0]) <= 1.0:
        raise ValueError("core_pb must be within [0, 1]")
    if original.size != 7:
        raise ValueError("original_7d must contain exactly 7 values")
    if not np.all(np.isfinite(original)):
        raise ValueError("feature blocks must contain only finite values")

    progress = shoe_progress_weight(float(original[1]))
    merged = np.hstack((core, [progress], original[1:], physics)).astype(np.float32, copy=False)
    if merged.size != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {merged.size}")
    return merged.reshape(1, FEATURE_DIM)


def build_xgboost_classifier(*, random_state: int = RANDOM_STATE) -> Any:
    """Conservative direct-classification configuration; no residual target."""
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required; install requirements-xgb.txt")
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=320,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.35,
        reg_lambda=12.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def _positive_class_probability(model: Any, features: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise RuntimeError("predict_proba must return one row per feature row")
    classes = np.asarray(getattr(model, "classes_", (0, 1)))
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1 or probabilities.shape[1] <= int(positive[0]):
        raise RuntimeError("XGBoost classifier must expose class 1 (Banker)")
    return _clip(float(probabilities[0, int(positive[0])]))


def _direct_prediction_payload(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Create the direct XGBoost result and decode its 48D physical forecast."""
    physics = _physics_vector(physics_48d)
    features = build_56d_feature_matrix(core_pb, original_7d, physics)
    raw_pb = _positive_class_probability(xgboost_model, features)
    round_index = float(np.asarray(original_7d, dtype=np.float32).reshape(-1)[1])
    lo, hi = dynamic_probability_bounds(round_index)
    final_pb = _clip(raw_pb, lo, hi)
    return {
        "core_p_b": float(core_pb),
        "raw_p_b": raw_pb,
        "final_p_b": final_pb,
        "direction": "B" if final_pb > 0.50 else "P",
        "probability_bounds": {"min": lo, "max": hi},
        "features": features,
        "physics_forecast": unpack_physics_forecast(physics),
        "shoe_progress_weight": shoe_progress_weight(round_index),
        "physics_integrity": physics_integrity_report(physics),
    }


def predict_final_probability(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Return final P(B) and unpacked next-hand physical estimates.

    The former ``Core P(B) + Delta`` operation does not exist in this path.
    ``[0.40, 0.60]`` is the direct-probability equivalent of the previous
    neutral-centred +/-0.10 safety envelope.  ``final_p_b`` is obtained from
    ``xgboost_model.predict_proba(features)[:, 1]`` (the Banker class).
    """
    return _direct_prediction_payload(
        core_pb,
        original_7d,
        physics_48d,
        xgboost_model=xgboost_model,
        probability_bounds=probability_bounds,
    )


def predict_final_result(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Compatibility name for the complete direct probability result payload."""
    return predict_final_probability(
        core_pb,
        original_7d,
        physics_48d,
        xgboost_model=xgboost_model,
        probability_bounds=probability_bounds,
    )


def _actual_b(record: Mapping[str, Any]) -> int:
    if record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    outcome = str(record.get("actual_outcome") or record.get("actual") or "").upper()
    if outcome == "B":
        return 1
    if outcome == "P":
        return 0
    raise ValueError("only directional B/P rows are valid training targets")


def _history(record: Mapping[str, Any]) -> str | Sequence[str]:
    return record.get("history") or record.get("history_fingerprint") or ""


def _core_pb(record: Mapping[str, Any]) -> float:
    value = record.get("core_p_b", record.get("core_pb"))
    if value is None:
        raise ValueError("missing core_p_b")
    return _clip(float(value))


def _history_tokens(history: str | Sequence[str]) -> list[str]:
    values = history.upper() if isinstance(history, str) else history
    return [str(value).strip().upper() for value in values if str(value).strip().upper() in {"B", "P", "T"}]


def _transition_tokens(history: Sequence[str]) -> list[str]:
    directional = [value for value in history if value in {"B", "P"}]
    return ["S" if directional[index] == directional[index - 1] else "X" for index in range(1, len(directional))]


def _current_stage(history: Sequence[str]) -> int:
    directional = [value for value in history if value in {"B", "P"}]
    if not directional:
        return 0
    side, count = directional[-1], 1
    for value in reversed(directional[:-1]):
        if value != side:
            break
        count += 1
    return count


def _current_depth(history: Sequence[str]) -> int:
    transitions = _transition_tokens(history)
    if not transitions:
        return 0
    token, count = transitions[-1], 1
    for value in reversed(transitions[:-1]):
        if value != token:
            break
        count += 1
    return count


def _sx_markov_p_same(history: Sequence[str]) -> float:
    transitions = _transition_tokens(history)
    if not transitions:
        return 0.5
    current = transitions[-1]
    start = max(0, len(transitions) - 1 - 24)
    same = switched = 0
    for index in range(start, len(transitions) - 1):
        if transitions[index] != current:
            continue
        if transitions[index + 1] == "S":
            same += 1
        else:
            switched += 1
    return _clip((same + 1.0) / (same + switched + 2.0))


def _rebuild_original_7d(record: Mapping[str, Any], core_pb: float) -> np.ndarray:
    history = _history_tokens(_history(record))
    total_hands = _clip(float(record.get("estimated_total_hands", 60) or 60), 40.0, 90.0)
    round_index = float(max(1, min(70, len(history) + 1)))
    stage = float(record["stage"]) if record.get("stage") is not None else float(_current_stage(history))
    depth = float(record["depth"]) if record.get("depth") is not None else float(_current_depth(history))
    return np.asarray([
        core_pb,
        round_index,
        total_hands,
        _clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands)),
        _sx_markov_p_same(history),
        stage,
        depth,
    ], dtype=np.float32)


def _original_7d(record: Mapping[str, Any], core_pb: float) -> np.ndarray:
    if all(record.get(name) is not None for name in ORIGINAL_7D_FEATURE_NAMES):
        return np.asarray([float(record[name]) for name in ORIGINAL_7D_FEATURE_NAMES], dtype=np.float32)
    return _rebuild_original_7d(record, core_pb)


def _physics_48d(record: Mapping[str, Any], extractor: PhysicsFeatureExtractor | None) -> np.ndarray:
    supplied = record.get("physics_48d")
    if supplied is not None:
        return np.asarray(supplied, dtype=np.float32).reshape(-1)
    if extractor is None:
        raise ValueError("physics_48d is missing and no physics extractor was supplied")
    return extractor.predict_features(_history(record))


def _snapshot_56d(record: Mapping[str, Any]) -> np.ndarray | None:
    """Return the exact feature vector captured at prediction time, when present."""
    snapshot = record.get("features_56d", record.get("feature_snapshot_56d"))
    if snapshot is None:
        return None
    vector = np.asarray(snapshot, dtype=np.float32).reshape(-1)
    if vector.size != FEATURE_DIM:
        raise ValueError(f"features_56d must contain exactly {FEATURE_DIM} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError("features_56d must contain only finite values")
    return vector.reshape(1, FEATURE_DIM)


def make_training_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    physics_extractor: PhysicsFeatureExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    """Build direct-label training data and retain the matching chronological rows.

    New browser records provide the exact 56D vector that was used at
    prediction-time.  Older records remain supported by rebuilding the feature
    blocks only when no snapshot is available.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    used_records: list[Mapping[str, Any]] = []
    for record in records:
        try:
            x = _snapshot_56d(record)
            if x is None:
                pb = _core_pb(record)
                x = build_56d_feature_matrix(
                    pb,
                    _original_7d(record, pb),
                    _physics_48d(record, physics_extractor),
                )
            y = _actual_b(record)
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(x[0])
        labels.append(y)
        used_records.append(record)
    if not rows:
        raise ValueError("no valid B/P training rows")
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int8),
        used_records,
    )


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    physics_extractor: PhysicsFeatureExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological 56D inputs with absolute binary labels (B=1/P=0)."""
    x, y, _ = make_training_dataset(records, physics_extractor=physics_extractor)
    return x, y


def chronological_validation_mask(row_count: int, *, fraction: float = 0.20) -> np.ndarray:
    """Strict walk-forward split: earlier rows train, final rows validate."""
    if row_count < 2:
        raise ValueError("at least two chronological rows are required")
    fraction = min(0.50, max(0.05, float(fraction)))
    cut = max(1, min(row_count - 1, int(math.floor(row_count * (1.0 - fraction)))))
    mask = np.zeros(row_count, dtype=bool)
    mask[cut:] = True
    return mask


def shoe_level_validation_mask(
    records: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.20,
) -> np.ndarray:
    """Hold out complete latest shoes, preserving the supplied chronological order."""
    if len(records) < 2:
        raise ValueError("at least two chronological rows are required")
    fraction = min(0.50, max(0.05, float(fraction)))
    shoe_keys: list[str] = []
    ordered_shoes: list[str] = []
    seen: set[str] = set()
    for record in records:
        value = record.get("shoe_id")
        shoe_id = str(value).strip() if value is not None else ""
        if not shoe_id:
            raise ValueError("strict shoe-level validation requires a non-empty shoe_id")
        shoe_keys.append(shoe_id)
        if shoe_id not in seen:
            ordered_shoes.append(shoe_id)
            seen.add(shoe_id)
    if len(ordered_shoes) < 2:
        raise ValueError("strict shoe-level validation requires at least two shoes")
    validation_shoe_count = max(
        1,
        min(len(ordered_shoes) - 1, int(math.ceil(len(ordered_shoes) * fraction))),
    )
    validation_shoes = set(ordered_shoes[-validation_shoe_count:])
    return np.asarray([shoe_id in validation_shoes for shoe_id in shoe_keys], dtype=bool)


def _accuracy(probability_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((probability_b > 0.50) == (actual_b > 0)))


def _brier(probability_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((probability_b.astype(float) - actual_b.astype(float)) ** 2))


def evaluate(model: Any, x: np.ndarray, y: np.ndarray, *, probability_bounds: Sequence[float]) -> dict[str, Any]:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", (0, 1)))
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise RuntimeError("classifier has no Banker class")
    raw = np.clip(probabilities[:, int(positive[0])], 0.0, 1.0)
    lo, hi = _probability_bounds(probability_bounds)
    final = np.clip(raw, lo, hi)
    return {
        "samples": float(len(y)),
        "raw_accuracy": _accuracy(raw, y),
        "raw_brier": _brier(raw, y),
        "bounded_accuracy": _accuracy(final, y),
        "bounded_brier": _brier(final, y),
    }


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node: Mapping[str, Any] = tree
    for _ in range(256):
        if "leaf" in node:
            return float(node.get("leaf", 0.0))
        split = str(node.get("split", ""))
        index = int(split[1:]) if split.startswith("f") and split[1:].isdigit() else FEATURE_NAMES.index(split)
        value = float(np.float32(vector[index])) if 0 <= index < len(vector) else math.nan
        threshold = float(np.float32(node.get("split_condition", 0.0)))
        next_id = node.get("missing") if not math.isfinite(value) else (node.get("yes") if value < threshold else node.get("no"))
        node = next((child for child in node.get("children") or [] if int(child.get("nodeid", -1)) == int(next_id)), {})
        if not node:
            return 0.0
    return 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def _logit(probability: float) -> float:
    probability = _clip(probability, 1e-7, 1.0 - 1e-7)
    return math.log(probability / (1.0 - probability))


def export_browser_bundle(
    model: Any,
    reference_x: np.ndarray,
    output_path: str | Path,
    *,
    metrics: Mapping[str, Any],
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Export sigmoid-linked classifier trees for the static browser runtime."""
    lo, hi = _probability_bounds(probability_bounds)
    trees = [json.loads(item) for item in model.get_booster().get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=np.float32)
    tree_sum = sum(_tree_leaf(tree, reference) for tree in trees)
    native_pb = _positive_class_probability(model, reference.reshape(1, -1))
    base_margin = _logit(native_pb) - tree_sum

    for vector in np.asarray(reference_x[: min(128, len(reference_x))], dtype=np.float32):
        portable_pb = _sigmoid(base_margin + sum(_tree_leaf(tree, vector) for tree in trees))
        native_pb = _positive_class_probability(model, vector.reshape(1, -1))
        if abs(portable_pb - native_pb) > 2e-5:
            raise RuntimeError(f"portable probability mismatch {portable_pb} vs {native_pb}")

    bundle = {
        "schema_version": 1,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "link": "sigmoid",
        "base_margin": float(base_margin),
        "probability_bounds": [lo, hi],
        "trees": trees,
        "training": {
            "target": "actual_b_binary",
            "label_mapping": {"P": 0, "B": 1},
            "residual": False,
            "feature_snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "metrics": dict(metrics),
        },
    }
    Path(output_path).write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    extractor = PhysicsFeatureExtractor.load(args.physics_model) if args.physics_model else None
    records = load_training_records(Path(args.input))
    x, y, training_records = make_training_dataset(records, physics_extractor=extractor)
    if len(x) < args.min_samples:
        raise SystemExit(f"need {args.min_samples} rows; got {len(x)}")

    validation = shoe_level_validation_mask(training_records, fraction=args.validation_fraction)
    model = build_xgboost_classifier(random_state=args.random_state)
    model.fit(x[~validation], y[~validation])
    metrics = evaluate(model, x[validation], y[validation], probability_bounds=args.probability_bounds)
    metrics.update({
        "validation_strategy": "chronological_shoe",
        "training_shoes": int(len({str(record["shoe_id"]) for record, held_out in zip(training_records, validation) if not held_out})),
        "validation_shoes": int(len({str(record["shoe_id"]) for record, held_out in zip(training_records, validation) if held_out})),
    })
    print(json.dumps({"validation": metrics}, ensure_ascii=False, indent=2))

    final_model = build_xgboost_classifier(random_state=args.random_state)
    final_model.fit(x, y)
    if args.joblib_output:
        joblib.dump(final_model, args.joblib_output)
    export_browser_bundle(
        final_model,
        x,
        args.output,
        metrics=metrics,
        probability_bounds=args.probability_bounds,
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the direct 56D final-probability XGBoost model")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--physics-model", default="")
    train.add_argument("--output", default="final_probability_model.json")
    train.add_argument("--joblib-output", default="")
    train.add_argument("--min-samples", type=int, default=3000)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--probability-bounds", nargs=2, type=float, default=PROBABILITY_BOUNDS, metavar=("MIN", "MAX"))
    train.add_argument("--random-state", type=int, default=RANDOM_STATE)
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
