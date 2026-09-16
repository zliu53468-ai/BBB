#!/usr/bin/env python3
"""External XGBoost residual-bias model for the BBB 256D/V23 core.

V1.1 keeps the architecture intentionally narrow:

    Frozen 256D/V23 core -> 7 existing features -> XGBRegressor residual -> scaled delta -> B/P

The regressor never predicts a raw B/P class. It learns:

    residual = actual_B - core_p_B

Production logic remains decisive (no PASS):

    scaled_delta = clip(xgb_residual * delta_scale, -max_delta, +max_delta)
    final_p_B = clip(core_p_B + scaled_delta, 0, 1)
    direction = B if final_p_B > 0.50 else P

V1.1 improvements:
- deterministic grouped 5-fold out-of-fold validation by shoe_id
- validation-selected residual delta scale
- flip/rescue/damage diagnostics
- full training-row de-duplication by shoe_id + history_fingerprint
- portable model bundle bound to the exact frozen core version
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
MODEL_VERSION = "XGB_RESIDUAL_BIAS_V1_1"
BASE_CORE_VERSION = "V23_SHORT_X_DYNAMIC_HAZARD_R1"
SCHEMA_VERSION = 2
DEFAULT_MAX_DELTA = 0.10
DEFAULT_RANDOM_STATE = 20260916
DEFAULT_FOLDS = 5
DEFAULT_DELTA_SCALES: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)


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


def build_features(
    *,
    core_p_b: float,
    history: str | Sequence[str],
    estimated_total_hands: float = 60.0,
    stage: float | None = None,
    depth: float | None = None,
) -> ResidualFeatures:
    seq = normalize_history(history)
    total_hands = clip(float(estimated_total_hands), 40.0, 90.0)
    round_index = float(max(1, min(70, len(seq) + 1)))
    remaining_ratio = clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b), 0.0, 1.0),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        sx_markov_p_same=sx_markov_p_same(seq),
        stage=float(current_stage(seq) if stage is None else stage),
        depth=float(current_depth(seq) if depth is None else depth),
    )


def build_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    # V1.1 removes row/feature sampling. With only seven features this makes the
    # residual layer easier to reproduce and audit without changing model class.
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
    """Reusable Python API around the residual XGBRegressor."""

    def __init__(
        self,
        model: XGBRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
        delta_scale: float = 1.0,
    ) -> None:
        self.model = model or build_regressor()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)
        self.delta_scale = clip(float(delta_scale), 0.0, 1.0)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "ResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        residual = y - core_pb
        self.model.fit(x, residual)
        return self

    def predict_delta(self, feature_row: Sequence[float]) -> float:
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        raw = float(self.model.predict(x)[0])
        return clip(raw * self.delta_scale, -self.max_delta, self.max_delta)

    def correct(self, feature_row: Sequence[float]) -> dict[str, Any]:
        x = np.asarray(feature_row, dtype=np.float32)
        core_pb = clip(float(x[FEATURE_NAMES.index("core_p_b")]))
        delta = self.predict_delta(x)
        final_pb = clip(core_pb + delta)
        return {
            "core_p_b": core_pb,
            "delta_scale": self.delta_scale,
            "delta": delta,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.5 else "P",
        }


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
    features = build_features(
        core_p_b=float(record.get("core_p_b", record.get("core_pb", 0.5))),
        history=record.get("history") or record.get("history_fingerprint") or "",
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    )
    return features.as_dict()


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


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    seen: set[tuple[str, str]] = set()
    stats = {"input_rows": len(records), "duplicates_removed": 0, "invalid_rows": 0, "wrong_core_rows": 0}

    for idx, record in enumerate(records):
        shoe_id = str(record.get("shoe_id") or f"row_{idx}")
        fingerprint = str(record.get("history_fingerprint") or record.get("history") or f"row_{idx}")
        dedupe_key = (shoe_id, fingerprint)
        if dedupe_key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(dedupe_key)

        row_core_version = str(record.get("core_version") or BASE_CORE_VERSION)
        if row_core_version != BASE_CORE_VERSION:
            stats["wrong_core_rows"] += 1
            continue

        try:
            actual_b = _parse_actual_b(record)
            row = _feature_row(record)
            vector = [float(row[name]) for name in FEATURE_NAMES]
            if not all(math.isfinite(x) for x in vector):
                stats["invalid_rows"] += 1
                continue
            core_p_b = clip(row["core_p_b"])
        except (TypeError, ValueError, KeyError):
            stats["invalid_rows"] += 1
            continue

        vectors.append(vector)
        residuals.append(float(actual_b) - core_p_b)
        actuals.append(actual_b)
        shoes.append(shoe_id)

    if not vectors:
        raise ValueError("no valid B/P training rows")
    stats["valid_rows"] = len(vectors)
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        shoes,
        stats,
    )


def _shoe_fold(shoe_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(shoe_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def grouped_fold_ids(shoes: Sequence[str], *, requested_folds: int = DEFAULT_FOLDS) -> tuple[np.ndarray, int]:
    unique_shoes = sorted(set(map(str, shoes)))
    folds = min(max(2, int(requested_folds)), len(unique_shoes))
    if folds < 2:
        raise ValueError("need at least two distinct shoe_id values for grouped validation")
    ids = np.asarray([_shoe_fold(str(shoe), folds) for shoe in shoes], dtype=np.int16)

    # Hashing can theoretically leave a fold empty on very small datasets.
    # Reassign distinct shoes deterministically until every fold is represented.
    present = set(map(int, np.unique(ids)))
    if len(present) < folds:
        mapping = {shoe: i % folds for i, shoe in enumerate(unique_shoes)}
        ids = np.asarray([mapping[str(shoe)] for shoe in shoes], dtype=np.int16)
    return ids, folds


def direction_accuracy(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b > 0.5) == (actual_b > 0)))


def brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def evaluate_from_raw_delta(
    raw_delta: np.ndarray,
    x: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
    delta_scale: float,
) -> dict[str, float | int]:
    raw_delta = np.asarray(raw_delta, dtype=float)
    delta = np.clip(raw_delta * float(delta_scale), -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)
    core_correct = (core_pb > 0.5) == (actual_b > 0)
    final_correct = (final_pb > 0.5) == (actual_b > 0)
    flipped = (core_pb > 0.5) != (final_pb > 0.5)
    flip_count = int(np.sum(flipped))
    rescue = int(np.sum((~core_correct) & final_correct))
    damage = int(np.sum(core_correct & (~final_correct)))
    return {
        "samples": int(len(x)),
        "core_accuracy": direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": direction_accuracy(final_pb, actual_b),
        "accuracy_gain": direction_accuracy(final_pb, actual_b) - direction_accuracy(core_pb, actual_b),
        "core_brier": brier(core_pb, actual_b),
        "corrected_brier": brier(final_pb, actual_b),
        "brier_gain": brier(core_pb, actual_b) - brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "delta_scale": float(delta_scale),
        "flip_count": flip_count,
        "flip_rate": float(flip_count / len(x)) if len(x) else 0.0,
        "flip_win_rate": float(np.mean(final_correct[flipped])) if flip_count else 0.0,
        "rescue_count": rescue,
        "damage_count": damage,
        "net_flip_gain": rescue - damage,
    }


def choose_delta_scale(
    raw_delta: np.ndarray,
    x: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
    scales: Sequence[float] = DEFAULT_DELTA_SCALES,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for scale in scales:
        metrics = evaluate_from_raw_delta(
            raw_delta,
            x,
            actual_b,
            max_delta=max_delta,
            delta_scale=clip(float(scale), 0.0, 1.0),
        )
        candidates.append(metrics)

    # Primary objective: lowest OOF Brier. Secondary: highest OOF accuracy.
    # Tertiary: smaller scale, preferring conservative correction when tied.
    best = min(
        candidates,
        key=lambda m: (
            round(float(m["corrected_brier"]), 12),
            -round(float(m["corrected_accuracy"]), 12),
            float(m["delta_scale"]),
        ),
    )
    return float(best["delta_scale"]), best, candidates


def grouped_oof_predictions(
    x: np.ndarray,
    residual: np.ndarray,
    shoes: Sequence[str],
    *,
    requested_folds: int,
    random_state: int,
) -> tuple[np.ndarray, int, list[dict[str, int]]]:
    fold_ids, folds = grouped_fold_ids(shoes, requested_folds=requested_folds)
    raw_oof = np.full(len(x), np.nan, dtype=float)
    fold_stats: list[dict[str, int]] = []

    for fold in range(folds):
        validation = fold_ids == fold
        train = ~validation
        if not np.any(validation) or not np.any(train):
            raise RuntimeError(f"invalid grouped fold {fold}: train={int(np.sum(train))}, validation={int(np.sum(validation))}")
        model = build_regressor(random_state=random_state)
        model.fit(x[train], residual[train])
        raw_oof[validation] = np.asarray(model.predict(x[validation]), dtype=float)
        fold_stats.append({
            "fold": int(fold),
            "train_rows": int(np.sum(train)),
            "validation_rows": int(np.sum(validation)),
            "validation_shoes": int(len(set(np.asarray(shoes, dtype=object)[validation].tolist()))),
        })

    if not np.all(np.isfinite(raw_oof)):
        raise RuntimeError("OOF prediction generation left non-finite rows")
    return raw_oof, folds, fold_stats


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
        next_id = node.get("missing") if not math.isfinite(value) else (
            node.get("yes") if value < split_condition else node.get("no")
        )
        children = node.get("children") or []
        found = next((child for child in children if int(child.get("nodeid", -999)) == int(next_id)), None)
        if found is None:
            return 0.0
        node = found
    return 0.0


def export_portable_bundle(
    model: XGBRegressor,
    *,
    reference_x: np.ndarray,
    output_path: Path,
    max_delta: float,
    delta_scale: float,
    metrics: Mapping[str, Any],
    training_rows: int,
    validation_folds: int,
    data_stats: Mapping[str, Any],
) -> dict[str, Any]:
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

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "base_core_version": BASE_CORE_VERSION,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "base_score": float(base_score),
        "max_delta": float(max_delta),
        "delta_scale": float(delta_scale),
        "trees": trees,
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "no_pass": True,
            "validation": f"grouped_{validation_folds}_fold_out_of_fold_by_shoe_id",
            "metrics": dict(metrics),
            "data_stats": dict(data_stats),
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    x, residual, actual_b, shoes, data_stats = make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")
    unique_shoes = len(set(shoes))
    if unique_shoes < 2:
        raise SystemExit("need at least two distinct shoe_id values for out-of-shoe validation")

    raw_oof, folds, fold_stats = grouped_oof_predictions(
        x,
        residual,
        shoes,
        requested_folds=args.folds,
        random_state=args.random_state,
    )
    selected_scale, validation_metrics, scale_candidates = choose_delta_scale(
        raw_oof,
        x,
        actual_b,
        max_delta=args.max_delta,
        scales=args.delta_scales,
    )
    validation_metrics = dict(validation_metrics)
    validation_metrics["folds"] = folds
    validation_metrics["unique_shoes"] = unique_shoes
    validation_metrics["fold_stats"] = fold_stats
    validation_metrics["scale_candidates"] = scale_candidates

    accepted = (
        float(validation_metrics["corrected_brier"]) <= float(validation_metrics["core_brier"]) + args.max_brier_regression
        and float(validation_metrics["corrected_accuracy"]) >= float(validation_metrics["core_accuracy"]) - args.max_accuracy_regression
        and int(validation_metrics["net_flip_gain"]) >= args.min_net_flip_gain
    )
    report = {
        "model_version": MODEL_VERSION,
        "base_core_version": BASE_CORE_VERSION,
        "validation": validation_metrics,
        "data_stats": data_stats,
        "selected_delta_scale": selected_scale,
        "accepted": accepted,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected residual model; use --force only for diagnostics")

    final_model = build_regressor(random_state=args.random_state)
    final_model.fit(x, residual)
    export_portable_bundle(
        final_model,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        delta_scale=selected_scale,
        metrics=validation_metrics,
        training_rows=len(x),
        validation_folds=folds,
        data_stats=data_stats,
    )
    print(f"wrote {args.output}")
    return 0


def _parse_scales(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = clip(float(part), 0.0, 1.0)
        if value > 0 and value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("delta scales must contain at least one value in (0, 1]")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost residual bias trainer V1.1")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train XGBRegressor and export browser model JSON")
    train.add_argument("--input", required=True, help="browser-exported JSON or CSV")
    train.add_argument("--output", default="residual_bias_model.json")
    train.add_argument("--min-samples", type=int, default=500)
    train.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA)
    train.add_argument("--delta-scales", type=_parse_scales, default=DEFAULT_DELTA_SCALES)
    train.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train.add_argument("--max-brier-regression", type=float, default=0.0)
    train.add_argument("--max-accuracy-regression", type=float, default=0.0)
    train.add_argument("--min-net-flip-gain", type=int, default=0)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
