#!/usr/bin/env python3
"""Enhanced PF 4D tensor -> 11D XGBoost residual correction for BBB.

Frozen upstream:
    history -> 256D/V23 Core -> core_p_b -> fixed 7D features

Downstream:
    EnhancedShoeParticleFilter -> pseudo_card_feature[4]
    features_11d = fixed_7d + [p_4cards, p_6cards, win_point, lose_point]
    delta = XGBoost(features_11d)
    delta_clipped = clip(delta, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)

Training is causal: the four PF features for row t are captured before outcome t
updates the particle population.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from xgboost import XGBRegressor

import xgb_residual_bias as base
from shoe_particle_filter import (
    PF_CONFIG,
    PSEUDO_CARD_FEATURE_NAMES,
    EnhancedShoeParticleFilter,
    new_shoe_particle_filter,
)

UPSTREAM_FEATURE_NAMES: tuple[str, ...] = base.FEATURE_NAMES
MODEL_FEATURE_NAMES: tuple[str, ...] = (
    *UPSTREAM_FEATURE_NAMES,
    *PSEUDO_CARD_FEATURE_NAMES,
)
MODEL_TYPE = "xgb_enhanced_pf_tensor_residual"
SCHEMA_VERSION = 9
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 75,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 2.0,
    "reg_alpha": 0.05,
    "reg_lambda": 0.30,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMS)


def _optional_int(record: Mapping[str, Any], *names: str) -> int | None:
    physical = record.get("physical_observation")
    for name in names:
        value = record.get(name)
        if value is None and isinstance(physical, Mapping):
            value = physical.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return parsed
    return None


def extract_physical_observation(
    record: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None]:
    total_cards = _optional_int(
        record,
        "observed_total_cards",
        "total_cards",
        "round_total_cards",
    )
    player_point = _optional_int(
        record,
        "observed_player_point",
        "player_point",
        "player_points",
    )
    banker_point = _optional_int(
        record,
        "observed_banker_point",
        "banker_point",
        "banker_points",
    )

    if total_cards not in (4, 5, 6):
        total_cards = None
    if player_point is not None and not (0 <= player_point <= 9):
        player_point = None
    if banker_point is not None and not (0 <= banker_point <= 9):
        banker_point = None
    return total_cards, player_point, banker_point


def combine_features_11d(
    features_7d: Sequence[float],
    pseudo_card_feature: Sequence[float],
) -> np.ndarray:
    x7 = np.asarray(features_7d, dtype=np.float32).reshape(-1)
    tensor = np.asarray(pseudo_card_feature, dtype=np.float32).reshape(-1)

    if x7.shape[0] != len(UPSTREAM_FEATURE_NAMES):
        raise ValueError(
            f"features_7d must contain {len(UPSTREAM_FEATURE_NAMES)} values"
        )
    if tensor.shape[0] != len(PSEUDO_CARD_FEATURE_NAMES):
        raise ValueError(
            f"pseudo_card_feature must contain {len(PSEUDO_CARD_FEATURE_NAMES)} values"
        )

    return np.concatenate([x7, tensor]).astype(np.float32, copy=False)


def make_training_arrays_11d(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    filters: dict[str, EnhancedShoeParticleFilter] = {}

    for idx, record in enumerate(records):
        try:
            actual_b = base._parse_actual_b(record)
            row = base._feature_row(record)
            vector7 = [float(row[name]) for name in UPSTREAM_FEATURE_NAMES]
            if not all(math.isfinite(value) for value in vector7):
                continue
            core_p_b = base.clip(float(row["core_p_b"]))
            round_index = float(row["round_index"])
        except (TypeError, ValueError, KeyError):
            continue

        shoe_id = str(record.get("shoe_id") or f"row_{idx}")
        particle_filter = filters.get(shoe_id)
        if particle_filter is None:
            particle_filter = new_shoe_particle_filter()
            filters[shoe_id] = particle_filter

        # Capture prior-to-label physical inference for the current row.
        tensor = particle_filter.current_pseudo_card_feature()
        residual_target = float(actual_b) - core_p_b

        vector11 = combine_features_11d(vector7, tensor)
        vectors.append(vector11.astype(float).tolist())
        residuals.append(residual_target)
        actuals.append(actual_b)
        shoes.append(shoe_id)

        observed_total_cards, player_point, banker_point = extract_physical_observation(
            record
        )
        particle_filter.observe_and_project(
            real_outcome=actual_b,
            core_pb=core_p_b,
            current_round=round_index,
            observed_total_cards=observed_total_cards,
            observed_player_point=player_point,
            observed_banker_point=banker_point,
        )

    if not vectors:
        raise ValueError("no valid B/P training rows")

    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        shoes,
    )


class EnhancedPFTensorXGBResidualPredictor:
    """4D physical tensor + frozen 7D -> 11D XGBoost residual model."""

    def __init__(
        self,
        xgb_model: XGBRegressor | None = None,
        particle_filter: EnhancedShoeParticleFilter | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.xgb_model = xgb_model or build_xgb_regressor()
        self.particle_filter = particle_filter or new_shoe_particle_filter()
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(
        self,
        feature_rows_11d: np.ndarray,
        residual_targets: Sequence[float],
    ) -> "EnhancedPFTensorXGBResidualPredictor":
        x = np.asarray(feature_rows_11d, dtype=np.float32)
        y = np.asarray(residual_targets, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(MODEL_FEATURE_NAMES):
            raise ValueError(
                f"feature_rows_11d must be N x {len(MODEL_FEATURE_NAMES)}"
            )
        if len(x) != len(y):
            raise ValueError("feature_rows_11d and residual_targets must align")
        self.xgb_model.fit(x, y)
        return self

    def reset_shoe(self) -> None:
        self.particle_filter.reset()

    def current_pseudo_card_feature(self) -> np.ndarray:
        return self.particle_filter.current_pseudo_card_feature()

    def update_after_outcome(
        self,
        *,
        actual_b: int | float,
        core_pb: float,
        round_index: float,
        observed_total_cards: int | None = None,
        observed_player_point: int | None = None,
        observed_banker_point: int | None = None,
    ) -> np.ndarray:
        return self.particle_filter.observe_and_project(
            real_outcome=float(actual_b),
            core_pb=float(core_pb),
            current_round=float(round_index),
            observed_total_cards=observed_total_cards,
            observed_player_point=observed_player_point,
            observed_banker_point=observed_banker_point,
        )

    def predict_delta(self, features_7d: Sequence[float]) -> dict[str, Any]:
        tensor = self.current_pseudo_card_feature()
        x11 = combine_features_11d(features_7d, tensor).reshape(1, 11)
        raw_delta = float(self.xgb_model.predict(x11)[0])
        delta_clipped = float(
            np.clip(raw_delta, -self.max_delta, self.max_delta)
        )
        return {
            "pseudo_card_feature": {
                name: float(value)
                for name, value in zip(PSEUDO_CARD_FEATURE_NAMES, tensor)
            },
            "delta_raw": raw_delta,
            "delta_clipped": delta_clipped,
        }

    def correct(
        self,
        features_7d: Sequence[float],
        core_pb: float | None = None,
    ) -> dict[str, Any]:
        x7 = np.asarray(features_7d, dtype=np.float32).reshape(-1)
        if x7.shape[0] != len(UPSTREAM_FEATURE_NAMES):
            raise ValueError(
                f"features_7d must contain {len(UPSTREAM_FEATURE_NAMES)} values"
            )

        core_value = (
            base.clip(float(core_pb), 0.0, 1.0)
            if core_pb is not None
            else base.clip(
                float(x7[UPSTREAM_FEATURE_NAMES.index("core_p_b")]),
                0.0,
                1.0,
            )
        )
        prediction = self.predict_delta(x7)
        final_pb = base.clip(
            core_value + prediction["delta_clipped"],
            0.0,
            1.0,
        )
        return {
            "core_p_b": core_value,
            **prediction,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.50 else "P",
        }


def _tree_leaf_11d(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
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
                index = MODEL_FEATURE_NAMES.index(split)
            except ValueError:
                index = -1

        value = (
            float(np.float32(vector[index]))
            if 0 <= index < len(vector)
            else math.nan
        )
        split_condition = float(np.float32(node.get("split_condition", 0.0)))
        next_id = (
            node.get("missing")
            if not math.isfinite(value)
            else node.get("yes") if value < split_condition else node.get("no")
        )
        children = node.get("children") or []
        found = next(
            (
                child
                for child in children
                if int(child.get("nodeid", -999)) == int(next_id)
            ),
            None,
        )
        if found is None:
            return 0.0
        node = found

    return 0.0


def _portable_xgb_payload(
    model: XGBRegressor,
    reference_x: np.ndarray,
) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(text) for text in booster.get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf_11d(tree, reference) for tree in trees)
    native_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = native_reference - tree_sum

    for vector in np.asarray(
        reference_x[: min(64, len(reference_x))],
        dtype=float,
    ):
        portable = base_score + sum(_tree_leaf_11d(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(
                f"portable 11D export mismatch: {portable} vs {native}"
            )

    return {
        "base_score": float(base_score),
        "trees": trees,
        "params": {
            "n_estimators": 75,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 2.0,
            "alpha": 0.05,
            "lambda": 0.30,
            "random_state": 42,
        },
    }


def evaluate_11d(
    model: XGBRegressor,
    x11: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    raw_delta = np.asarray(model.predict(x11), dtype=float)
    delta = np.clip(raw_delta, -max_delta, max_delta)
    core_pb = x11[:, MODEL_FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)

    return {
        "samples": float(len(x11)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": base.direction_accuracy(final_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "corrected_brier": base.brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "mean_p_4cards": float(np.mean(x11[:, MODEL_FEATURE_NAMES.index("p_4cards")])),
        "mean_p_6cards": float(np.mean(x11[:, MODEL_FEATURE_NAMES.index("p_6cards")])),
        "mean_win_point": float(np.mean(x11[:, MODEL_FEATURE_NAMES.index("win_point")])),
        "mean_lose_point": float(np.mean(x11[:, MODEL_FEATURE_NAMES.index("lose_point")])),
    }


def export_portable_bundle(
    model: XGBRegressor,
    *,
    reference_x: np.ndarray,
    output_path: Path,
    max_delta: float,
    metrics: Mapping[str, Any],
    training_rows: int,
) -> dict[str, Any]:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(UPSTREAM_FEATURE_NAMES),
        "model_feature_names": list(MODEL_FEATURE_NAMES),
        "pseudo_card_feature_names": list(PSEUDO_CARD_FEATURE_NAMES),
        "feature_schema": "7D_PLUS_4D_PSEUDO_CARD_TENSOR",
        "max_delta": float(max_delta),
        "xgb": _portable_xgb_payload(model, reference_x),
        "enhanced_shoe_particle_filter": dict(PF_CONFIG),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "tensor_timing": "state_before_current_outcome",
            "high_information_rounds": "observed total cards 5 or 6 => likelihood precision x1.5",
            "physical_constraint": "remaining cards constrained around 416 - 4.8*current_round",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = base.load_training_records(Path(args.input))
    x11, residual, actual_b, shoes = make_training_arrays_11d(records)

    if len(x11) < args.min_samples:
        raise SystemExit(
            f"need at least {args.min_samples} valid rows; got {len(x11)}"
        )

    validation = base.deterministic_validation_mask(
        shoes,
        fraction=args.validation_fraction,
    )
    train = ~validation

    model = build_xgb_regressor()
    model.fit(x11[train], residual[train])
    validation_metrics = evaluate_11d(
        model,
        x11[validation],
        actual_b[validation],
        max_delta=args.max_delta,
    )

    accepted = (
        validation_metrics["corrected_brier"]
        <= validation_metrics["core_brier"] + args.max_brier_regression
        and validation_metrics["corrected_accuracy"]
        >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    )

    print(
        json.dumps(
            {"validation": validation_metrics, "accepted": accepted},
            ensure_ascii=False,
            indent=2,
        )
    )

    if not accepted and not args.force:
        raise SystemExit(
            "validation gate rejected enhanced-PF 11D XGBoost model; "
            "use --force only for diagnostics"
        )

    final_model = build_xgb_regressor()
    final_model.fit(x11, residual)
    export_portable_bundle(
        final_model,
        reference_x=x11,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x11),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BBB enhanced PF 4D tensor -> 11D XGBoost residual trainer"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="residual_bias_model.json")
    train.add_argument("--min-samples", type=int, default=500)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA)
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
