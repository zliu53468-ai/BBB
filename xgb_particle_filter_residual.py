#!/usr/bin/env python3
"""BBB dual-brain residual correction: XGBoost + temporal Transformer.

Frozen upstream:
    history -> 256D/V23 Core -> core_p_b -> fixed 7D features

Downstream:
    ShoeParticleFilter -> 3D blind physical forecast
    fixed 7D + physical 3D = 10D

Parallel residual brains:
    delta_xgb   = XGBoost(current 10D)
    delta_trans = Transformer(last 10 x 10D)
    delta_final = 0.5 * delta_xgb + 0.5 * delta_trans

Safety:
    delta_clipped = clip(delta_final, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)

Training is causal: row t physical features are generated before outcome t
updates the particle filter. Transformer windows include current row t plus up
to nine earlier rows from the same shoe, with left zero-padding.
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
    PHYSICAL_FEATURE_NAMES,
    ShoeParticleFilter,
    new_shoe_particle_filter,
)
from transformer_residual import (
    WINDOW_SIZE,
    TemporalResidualTransformer,
    TemporalWindowBuffer,
    TransformerTrainConfig,
    build_sequence_windows,
    export_transformer_payload,
    predict_transformer,
    train_transformer,
    train_transformer_full,
)

UPSTREAM_FEATURE_NAMES: tuple[str, ...] = base.FEATURE_NAMES
MODEL_FEATURE_NAMES: tuple[str, ...] = (
    *UPSTREAM_FEATURE_NAMES,
    *PHYSICAL_FEATURE_NAMES,
)
MODEL_TYPE = "xgb_transformer_dual_residual"
SCHEMA_VERSION = 11
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA
FUSION_XGB_WEIGHT = 0.50
FUSION_TRANSFORMER_WEIGHT = 0.50

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 65,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 2.0,
    "reg_alpha": 0.05,
    "reg_lambda": 0.25,
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
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_optional_physical_observation(
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


def combine_features_10d(
    features_7d: Sequence[float],
    physical_3d: Sequence[float],
) -> np.ndarray:
    x7 = np.asarray(features_7d, dtype=np.float32).reshape(-1)
    x3 = np.asarray(physical_3d, dtype=np.float32).reshape(-1)
    if x7.shape[0] != len(UPSTREAM_FEATURE_NAMES):
        raise ValueError(
            f"features_7d must contain {len(UPSTREAM_FEATURE_NAMES)} values"
        )
    if x3.shape[0] != len(PHYSICAL_FEATURE_NAMES):
        raise ValueError(
            f"physical_3d must contain {len(PHYSICAL_FEATURE_NAMES)} values"
        )
    return np.concatenate([x7, x3]).astype(np.float32, copy=False)


def make_training_arrays_10d(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    filters: dict[str, ShoeParticleFilter] = {}

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

        physical_3d = particle_filter.predict_physical_features(
            core_pb=core_p_b,
            current_round=round_index,
        )
        vector10 = combine_features_10d(vector7, physical_3d)
        residual_target = float(actual_b) - core_p_b

        vectors.append(vector10.astype(float).tolist())
        residuals.append(residual_target)
        actuals.append(actual_b)
        shoes.append(shoe_id)

        observed_total_cards, player_point, banker_point = (
            extract_optional_physical_observation(record)
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


class DualBrainResidualPredictor:
    """PF 10D feature builder + XGBoost/Transformer 50:50 residual fusion."""

    def __init__(
        self,
        xgb_model: XGBRegressor,
        transformer_model: TemporalResidualTransformer,
        particle_filter: ShoeParticleFilter | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.xgb_model = xgb_model
        self.transformer_model = transformer_model
        self.particle_filter = particle_filter or new_shoe_particle_filter()
        self.window_buffer = TemporalWindowBuffer(WINDOW_SIZE)
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)
        self._last_round_index: float | None = None

    def reset_shoe(self) -> None:
        self.particle_filter.reset()
        self.window_buffer.reset()
        self._last_round_index = None

    def update_after_outcome(
        self,
        *,
        actual_b: int | float,
        core_pb: float,
        round_index: float,
        observed_total_cards: int | None = None,
        observed_player_point: int | None = None,
        observed_banker_point: int | None = None,
    ) -> None:
        self.particle_filter.observe_and_project(
            real_outcome=float(actual_b),
            core_pb=float(core_pb),
            current_round=float(round_index),
            observed_total_cards=observed_total_cards,
            observed_player_point=observed_player_point,
            observed_banker_point=observed_banker_point,
        )

    def _current_10d(
        self,
        features_7d: Sequence[float],
        *,
        core_pb: float,
        round_index: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        physical_3d = self.particle_filter.predict_physical_features(
            core_pb=core_pb,
            current_round=round_index,
        )
        x10 = combine_features_10d(features_7d, physical_3d)

        if self._last_round_index == float(round_index) and self.window_buffer.rows:
            self.window_buffer.rows[-1] = x10.copy()
        else:
            self.window_buffer.push(x10)
            self._last_round_index = float(round_index)

        return x10, physical_3d

    def predict_delta(
        self,
        features_7d: Sequence[float],
        *,
        core_pb: float,
        round_index: float,
    ) -> dict[str, Any]:
        x10, physical_3d = self._current_10d(
            features_7d,
            core_pb=core_pb,
            round_index=round_index,
        )
        delta_xgb = float(self.xgb_model.predict(x10.reshape(1, 10))[0])

        window, valid_mask = self.window_buffer.as_arrays()
        delta_trans = float(
            predict_transformer(
                self.transformer_model,
                window,
                valid_mask,
            )[0]
        )

        delta_final = (
            FUSION_XGB_WEIGHT * delta_xgb
            + FUSION_TRANSFORMER_WEIGHT * delta_trans
        )
        delta_clipped = float(
            np.clip(delta_final, -self.max_delta, self.max_delta)
        )
        return {
            "physical_prediction": {
                name: float(value)
                for name, value in zip(PHYSICAL_FEATURE_NAMES, physical_3d)
            },
            "delta_xgb": delta_xgb,
            "delta_transformer": delta_trans,
            "delta_final": delta_final,
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
        round_index = float(
            x7[UPSTREAM_FEATURE_NAMES.index("round_index")]
        )
        prediction = self.predict_delta(
            x7,
            core_pb=core_value,
            round_index=round_index,
        )
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


def _tree_leaf_10d(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
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
    tree_sum = sum(_tree_leaf_10d(tree, reference) for tree in trees)
    native_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = native_reference - tree_sum

    for vector in np.asarray(
        reference_x[: min(64, len(reference_x))],
        dtype=float,
    ):
        portable = base_score + sum(_tree_leaf_10d(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(
                f"portable 10D export mismatch: {portable} vs {native}"
            )

    return {
        "base_score": float(base_score),
        "trees": trees,
        "params": {
            "n_estimators": 65,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 2.0,
            "alpha": 0.05,
            "lambda": 0.25,
            "random_state": 42,
        },
    }


def evaluate_dual(
    xgb_model: XGBRegressor,
    transformer_model: TemporalResidualTransformer,
    x10: np.ndarray,
    windows: np.ndarray,
    valid_masks: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    delta_xgb = np.asarray(xgb_model.predict(x10), dtype=float)
    delta_trans = np.asarray(
        predict_transformer(transformer_model, windows, valid_masks),
        dtype=float,
    )
    delta_fused_raw = (
        FUSION_XGB_WEIGHT * delta_xgb
        + FUSION_TRANSFORMER_WEIGHT * delta_trans
    )

    core_pb = x10[:, MODEL_FEATURE_NAMES.index("core_p_b")].astype(float)
    xgb_pb = np.clip(core_pb + np.clip(delta_xgb, -max_delta, max_delta), 0, 1)
    trans_pb = np.clip(core_pb + np.clip(delta_trans, -max_delta, max_delta), 0, 1)
    fused_delta = np.clip(delta_fused_raw, -max_delta, max_delta)
    fused_pb = np.clip(core_pb + fused_delta, 0, 1)

    return {
        "samples": float(len(x10)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "xgb_accuracy": base.direction_accuracy(xgb_pb, actual_b),
        "transformer_accuracy": base.direction_accuracy(trans_pb, actual_b),
        "fused_accuracy": base.direction_accuracy(fused_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "xgb_brier": base.brier(xgb_pb, actual_b),
        "transformer_brier": base.brier(trans_pb, actual_b),
        "fused_brier": base.brier(fused_pb, actual_b),
        "mean_abs_delta_xgb": float(np.mean(np.abs(delta_xgb))),
        "mean_abs_delta_transformer": float(np.mean(np.abs(delta_trans))),
        "mean_abs_delta_fused": float(np.mean(np.abs(fused_delta))),
        "max_abs_delta_fused": (
            float(np.max(np.abs(fused_delta))) if len(fused_delta) else 0.0
        ),
    }


def export_portable_bundle(
    xgb_model: XGBRegressor,
    transformer_model: TemporalResidualTransformer,
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
        "physical_feature_names": list(PHYSICAL_FEATURE_NAMES),
        "feature_schema": "7D_PLUS_3D_BLIND_PHYSICAL",
        "window_size": WINDOW_SIZE,
        "max_delta": float(max_delta),
        "fusion": {
            "xgb_weight": FUSION_XGB_WEIGHT,
            "transformer_weight": FUSION_TRANSFORMER_WEIGHT,
            "formula": "(delta_xgb + delta_transformer) / 2",
        },
        "xgb": _portable_xgb_payload(xgb_model, reference_x),
        "transformer": export_transformer_payload(transformer_model),
        "shoe_particle_filter": dict(PF_CONFIG),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "physical_feature_timing": "forecast_before_current_outcome",
            "transformer_window": "current_10D_plus_previous_9_same_shoe_left_zero_padding",
            "blind_mode": "B/P + current_round + core_pb; card count/points optional",
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
    x10, residual, actual_b, shoes = make_training_arrays_10d(records)

    if len(x10) < args.min_samples:
        raise SystemExit(
            f"need at least {args.min_samples} valid rows; got {len(x10)}"
        )

    windows, valid_masks = build_sequence_windows(
        x10,
        shoes,
        window_size=WINDOW_SIZE,
    )
    validation = base.deterministic_validation_mask(
        shoes,
        fraction=args.validation_fraction,
    )
    train = ~validation

    xgb_model = build_xgb_regressor()
    xgb_model.fit(x10[train], residual[train])

    transformer_cfg = TransformerTrainConfig(
        epochs=args.transformer_epochs,
        batch_size=args.transformer_batch_size,
        learning_rate=args.transformer_learning_rate,
        weight_decay=args.transformer_weight_decay,
        patience=args.transformer_patience,
    )
    transformer_model = train_transformer(
        windows,
        valid_masks,
        residual,
        train,
        validation,
        config=transformer_cfg,
    )

    validation_metrics = evaluate_dual(
        xgb_model,
        transformer_model,
        x10[validation],
        windows[validation],
        valid_masks[validation],
        actual_b[validation],
        max_delta=args.max_delta,
    )

    accepted = (
        validation_metrics["fused_brier"]
        <= validation_metrics["core_brier"] + args.max_brier_regression
        and validation_metrics["fused_accuracy"]
        >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    )

    print(
        json.dumps(
            {
                "validation": validation_metrics,
                "transformer_best_epoch": int(
                    getattr(transformer_model, "best_epoch", 1)
                ),
                "accepted": accepted,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not accepted and not args.force:
        raise SystemExit(
            "validation gate rejected dual-brain model; "
            "use --force only for diagnostics"
        )

    final_xgb = build_xgb_regressor()
    final_xgb.fit(x10, residual)

    best_epoch = int(getattr(transformer_model, "best_epoch", 1))
    final_transformer = train_transformer_full(
        windows,
        valid_masks,
        residual,
        epochs=best_epoch,
        config=transformer_cfg,
    )

    export_portable_bundle(
        final_xgb,
        final_transformer,
        reference_x=x10,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x10),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BBB PF 10D XGBoost + Transformer dual-brain trainer"
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
    train.add_argument("--transformer-epochs", type=int, default=120)
    train.add_argument("--transformer-batch-size", type=int, default=64)
    train.add_argument("--transformer-learning-rate", type=float, default=1e-3)
    train.add_argument("--transformer-weight-decay", type=float, default=1e-4)
    train.add_argument("--transformer-patience", type=int, default=15)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
