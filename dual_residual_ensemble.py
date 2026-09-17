#!/usr/bin/env python3
"""XGBoost + LightGBM residual ensemble for BBB.

The upstream pipeline is frozen:
    history -> 256D/V23 core -> core_p_b -> fixed 7D features

Only the residual layer is trained here. Both models consume the exact same
7D matrix and the exact same residual target:

    residual = actual_B - core_p_B

Inference:
    delta_xgb = XGB(features_7d)
    delta_lgb = LGBM(features_7d)
    delta_final = (delta_xgb + delta_lgb) / 2
    delta_clipped = clip(delta_final, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import xgb_residual_bias as base

FEATURE_NAMES = base.FEATURE_NAMES
MODEL_TYPE = "xgb_lgb_residual_ensemble"
SCHEMA_VERSION = 2
FUSION_METHOD = "arithmetic_mean"
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA
DEFAULT_RANDOM_STATE = 42

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 50,
    "learning_rate": 0.025,
    "max_depth": 3,
    "min_child_weight": 2.5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}

LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 50,
    "learning_rate": 0.025,
    "max_depth": 3,
    "num_leaves": 6,
    "min_data_in_leaf": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 1,
}


def build_xgb_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    params = dict(XGB_PARAMS)
    params["random_state"] = int(random_state)
    return XGBRegressor(**params)


def build_lgb_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> LGBMRegressor:
    params = dict(LGB_PARAMS)
    params["random_state"] = int(random_state)
    return LGBMRegressor(**params)


class DualResidualBiasPredictor:
    """Reusable 50/50 XGBoost + LightGBM residual ensemble."""

    def __init__(
        self,
        xgb_model: XGBRegressor | None = None,
        lgb_model: LGBMRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.xgb_model = xgb_model or build_xgb_regressor(random_state=random_state)
        self.lgb_model = lgb_model or build_lgb_regressor(random_state=random_state)
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "DualResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"feature_rows must be N x {len(FEATURE_NAMES)}")

        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        residual = y - core_pb

        # Exact same 7D rows and exact same residual target for both models.
        self.xgb_model.fit(x, residual)
        self.lgb_model.fit(x, residual)
        return self

    def predict_components(self, features_7d: Sequence[float]) -> dict[str, float]:
        x = np.asarray(features_7d, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(FEATURE_NAMES)} values")

        delta_xgb = float(self.xgb_model.predict(x)[0])
        delta_lgb = float(self.lgb_model.predict(x)[0])
        delta_final = (delta_xgb + delta_lgb) / 2.0
        delta_clipped = float(np.clip(delta_final, -self.max_delta, self.max_delta))
        return {
            "delta_xgb": delta_xgb,
            "delta_lgb": delta_lgb,
            "delta_final": delta_final,
            "delta_clipped": delta_clipped,
        }

    def correct(self, features_7d: Sequence[float]) -> dict[str, Any]:
        x = np.asarray(features_7d, dtype=np.float32)
        if x.shape[0] != len(FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(FEATURE_NAMES)} values")

        core_pb = base.clip(float(x[FEATURE_NAMES.index("core_p_b")]), 0.0, 1.0)
        parts = self.predict_components(x)
        final_pb = base.clip(core_pb + parts["delta_clipped"], 0.0, 1.0)
        return {
            "core_p_b": core_pb,
            **parts,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.50 else "P",
        }


def _corrected_metrics(
    x: np.ndarray,
    actual_b: np.ndarray,
    raw_delta: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    delta = np.clip(np.asarray(raw_delta, dtype=float), -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)
    return {
        "accuracy": base.direction_accuracy(final_pb, actual_b),
        "brier": base.brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
    }


def evaluate_ensemble(
    xgb_model: XGBRegressor,
    lgb_model: LGBMRegressor,
    x: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    raw_xgb = np.asarray(xgb_model.predict(x), dtype=float)
    raw_lgb = np.asarray(lgb_model.predict(x), dtype=float)
    raw_final = (raw_xgb + raw_lgb) / 2.0

    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    xgb_metrics = _corrected_metrics(x, actual_b, raw_xgb, max_delta=max_delta)
    lgb_metrics = _corrected_metrics(x, actual_b, raw_lgb, max_delta=max_delta)
    fused_metrics = _corrected_metrics(x, actual_b, raw_final, max_delta=max_delta)

    return {
        "samples": float(len(x)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "xgb_corrected_accuracy": xgb_metrics["accuracy"],
        "xgb_corrected_brier": xgb_metrics["brier"],
        "lgb_corrected_accuracy": lgb_metrics["accuracy"],
        "lgb_corrected_brier": lgb_metrics["brier"],
        "corrected_accuracy": fused_metrics["accuracy"],
        "corrected_brier": fused_metrics["brier"],
        "mean_abs_delta": fused_metrics["mean_abs_delta"],
        "max_abs_delta": fused_metrics["max_abs_delta"],
    }


def _xgb_payload(model: XGBRegressor, reference_x: np.ndarray) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(text) for text in booster.get_dump(dump_format="json")]

    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(base._tree_leaf(tree, reference) for tree in trees)
    native_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = native_reference - tree_sum

    for vector in np.asarray(reference_x[: min(64, len(reference_x))], dtype=float):
        portable = base_score + sum(base._tree_leaf(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(f"XGBoost portable export mismatch: {portable} vs {native}")

    return {
        "base_score": float(base_score),
        "trees": trees,
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.025,
            "max_depth": 3,
            "min_child_weight": 2.5,
            "alpha": 0.1,
            "lambda": 0.3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": DEFAULT_RANDOM_STATE,
        },
    }


def _lgb_tree_leaf(node: Mapping[str, Any], vector: Sequence[float]) -> float:
    current: Mapping[str, Any] = node
    guard = 0
    while guard < 256:
        guard += 1
        if "leaf_value" in current:
            return float(current.get("leaf_value", 0.0))

        index = int(current.get("split_feature", -1))
        value = float(vector[index]) if 0 <= index < len(vector) else math.nan

        if not math.isfinite(value):
            go_left = bool(current.get("default_left", True))
        else:
            decision_type = str(current.get("decision_type", "<="))
            threshold = current.get("threshold", 0.0)
            if decision_type == "==":
                go_left = str(value) in str(threshold).split("||")
            else:
                go_left = value <= float(threshold)

        next_node = current.get("left_child") if go_left else current.get("right_child")
        if not isinstance(next_node, Mapping):
            return 0.0
        current = next_node

    return 0.0


def _lgb_payload(model: LGBMRegressor, reference_x: np.ndarray) -> dict[str, Any]:
    dump = model.booster_.dump_model()
    trees = [item["tree_structure"] for item in dump.get("tree_info", [])]

    for vector in np.asarray(reference_x[: min(64, len(reference_x))], dtype=float):
        portable = sum(_lgb_tree_leaf(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-8:
            raise RuntimeError(f"LightGBM portable export mismatch: {portable} vs {native}")

    return {
        "trees": trees,
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.025,
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 0.3,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "random_state": DEFAULT_RANDOM_STATE,
        },
    }


def export_portable_bundle(
    xgb_model: XGBRegressor,
    lgb_model: LGBMRegressor,
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
        "feature_names": list(FEATURE_NAMES),
        "fusion": {
            "method": FUSION_METHOD,
            "xgb_weight": 0.5,
            "lgb_weight": 0.5,
        },
        "max_delta": float(max_delta),
        "xgb": _xgb_payload(xgb_model, reference_x),
        "lgb": _lgb_payload(lgb_model, reference_x),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
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
    x, residual, actual_b, shoes = base.make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")

    validation = base.deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~validation

    xgb_model = build_xgb_regressor(random_state=args.random_state)
    lgb_model = build_lgb_regressor(random_state=args.random_state)

    # Same X_train and same y_train for both residual regressors.
    x_train = x[train]
    y_train = residual[train]
    xgb_model.fit(x_train, y_train)
    lgb_model.fit(x_train, y_train)

    validation_metrics = evaluate_ensemble(
        xgb_model,
        lgb_model,
        x[validation],
        actual_b[validation],
        max_delta=args.max_delta,
    )

    accepted = (
        validation_metrics["corrected_brier"]
        <= validation_metrics["core_brier"] + args.max_brier_regression
        and validation_metrics["corrected_accuracy"]
        >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": validation_metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected dual residual model; use --force only for diagnostics")

    final_xgb = build_xgb_regressor(random_state=args.random_state)
    final_lgb = build_lgb_regressor(random_state=args.random_state)
    final_xgb.fit(x, residual)
    final_lgb.fit(x, residual)

    export_portable_bundle(
        final_xgb,
        final_lgb,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost + LightGBM residual ensemble trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train both residual regressors and export browser JSON")
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