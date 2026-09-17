#!/usr/bin/env python3
"""Twin XGBoost residual ensemble for the frozen BBB 256D/V23 core.

The upstream pipeline is intentionally unchanged:
    history -> 256D/V23 core -> core_p_b -> fixed 7D features

Both downstream regressors consume the exact same 7D matrix and the exact same
residual target:
    residual = actual_B - core_p_B

Inference:
    delta_sens = xgb_sensitive(features_7d)
    delta_robu = xgb_robust(features_7d)
    delta_final = (delta_sens + delta_robu) / 2.0
    delta_clipped = clip(delta_final, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0.0, 1.0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from xgboost import XGBRegressor

import xgb_residual_bias as base

FEATURE_NAMES = base.FEATURE_NAMES
MODEL_TYPE = "xgb_twin_residual_ensemble"
SCHEMA_VERSION = 2
FUSION_METHOD = "arithmetic_mean"
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA

SENSITIVE_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 60,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.01,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}

ROBUST_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 50,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 3.0,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "random_state": 100,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}


def build_sensitive_regressor() -> XGBRegressor:
    return XGBRegressor(**SENSITIVE_PARAMS)


def build_robust_regressor() -> XGBRegressor:
    return XGBRegressor(**ROBUST_PARAMS)


class TwinResidualBiasPredictor:
    """Reusable 50/50 sensitive + robust XGBoost residual ensemble."""

    def __init__(
        self,
        sensitive_model: XGBRegressor | None = None,
        robust_model: XGBRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.sensitive_model = sensitive_model or build_sensitive_regressor()
        self.robust_model = robust_model or build_robust_regressor()
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "TwinResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"feature_rows must be N x {len(FEATURE_NAMES)}")
        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        residual = y - core_pb
        self.sensitive_model.fit(x, residual)
        self.robust_model.fit(x, residual)
        return self

    def predict_components(self, features_7d: Sequence[float]) -> dict[str, float]:
        x = np.asarray(features_7d, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(FEATURE_NAMES)} values")
        delta_sens = float(self.sensitive_model.predict(x)[0])
        delta_robu = float(self.robust_model.predict(x)[0])
        delta_final = (delta_sens + delta_robu) / 2.0
        delta_clipped = float(np.clip(delta_final, -self.max_delta, self.max_delta))
        return {
            "delta_sensitive": delta_sens,
            "delta_robust": delta_robu,
            "delta_final": delta_final,
            "delta_clipped": delta_clipped,
        }

    def correct(self, features_7d: Sequence[float], core_pb: float | None = None) -> dict[str, Any]:
        x = np.asarray(features_7d, dtype=np.float32)
        if x.ndim != 1 or x.shape[0] != len(FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(FEATURE_NAMES)} values")
        core_value = (
            base.clip(float(core_pb), 0.0, 1.0)
            if core_pb is not None
            else base.clip(float(x[FEATURE_NAMES.index("core_p_b")]), 0.0, 1.0)
        )
        parts = self.predict_components(x)
        final_pb = base.clip(core_value + parts["delta_clipped"], 0.0, 1.0)
        return {
            "core_p_b": core_value,
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


def evaluate_twin(
    sensitive_model: XGBRegressor,
    robust_model: XGBRegressor,
    x: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    delta_sens = np.asarray(sensitive_model.predict(x), dtype=float)
    delta_robu = np.asarray(robust_model.predict(x), dtype=float)
    delta_final = (delta_sens + delta_robu) / 2.0
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    sens_metrics = _corrected_metrics(x, actual_b, delta_sens, max_delta=max_delta)
    robu_metrics = _corrected_metrics(x, actual_b, delta_robu, max_delta=max_delta)
    fused_metrics = _corrected_metrics(x, actual_b, delta_final, max_delta=max_delta)
    return {
        "samples": float(len(x)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "sensitive_accuracy": sens_metrics["accuracy"],
        "sensitive_brier": sens_metrics["brier"],
        "robust_accuracy": robu_metrics["accuracy"],
        "robust_brier": robu_metrics["brier"],
        "corrected_accuracy": fused_metrics["accuracy"],
        "corrected_brier": fused_metrics["brier"],
        "mean_abs_delta": fused_metrics["mean_abs_delta"],
        "max_abs_delta": fused_metrics["max_abs_delta"],
    }


def _xgb_payload(model: XGBRegressor, reference_x: np.ndarray, params: Mapping[str, Any]) -> dict[str, Any]:
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
            raise RuntimeError(f"portable export mismatch: {portable} vs {native}")

    exported_params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "alpha": float(params["reg_alpha"]),
        "lambda": float(params["reg_lambda"]),
        "random_state": int(params["random_state"]),
    }
    return {
        "base_score": float(base_score),
        "trees": trees,
        "params": exported_params,
    }


def export_portable_bundle(
    sensitive_model: XGBRegressor,
    robust_model: XGBRegressor,
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
            "sensitive_weight": 0.5,
            "robust_weight": 0.5,
        },
        "max_delta": float(max_delta),
        "sensitive": _xgb_payload(sensitive_model, reference_x, SENSITIVE_PARAMS),
        "robust": _xgb_payload(robust_model, reference_x, ROBUST_PARAMS),
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
    x_train = x[train]
    y_train = residual[train]

    sensitive = build_sensitive_regressor()
    robust = build_robust_regressor()
    sensitive.fit(x_train, y_train)
    robust.fit(x_train, y_train)

    validation_metrics = evaluate_twin(
        sensitive,
        robust,
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
        raise SystemExit("validation gate rejected twin residual model; use --force only for diagnostics")

    final_sensitive = build_sensitive_regressor()
    final_robust = build_robust_regressor()
    final_sensitive.fit(x, residual)
    final_robust.fit(x, residual)

    export_portable_bundle(
        final_sensitive,
        final_robust,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB twin XGBoost residual trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train sensitive + robust XGBoost regressors")
    train.add_argument("--input", required=True, help="browser-exported JSON or CSV")
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
