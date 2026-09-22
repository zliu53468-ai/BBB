#!/usr/bin/env python3
"""Physics-extended XGBoost residual layer.

This file is deliberately separate from xgb_residual_bias.py so the existing
7D production residual layer remains untouched and rollback-safe.

Extended vector:
    [core_p_b] + [original fixed 7D, unchanged] + [48D physics expectations]

Residual target:
    actual_B - core_p_b

Final correction:
    delta = clip(xgb_residual, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta, 0, 1)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from xgboost import XGBRegressor

from physics_feature_extractor import (
    PHYSICS_DIM,
    PHYSICS_FEATURE_NAMES,
    PhysicsFeatureExtractor,
    prepare_xgboost_input,
)
from xgb_residual_bias import (
    FEATURE_NAMES as ORIGINAL_7D_FEATURE_NAMES,
    build_features as build_original_7d,
    deterministic_validation_mask,
    load_training_records,
)

MAX_DELTA = 0.10
RANDOM_STATE = 20260922
EXTENDED_FEATURE_NAMES: tuple[str, ...] = (
    ("core_p_b_external",)
    + tuple(f"original7_{name}" for name in ORIGINAL_7D_FEATURE_NAMES)
    + PHYSICS_FEATURE_NAMES
)
EXTENDED_DIM = len(EXTENDED_FEATURE_NAMES)
assert len(ORIGINAL_7D_FEATURE_NAMES) == 7
assert EXTENDED_DIM == 1 + 7 + PHYSICS_DIM


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _actual_b(record: Mapping[str, Any]) -> int:
    if record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").strip().upper()
    if actual == "B":
        return 1
    if actual == "P":
        return 0
    raise ValueError("row must contain actual_b or actual_outcome B/P")


def _history(record: Mapping[str, Any]) -> str | Sequence[str]:
    return record.get("history") or record.get("history_fingerprint") or ""


def _core_pb(record: Mapping[str, Any]) -> float:
    if record.get("core_p_b") is not None:
        return clip(float(record["core_p_b"]))
    if record.get("core_pb") is not None:
        return clip(float(record["core_pb"]))
    raise ValueError("row missing core_p_b")


def _original_7d(record: Mapping[str, Any], core_pb: float) -> np.ndarray:
    if all(record.get(name) is not None for name in ORIGINAL_7D_FEATURE_NAMES):
        return np.asarray([float(record[name]) for name in ORIGINAL_7D_FEATURE_NAMES], dtype=np.float32)

    features = build_original_7d(
        core_p_b=core_pb,
        history=_history(record),
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    )
    return features.as_vector().astype(np.float32)


def record_to_extended_vector(
    record: Mapping[str, Any],
    extractor: PhysicsFeatureExtractor,
) -> tuple[np.ndarray, int, float, str]:
    core_pb = _core_pb(record)
    original = _original_7d(record, core_pb)
    vector = prepare_xgboost_input(
        core_pb,
        original,
        _history(record),
        extractor=extractor,
    )
    actual_b = _actual_b(record)
    shoe_id = str(record.get("shoe_id") or record.get("session_id") or "")
    return vector, actual_b, core_pb, shoe_id


def make_extended_training_arrays(
    records: Sequence[Mapping[str, Any]],
    extractor: PhysicsFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    residuals: list[float] = []
    actuals: list[int] = []
    core_probabilities: list[float] = []
    shoes: list[str] = []

    for i, record in enumerate(records):
        try:
            vector, actual_b, core_pb, shoe_id = record_to_extended_vector(record, extractor)
        except (TypeError, ValueError, KeyError):
            continue
        if vector.size != EXTENDED_DIM or not np.all(np.isfinite(vector)):
            continue
        vectors.append(vector)
        residuals.append(float(actual_b) - core_pb)
        actuals.append(actual_b)
        core_probabilities.append(core_pb)
        shoes.append(shoe_id or f"row_{i}")

    if not vectors:
        raise ValueError("no valid B/P rows for physics-extended residual training")

    return (
        np.vstack(vectors).astype(np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        np.asarray(core_probabilities, dtype=np.float32),
        shoes,
    )


def build_residual_regressor(*, random_state: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=280,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=10.0,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.25,
        reg_lambda=10.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def _direction_accuracy(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b > 0.5) == (actual_b > 0)))


def _brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def evaluate(
    model: XGBRegressor,
    x: np.ndarray,
    actual_b: np.ndarray,
    core_pb: np.ndarray,
    *,
    max_delta: float = MAX_DELTA,
) -> dict[str, float]:
    raw = np.asarray(model.predict(x), dtype=np.float64)
    delta = np.clip(raw, -max_delta, max_delta)
    final_pb = np.clip(core_pb.astype(np.float64) + delta, 0.0, 1.0)
    return {
        "samples": float(len(x)),
        "core_accuracy": _direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": _direction_accuracy(final_pb, actual_b),
        "core_brier": _brier(core_pb, actual_b),
        "corrected_brier": _brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
    }


class PhysicsResidualBiasPredictor:
    def __init__(
        self,
        model: XGBRegressor | None = None,
        *,
        physics_extractor: PhysicsFeatureExtractor,
        max_delta: float = MAX_DELTA,
    ) -> None:
        self.model = model or build_residual_regressor()
        self.physics_extractor = physics_extractor
        self.max_delta = min(MAX_DELTA, max(0.0, float(max_delta)))
        self.is_fitted = model is not None

    def fit(
        self,
        x: np.ndarray,
        residual_target: np.ndarray,
    ) -> "PhysicsResidualBiasPredictor":
        self.model.fit(np.asarray(x, dtype=np.float32), np.asarray(residual_target, dtype=np.float32))
        self.is_fitted = True
        return self

    def prepare(
        self,
        core_pb: float,
        original_7d: Sequence[float],
        history_path: str | Sequence[str],
    ) -> np.ndarray:
        return prepare_xgboost_input(
            core_pb,
            original_7d,
            history_path,
            extractor=self.physics_extractor,
        )

    def correct(
        self,
        core_pb: float,
        original_7d: Sequence[float],
        history_path: str | Sequence[str],
    ) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("residual model is not fitted")
        x = self.prepare(core_pb, original_7d, history_path).reshape(1, -1)
        raw_delta = float(self.model.predict(x)[0])
        delta = float(np.clip(raw_delta, -self.max_delta, self.max_delta))
        final_pb = clip(float(core_pb) + delta)
        return {
            "core_p_b": clip(core_pb),
            "raw_delta": raw_delta,
            "delta": delta,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.5 else "P",
            "feature_dim": EXTENDED_DIM,
        }


def save_bundle(
    path: str | Path,
    model: XGBRegressor,
    *,
    physics_model_path: str,
    max_delta: float,
    metrics: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "model_type": "xgb_residual_physics_extended",
        "feature_names": list(EXTENDED_FEATURE_NAMES),
        "original_7d_feature_names": list(ORIGINAL_7D_FEATURE_NAMES),
        "physics_feature_names": list(PHYSICS_FEATURE_NAMES),
        "max_delta": float(max_delta),
        "physics_model_path": physics_model_path,
        "metrics": dict(metrics),
        "model": model,
    }
    joblib.dump(payload, path)


def load_bundle(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if payload.get("model_type") != "xgb_residual_physics_extended":
        raise ValueError("invalid residual bundle")
    if tuple(payload.get("feature_names") or ()) != EXTENDED_FEATURE_NAMES:
        raise ValueError("extended feature schema mismatch")
    return payload


def train_command(args: argparse.Namespace) -> int:
    physics = PhysicsFeatureExtractor.load(args.physics_model)
    records = load_training_records(Path(args.input))
    x, residual, actual_b, core_pb, shoes = make_extended_training_arrays(records, physics)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid B/P rows; got {len(x)}")

    valid = deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~valid

    probe = build_residual_regressor(random_state=args.random_state)
    probe.fit(x[train], residual[train])
    metrics = evaluate(probe, x[valid], actual_b[valid], core_pb[valid], max_delta=args.max_delta)

    accepted = (
        metrics["corrected_brier"] <= metrics["core_brier"] + args.max_brier_regression
        and metrics["corrected_accuracy"] >= metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected physics residual model; use --force only for diagnostics")

    final_model = build_residual_regressor(random_state=args.random_state)
    final_model.fit(x, residual)
    save_bundle(
        args.output,
        final_model,
        physics_model_path=args.physics_model,
        max_delta=args.max_delta,
        metrics=metrics,
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB physics-extended XGBoost residual trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train residual XGB on core + original7D + physics48D")
    train.add_argument("--input", required=True, help="browser-exported labeled residual JSON/CSV")
    train.add_argument("--physics-model", required=True, help="trained PhysicsFeatureExtractor UBJ/JSON model")
    train.add_argument("--output", default="residual_bias_physics.joblib")
    train.add_argument("--min-samples", type=int, default=500)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--max-delta", type=float, default=MAX_DELTA)
    train.add_argument("--random-state", type=int, default=RANDOM_STATE)
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
