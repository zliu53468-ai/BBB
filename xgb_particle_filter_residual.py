#!/usr/bin/env python3
"""XGBoost + Particle Filter residual correction for BBB.

Upstream pipeline is frozen and unchanged:
    history -> 256D/V23 core -> core_p_b -> fixed 7D features

Downstream only:
    delta_xgb = XGBoost(features_7d)
    delta_pf = ParticleFilter latent residual estimate
    delta_final = (delta_xgb + delta_pf) / 2
    delta_clipped = clip(delta_final, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)
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
MODEL_TYPE = "xgb_particle_filter_residual"
SCHEMA_VERSION = 2
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 50,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 3.5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.4,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "state_dim": 1,
    "Q": 0.005,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
}


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMS)


class ResidualParticleFilter:
    """One-dimensional bootstrap particle filter for within-shoe residual drift."""

    def __init__(
        self,
        *,
        n_particles: int = 1000,
        q: float = 0.005,
        r: float = 0.25,
        resample_threshold: float = 500.0,
        random_state: int = 42,
    ) -> None:
        self.n_particles = int(n_particles)
        self.q = float(q)
        self.r = float(r)
        self.resample_threshold = float(resample_threshold)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.particles = np.zeros(self.n_particles, dtype=np.float64)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.particles = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(max(self.q, 1e-12)),
            size=self.n_particles,
        )
        self.weights.fill(1.0 / self.n_particles)

    def predict(self) -> float:
        self.particles += self.rng.normal(
            loc=0.0,
            scale=np.sqrt(max(self.q, 1e-12)),
            size=self.n_particles,
        )
        return self.estimate()

    def update(self, residual_observation: float) -> float:
        measurement = float(residual_observation)
        variance = max(self.r, 1e-12)
        error = measurement - self.particles
        log_likelihood = -0.5 * (error * error) / variance
        log_likelihood -= float(np.max(log_likelihood))
        self.weights *= np.exp(log_likelihood)
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        if self.effective_sample_size() < self.resample_threshold:
            self.systematic_resample()
        return self.estimate()

    def effective_sample_size(self) -> float:
        denom = float(np.sum(self.weights * self.weights))
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def systematic_resample(self) -> None:
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    def estimate(self) -> float:
        return float(np.sum(self.particles * self.weights))

    def observe_and_project(self, residual_observation: float) -> float:
        self.update(residual_observation)
        return self.predict()


class XGBParticleResidualPredictor:
    """Static XGBoost residual model fused 50/50 with an online particle filter."""

    def __init__(
        self,
        xgb_model: XGBRegressor | None = None,
        particle_filter: ResidualParticleFilter | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.xgb_model = xgb_model or build_xgb_regressor()
        self.particle_filter = particle_filter or ResidualParticleFilter(
            n_particles=PF_CONFIG["n_particles"],
            q=PF_CONFIG["Q"],
            r=PF_CONFIG["R"],
            resample_threshold=PF_CONFIG["resample_threshold"],
            random_state=42,
        )
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "XGBParticleResidualPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"feature_rows must be N x {len(FEATURE_NAMES)}")
        core_pb = x[:, FEATURE_NAMES.index("core_p_b")]
        residual = y - core_pb
        self.xgb_model.fit(x, residual)
        return self

    def reset_shoe(self) -> None:
        self.particle_filter.reset()

    def update_after_outcome(self, actual_b: int | float, core_pb: float) -> float:
        residual = float(actual_b) - float(core_pb)
        return self.particle_filter.observe_and_project(residual)

    def predict_components(self, features_7d: Sequence[float]) -> dict[str, float]:
        x = np.asarray(features_7d, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(FEATURE_NAMES)} values")
        delta_xgb = float(self.xgb_model.predict(x)[0])
        delta_pf = float(self.particle_filter.estimate())
        delta_final = (delta_xgb + delta_pf) / 2.0
        delta_clipped = float(np.clip(delta_final, -self.max_delta, self.max_delta))
        return {
            "delta_xgb": delta_xgb,
            "delta_pf": delta_pf,
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


def _portable_xgb_payload(model: XGBRegressor, reference_x: np.ndarray) -> dict[str, Any]:
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

    return {
        "base_score": float(base_score),
        "trees": trees,
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.02,
            "max_depth": 3,
            "min_child_weight": 3.5,
            "alpha": 0.1,
            "lambda": 0.4,
            "random_state": 42,
        },
    }


def _simulate_fused_validation(
    model: XGBRegressor,
    x: np.ndarray,
    residual: np.ndarray,
    actual_b: np.ndarray,
    shoes: Sequence[str],
    *,
    max_delta: float,
) -> dict[str, float]:
    raw_xgb = np.asarray(model.predict(x), dtype=float)
    raw_pf = np.zeros(len(x), dtype=float)
    current_shoe: str | None = None
    pf: ResidualParticleFilter | None = None

    for idx, shoe in enumerate(shoes):
        if shoe != current_shoe or pf is None:
            current_shoe = shoe
            pf = ResidualParticleFilter(
                n_particles=PF_CONFIG["n_particles"],
                q=PF_CONFIG["Q"],
                r=PF_CONFIG["R"],
                resample_threshold=PF_CONFIG["resample_threshold"],
                random_state=42,
            )
        raw_pf[idx] = pf.estimate()
        pf.observe_and_project(float(residual[idx]))

    raw_final = (raw_xgb + raw_pf) / 2.0
    delta = np.clip(raw_final, -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)
    return {
        "samples": float(len(x)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": base.direction_accuracy(final_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "corrected_brier": base.brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "mean_abs_xgb": float(np.mean(np.abs(raw_xgb))) if len(raw_xgb) else 0.0,
        "mean_abs_pf": float(np.mean(np.abs(raw_pf))) if len(raw_pf) else 0.0,
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
        "feature_names": list(FEATURE_NAMES),
        "fusion": {"method": "arithmetic_mean", "xgb_weight": 0.5, "pf_weight": 0.5},
        "max_delta": float(max_delta),
        "xgb": _portable_xgb_payload(model, reference_x),
        "particle_filter": dict(PF_CONFIG),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = base.load_training_records(Path(args.input))
    x, residual, actual_b, shoes = base.make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")

    validation = base.deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~validation

    model = build_xgb_regressor()
    model.fit(x[train], residual[train])
    validation_metrics = _simulate_fused_validation(
        model,
        x[validation],
        residual[validation],
        actual_b[validation],
        [shoe for shoe, keep in zip(shoes, validation) if keep],
        max_delta=args.max_delta,
    )

    accepted = (
        validation_metrics["corrected_brier"] <= validation_metrics["core_brier"] + args.max_brier_regression
        and validation_metrics["corrected_accuracy"] >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": validation_metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected XGB + particle-filter residual model; use --force only for diagnostics")

    final_model = build_xgb_regressor()
    final_model.fit(x, residual)
    export_portable_bundle(
        final_model,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost + particle-filter residual trainer")
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
