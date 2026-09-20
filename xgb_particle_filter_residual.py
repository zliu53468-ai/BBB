#!/usr/bin/env python3
"""Shoe-regime Particle Filter state injection into downstream XGBoost for BBB.

Frozen upstream pipeline:
    history -> 256D/V23 core -> core_p_b -> fixed 7D features

Downstream only:
    regime_state = online ShoeRegimeParticleFilter latent environment state
    features_8d = [features_7d..., regime_state]
    delta = XGBoost(features_8d)
    delta_clipped = clip(delta, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)

The filter tracks within-shoe regime quality, not card composition. Cut-card /
shoe depth changes state-transition responsiveness only and never directly
selects Banker or Player. Training is causal: row t's regime_state uses only
outcomes before row t.
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

UPSTREAM_FEATURE_NAMES: tuple[str, ...] = base.FEATURE_NAMES
MODEL_FEATURE_NAMES: tuple[str, ...] = (*UPSTREAM_FEATURE_NAMES, "regime_state")
MODEL_TYPE = "xgb_shoe_regime_feature_residual"
SCHEMA_VERSION = 4
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA

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

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "state_dim": 1,
    "Q_start": 0.005,
    "Q_end": 0.02,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
    "random_state": 42,
    "state_clip": 1.0,
    "observation_weights": {
        "direction_alignment": 0.55,
        "confidence_alignment": 0.30,
        "persistence": 0.15,
    },
}

_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = float(2**32)


class DeterministicRNG:
    """Cross-runtime LCG shared by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE

    def normal(self) -> float:
        u1 = max(self.uniform(), 1e-15)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMS)


class ShoeRegimeParticleFilter:
    """One-dimensional PF tracking the current shoe's structural regime.

    +1 = sustained core-aligned structure
     0 = turbulence / mixed evidence
    -1 = sustained core-opposed structure
    """

    def __init__(
        self,
        *,
        n_particles: int = 1000,
        q_start: float = 0.005,
        q_end: float = 0.02,
        r: float = 0.25,
        resample_threshold: float = 500.0,
        random_state: int = 42,
        state_clip: float = 1.0,
    ) -> None:
        self.n_particles = int(n_particles)
        self.q_start = float(q_start)
        self.q_end = float(q_end)
        self.r = float(r)
        self.resample_threshold = float(resample_threshold)
        self.random_state = int(random_state)
        self.state_clip = float(state_clip)
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.zeros(self.n_particles, dtype=np.float64)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_alignment: float | None = None
        self.last_observation = 0.0
        self.last_effective_q = self.q_start_start_start
        self.reset()

    def reset(self) -> None:
        self.rng = DeterministicRNG(self.random_state)
        std = math.sqrt(max(self.q_start, 1e-12))
        draws = np.asarray([self.rng.normal() * std for _ in range(self.n_particles)], dtype=np.float64)
        draws -= float(np.mean(draws))
        self.particles = np.clip(draws, -self.state_clip, self.state_clip)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_alignment = None
        self.last_observation = 0.0
        self.last_effective_q = self.q_start

    def estimate(self) -> float:
        value = float(np.sum(self.particles * self.weights))
        return float(np.clip(value, -self.state_clip, self.state_clip))

    def effective_sample_size(self) -> float:
        denom = float(np.sum(self.weights * self.weights))
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def systematic_resample(self) -> None:
        positions = (self.rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    @staticmethod
    def shoe_progress(round_index: float, estimated_total_hands: float) -> float:
        total = max(2.0, float(estimated_total_hands))
        return float(np.clip((float(round_index) - 1.0) / (total - 1.0), 0.0, 1.0))

    def effective_q(self, current_round: float, estimated_total_hands: float) -> float:
        progress = self.shoe_progress(current_round, estimated_total_hands)
        return max(1e-12, self.q_start + (self.q_end - self.q_start) * progress)

    def make_observation(self, *, actual_b: float, core_pb: float) -> tuple[float, float]:
        actual_is_b = float(actual_b) >= 0.5
        p_b = base.clip(float(core_pb), 0.0, 1.0)
        core_is_b = p_b > 0.5
        alignment = 1.0 if core_is_b == actual_is_b else -1.0
        actual_probability = p_b if actual_is_b else (1.0 - p_b)
        confidence_alignment = float(np.clip(2.0 * (actual_probability - 0.5), -1.0, 1.0))
        weights = PF_CONFIG["observation_weights"]

        if alignment > 0.0:
            persistence = 1.0 if self.last_alignment == 1.0 else 0.0
            observation = (
                float(weights["direction_alignment"])
                + float(weights["confidence_alignment"]) * max(0.0, confidence_alignment)
                + float(weights["persistence"]) * persistence
            )
        elif self.last_alignment == -1.0:
            # A repeated miss is evidence of a persistent Core-opposed regime.
            persistence = -1.0
            observation = -(
                float(weights["direction_alignment"])
                + float(weights["confidence_alignment"]) * abs(min(0.0, confidence_alignment))
                + float(weights["persistence"])
            )
        else:
            # A single break / abrupt disorder is turbulence, not an instant reversal.
            persistence = 0.0
            observation = 0.0

        return float(np.clip(observation, -1.0, 1.0)), alignment

    def update_observation(self, observation: float) -> float:
        measurement = float(np.clip(observation, -1.0, 1.0))
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
        self.last_observation = measurement
        return self.estimate()

    def predict(self, *, current_round: float, estimated_total_hands: float) -> float:
        q_eff = self.effective_q(current_round, estimated_total_hands)
        std = math.sqrt(q_eff)
        noise = np.asarray([self.rng.normal() * std for _ in range(self.n_particles)], dtype=np.float64)
        self.particles = np.clip(self.particles + noise, -self.state_clip, self.state_clip)
        self.last_effective_q = q_eff
        return self.estimate()

    def observe_and_project(
        self,
        *,
        actual_b: float,
        core_pb: float,
        current_round: float,
        estimated_total_hands: float,
    ) -> float:
        observation, alignment = self.make_observation(actual_b=actual_b, core_pb=core_pb)
        self.update_observation(observation)
        self.last_alignment = alignment
        self.updates += 1
        return self.predict(
            current_round=float(current_round) + 1.0,
            estimated_total_hands=estimated_total_hands,
        )


def _new_particle_filter() -> ShoeRegimeParticleFilter:
    return ShoeRegimeParticleFilter(
        n_particles=PF_CONFIG["n_particles"],
        q_start=PF_CONFIG["Q_start"],
        q_end=PF_CONFIG["Q_end"],
        r=PF_CONFIG["R"],
        resample_threshold=PF_CONFIG["resample_threshold"],
        random_state=PF_CONFIG["random_state"],
        state_clip=PF_CONFIG["state_clip"],
    )


def combine_features_8d(features_7d: Sequence[float], regime_state: float) -> np.ndarray:
    base_vector = np.asarray(features_7d, dtype=np.float32).reshape(-1)
    if base_vector.shape[0] != len(UPSTREAM_FEATURE_NAMES):
        raise ValueError(f"features_7d must contain {len(UPSTREAM_FEATURE_NAMES)} values")
    return np.concatenate([base_vector, np.asarray([float(regime_state)], dtype=np.float32)])


def make_training_arrays_8d(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build causal 8D rows with pre-outcome Shoe Regime state."""
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    filters: dict[str, ShoeRegimeParticleFilter] = {}

    for idx, record in enumerate(records):
        try:
            actual_b = base._parse_actual_b(record)
            row = base._feature_row(record)
            vector7 = [float(row[name]) for name in UPSTREAM_FEATURE_NAMES]
            if not all(math.isfinite(value) for value in vector7):
                continue
            core_p_b = base.clip(float(row["core_p_b"]))
            round_index = float(row["round_index"])
            estimated_total_hands = float(row["estimated_total_hands"])
        except (TypeError, ValueError, KeyError):
            continue

        shoe_id = str(record.get("shoe_id") or f"row_{idx}")
        pf = filters.get(shoe_id)
        if pf is None:
            pf = _new_particle_filter()
            filters[shoe_id] = pf

        regime_state = pf.estimate()
        residual = float(actual_b) - core_p_b
        vectors.append([*vector7, float(regime_state)])
        residuals.append(residual)
        actuals.append(actual_b)
        shoes.append(shoe_id)

        pf.observe_and_project(
            actual_b=actual_b,
            core_pb=core_p_b,
            current_round=round_index,
            estimated_total_hands=estimated_total_hands,
        )

    if not vectors:
        raise ValueError("no valid B/P training rows")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        shoes,
    )

class ShoeRegimeXGBResidualPredictor:
    """Shoe Regime state -> eighth feature -> single XGBoost residual correction."""

    def __init__(
        self,
        xgb_model: XGBRegressor | None = None,
        particle_filter: ShoeRegimeParticleFilter | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.xgb_model = xgb_model or build_xgb_regressor()
        self.particle_filter = particle_filter or _new_particle_filter()
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(self, feature_rows_8d: np.ndarray, residual_targets: Sequence[float]) -> "ShoeRegimeXGBResidualPredictor":
        x = np.asarray(feature_rows_8d, dtype=np.float32)
        y = np.asarray(residual_targets, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(MODEL_FEATURE_NAMES):
            raise ValueError(f"feature_rows_8d must be N x {len(MODEL_FEATURE_NAMES)}")
        if len(x) != len(y):
            raise ValueError("feature_rows_8d and residual_targets must have equal length")
        self.xgb_model.fit(x, y)
        return self

    def reset_shoe(self) -> None:
        self.particle_filter.reset()

    def current_regime_state(self) -> float:
        return self.particle_filter.estimate()

    def update_after_outcome(
        self,
        *,
        actual_b: int | float,
        core_pb: float,
        round_index: float,
        estimated_total_hands: float,
    ) -> float:
        return self.particle_filter.observe_and_project(
            actual_b=float(actual_b),
            core_pb=float(core_pb),
            current_round=float(round_index),
            estimated_total_hands=float(estimated_total_hands),
        )

    def predict_delta(self, features_7d: Sequence[float]) -> dict[str, float]:
        regime_state = self.current_regime_state()
        x8 = combine_features_8d(features_7d, regime_state).reshape(1, -1)
        raw_delta = float(self.xgb_model.predict(x8)[0])
        delta_clipped = float(np.clip(raw_delta, -self.max_delta, self.max_delta))
        return {"regime_state": regime_state, "delta_raw": raw_delta, "delta_clipped": delta_clipped}

    def correct(self, features_7d: Sequence[float], core_pb: float | None = None) -> dict[str, Any]:
        x7 = np.asarray(features_7d, dtype=np.float32).reshape(-1)
        if x7.shape[0] != len(UPSTREAM_FEATURE_NAMES):
            raise ValueError(f"features_7d must contain {len(UPSTREAM_FEATURE_NAMES)} values")
        core_value = (
            base.clip(float(core_pb), 0.0, 1.0)
            if core_pb is not None
            else base.clip(float(x7[UPSTREAM_FEATURE_NAMES.index("core_p_b")]), 0.0, 1.0)
        )
        prediction = self.predict_delta(x7)
        final_pb = base.clip(core_value + prediction["delta_clipped"], 0.0, 1.0)
        return {
            "core_p_b": core_value,
            **prediction,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.50 else "P",
        }


def _tree_leaf_8d(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
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


def _portable_xgb_payload(model: XGBRegressor, reference_x: np.ndarray) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(text) for text in booster.get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf_8d(tree, reference) for tree in trees)
    native_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = native_reference - tree_sum
    for vector in np.asarray(reference_x[: min(64, len(reference_x))], dtype=float):
        portable = base_score + sum(_tree_leaf_8d(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(f"portable export mismatch: {portable} vs {native}")
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


def evaluate_8d(model: XGBRegressor, x8: np.ndarray, actual_b: np.ndarray, *, max_delta: float) -> dict[str, float]:
    raw_delta = np.asarray(model.predict(x8), dtype=float)
    delta = np.clip(raw_delta, -max_delta, max_delta)
    core_pb = x8[:, MODEL_FEATURE_NAMES.index("core_p_b")].astype(float)
    final_pb = np.clip(core_pb + delta, 0.0, 1.0)
    regime_state = x8[:, MODEL_FEATURE_NAMES.index("regime_state")].astype(float)
    return {
        "samples": float(len(x8)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": base.direction_accuracy(final_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "corrected_brier": base.brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "mean_abs_regime_state": float(np.mean(np.abs(regime_state))) if len(regime_state) else 0.0,
        "mean_regime_state": float(np.mean(regime_state)) if len(regime_state) else 0.0,
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
        "feature_schema": "7D_UPSTREAM_PLUS_1D_SHOE_REGIME",
        "max_delta": float(max_delta),
        "xgb": _portable_xgb_payload(model, reference_x),
        "shoe_regime_filter": dict(PF_CONFIG),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "regime_state_timing": "state_before_current_outcome",
            "regime_state_meaning": "+1 core-aligned, 0 turbulent, -1 core-opposed",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = base.load_training_records(Path(args.input))
    x8, residual, actual_b, shoes = make_training_arrays_8d(records)
    if len(x8) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x8)}")

    validation = base.deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~validation
    model = build_xgb_regressor()
    model.fit(x8[train], residual[train])
    validation_metrics = evaluate_8d(model, x8[validation], actual_b[validation], max_delta=args.max_delta)
    accepted = (
        validation_metrics["corrected_brier"] <= validation_metrics["core_brier"] + args.max_brier_regression
        and validation_metrics["corrected_accuracy"] >= validation_metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": validation_metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected Shoe-Regime 8D XGBoost model; use --force only for diagnostics")

    final_model = build_xgb_regressor()
    final_model.fit(x8, residual)
    export_portable_bundle(
        final_model,
        reference_x=x8,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x8),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB Shoe-Regime-filter injected 8D XGBoost residual trainer")
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