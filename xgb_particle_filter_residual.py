#!/usr/bin/env python3
"""PF base-margin XGBoost residual correction for BBB.

Frozen upstream pipeline:
    history -> 256D/V23 Core -> core_p_b -> fixed 7D features

Downstream:
    pf_delta = blind ShoeParticleFilter residual prior in [-0.10, +0.10]
    DMatrix(features_7d).set_base_margin(pf_delta)
    total_delta = XGBoost Booster prediction
    delta_clipped = clip(total_delta, -0.10, +0.10)
    final_p_b = clip(core_p_b + delta_clipped, 0, 1)

The Particle Filter does not occupy a feature dimension. XGBoost always sees
exactly the original seven frozen features.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xgboost as xgb

import xgb_residual_bias as base
from shoe_particle_filter import PF_CONFIG, ShoeParticleFilter, new_shoe_particle_filter

UPSTREAM_FEATURE_NAMES: tuple[str, ...] = base.FEATURE_NAMES
MODEL_FEATURE_NAMES: tuple[str, ...] = UPSTREAM_FEATURE_NAMES
MODEL_TYPE = "xgb_pf_base_margin_residual"
SCHEMA_VERSION = 8
DEFAULT_MAX_DELTA = base.DEFAULT_MAX_DELTA

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eta": 0.02,
    "max_depth": 3,
    "alpha": 0.1,
    "lambda": 0.3,
    "seed": 42,
    "tree_method": "hist",
    "nthread": 1,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 50


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build causal 7D rows plus one base-margin value per row.

    pf_delta for round t is captured before outcome t is used to update the
    hidden-shoe particle filter. This prevents label leakage.
    """
    vectors_7d: list[list[float]] = []
    margins: list[float] = []
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

        pf_delta = particle_filter.current_pf_delta()
        residual_target = float(actual_b) - core_p_b

        vectors_7d.append(vector7)
        margins.append(float(pf_delta))
        residuals.append(residual_target)
        actuals.append(actual_b)
        shoes.append(shoe_id)

        particle_filter.observe_and_project(
            real_outcome=actual_b,
            core_pb=core_p_b,
            current_round=round_index,
        )

    if not vectors_7d:
        raise ValueError("no valid B/P training rows")

    return (
        np.asarray(vectors_7d, dtype=np.float32),
        np.asarray(margins, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        shoes,
    )


def make_dmatrix(
    features_7d: np.ndarray,
    *,
    base_margin: np.ndarray,
    labels: np.ndarray | None = None,
) -> xgb.DMatrix:
    matrix = np.asarray(features_7d, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(UPSTREAM_FEATURE_NAMES):
        raise ValueError(
            f"features_7d must have shape (N, {len(UPSTREAM_FEATURE_NAMES)})"
        )

    kwargs: dict[str, Any] = {
        "data": matrix,
        "feature_names": list(UPSTREAM_FEATURE_NAMES),
    }
    if labels is not None:
        kwargs["label"] = np.asarray(labels, dtype=np.float32)

    dmatrix = xgb.DMatrix(**kwargs)
    margin = np.asarray(base_margin, dtype=np.float32).reshape(-1)
    if len(margin) != matrix.shape[0]:
        raise ValueError("base_margin length must match number of rows")
    dmatrix.set_base_margin(margin)
    return dmatrix


def train_booster(
    features_7d: np.ndarray,
    residual_targets: np.ndarray,
    pf_delta: np.ndarray,
) -> xgb.Booster:
    dtrain = make_dmatrix(
        features_7d,
        labels=residual_targets,
        base_margin=pf_delta,
    )
    return xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
    )


class PFBaseMarginXGBResidualPredictor:
    """PF physical prior via base_margin + 7D XGBoost residual correction."""

    def __init__(
        self,
        booster: xgb.Booster | None = None,
        particle_filter: ShoeParticleFilter | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.booster = booster
        self.particle_filter = particle_filter or new_shoe_particle_filter()
        self.max_delta = base.clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def fit(
        self,
        feature_rows_7d: np.ndarray,
        residual_targets: Sequence[float],
        historical_pf_delta: Sequence[float],
    ) -> "PFBaseMarginXGBResidualPredictor":
        x7 = np.asarray(feature_rows_7d, dtype=np.float32)
        y = np.asarray(residual_targets, dtype=np.float32)
        margins = np.asarray(historical_pf_delta, dtype=np.float32)
        if len(x7) != len(y) or len(x7) != len(margins):
            raise ValueError("features, targets, and base margins must align")
        self.booster = train_booster(x7, y, margins)
        return self

    def reset_shoe(self) -> None:
        self.particle_filter.reset()

    def current_pf_delta(self) -> float:
        return self.particle_filter.current_pf_delta()

    def update_after_outcome(
        self,
        *,
        actual_b: int | float,
        core_pb: float,
        round_index: float,
    ) -> float:
        return self.particle_filter.observe_and_project(
            real_outcome=float(actual_b),
            core_pb=float(core_pb),
            current_round=float(round_index),
        )

    def predict_delta(self, features_7d: Sequence[float]) -> dict[str, float]:
        if self.booster is None:
            raise RuntimeError("XGBoost Booster is not trained")

        x7 = np.asarray(features_7d, dtype=np.float32).reshape(1, -1)
        pf_delta = self.current_pf_delta()
        dtest = make_dmatrix(
            x7,
            base_margin=np.asarray([pf_delta], dtype=np.float32),
        )
        total_delta = float(self.booster.predict(dtest)[0])
        delta_clipped = float(
            np.clip(total_delta, -self.max_delta, self.max_delta)
        )
        return {
            "pf_delta": pf_delta,
            "total_delta": total_delta,
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


def _tree_leaf_7d(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
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
                index = UPSTREAM_FEATURE_NAMES.index(split)
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
    booster: xgb.Booster,
    reference_x: np.ndarray,
    reference_margin: np.ndarray,
) -> dict[str, Any]:
    trees = [
        json.loads(text)
        for text in booster.get_dump(dump_format="json")
    ]

    sample_count = min(64, len(reference_x))
    for vector, margin in zip(
        np.asarray(reference_x[:sample_count], dtype=float),
        np.asarray(reference_margin[:sample_count], dtype=float),
    ):
        dtest = make_dmatrix(
            vector.reshape(1, -1),
            base_margin=np.asarray([margin], dtype=np.float32),
        )
        native = float(booster.predict(dtest)[0])
        tree_sum = sum(_tree_leaf_7d(tree, vector) for tree in trees)
        portable = float(margin) + tree_sum
        if abs(portable - native) > 1e-5:
            raise RuntimeError(
                f"portable base-margin export mismatch: {portable} vs {native}"
            )

    return {
        "uses_base_margin": True,
        "base_margin_source": "pf_delta",
        "trees": trees,
        "num_boost_round": NUM_BOOST_ROUND,
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.02,
            "max_depth": 3,
            "alpha": 0.1,
            "lambda": 0.3,
            "random_state": 42,
        },
    }


def evaluate_7d_with_margin(
    booster: xgb.Booster,
    features_7d: np.ndarray,
    pf_delta: np.ndarray,
    actual_b: np.ndarray,
    *,
    max_delta: float,
) -> dict[str, float]:
    dmatrix = make_dmatrix(
        features_7d,
        base_margin=pf_delta,
    )
    total_delta = np.asarray(booster.predict(dmatrix), dtype=float)
    clipped_delta = np.clip(total_delta, -max_delta, max_delta)

    core_pb = features_7d[
        :, UPSTREAM_FEATURE_NAMES.index("core_p_b")
    ].astype(float)
    final_pb = np.clip(core_pb + clipped_delta, 0.0, 1.0)

    return {
        "samples": float(len(features_7d)),
        "core_accuracy": base.direction_accuracy(core_pb, actual_b),
        "corrected_accuracy": base.direction_accuracy(final_pb, actual_b),
        "core_brier": base.brier(core_pb, actual_b),
        "corrected_brier": base.brier(final_pb, actual_b),
        "mean_abs_pf_delta": (
            float(np.mean(np.abs(pf_delta))) if len(pf_delta) else 0.0
        ),
        "mean_abs_total_delta": (
            float(np.mean(np.abs(total_delta))) if len(total_delta) else 0.0
        ),
        "max_abs_total_delta": (
            float(np.max(np.abs(total_delta))) if len(total_delta) else 0.0
        ),
    }


def export_portable_bundle(
    booster: xgb.Booster,
    *,
    reference_x: np.ndarray,
    reference_margin: np.ndarray,
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
        "feature_schema": "7D_WITH_PF_BASE_MARGIN",
        "max_delta": float(max_delta),
        "xgb": _portable_xgb_payload(
            booster,
            reference_x,
            reference_margin,
        ),
        "shoe_particle_filter": dict(PF_CONFIG),
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "pf_delta_timing": "state_before_current_outcome",
            "base_margin": "historical_pf_delta",
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
    x7, pf_delta, residual, actual_b, shoes = make_training_arrays(records)

    if len(x7) < args.min_samples:
        raise SystemExit(
            f"need at least {args.min_samples} valid rows; got {len(x7)}"
        )

    validation = base.deterministic_validation_mask(
        shoes,
        fraction=args.validation_fraction,
    )
    train = ~validation

    booster = train_booster(
        x7[train],
        residual[train],
        pf_delta[train],
    )
    validation_metrics = evaluate_7d_with_margin(
        booster,
        x7[validation],
        pf_delta[validation],
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
            "validation gate rejected PF-base-margin 7D XGBoost model; "
            "use --force only for diagnostics"
        )

    final_booster = train_booster(
        x7,
        residual,
        pf_delta,
    )
    export_portable_bundle(
        final_booster,
        reference_x=x7,
        reference_margin=pf_delta,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=validation_metrics,
        training_rows=len(x7),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BBB PF base-margin 7D XGBoost residual trainer"
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
