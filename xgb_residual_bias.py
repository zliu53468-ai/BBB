#!/usr/bin/env python3
"""BBB Core-aware residual trainer with strict shoe-level walk-forward evaluation.

Production architecture is preserved:

    history -> 256D/V23 Core -> Core P(B) -> fixed 7D
    -> XGBoost residual correction -> adaptive Delta clip
    -> Tie shrink -> Platt calibration -> Final P(B) -> B/P

The XGBoost model is trained with binary log-loss and Core logit as base_margin.
Its probability output is converted back into a probability-space residual
relative to the displayed Core P(B), so the external Core + Residual contract
remains intact.

Evaluation is strictly chronological by whole shoe. No random split and no
row-level fallback are permitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xgboost as xgb

FEATURE_NAMES: tuple[str, ...] = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "sx_markov_p_same",
    "stage",
    "depth",
)

MODEL_TYPE = "xgb_core_margin_residual_v2"
SCHEMA_VERSION = 2
DEFAULT_MAX_DELTA = 0.10
DEFAULT_MIN_DELTA = 0.025
DEFAULT_CONFIDENCE_SPAN = 0.08
DEFAULT_RANDOM_STATE = 20260915
EPS = 1e-6

XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.025,
    "max_depth": 3,
    "min_child_weight": 12.0,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "alpha": 0.30,
    "lambda": 12.0,
    "max_delta_step": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "seed": DEFAULT_RANDOM_STATE,
    "nthread": 1,
}
DEFAULT_NUM_BOOST_ROUND = 180


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    arr = np.clip(arr, -40.0, 40.0)
    out = 1.0 / (1.0 + np.exp(-arr))
    if np.ndim(value) == 0:
        return float(out)
    return out


def logit(probability: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(probability, dtype=np.float64)
    arr = np.clip(arr, EPS, 1.0 - EPS)
    out = np.log(arr / (1.0 - arr))
    if np.ndim(probability) == 0:
        return float(out)
    return out


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        values: Iterable[Any] = [
            x for x in history.upper()
            if x in {"B", "P", "T"}
        ]
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
    return [
        "S" if values[i] == values[i - 1] else "X"
        for i in range(1, len(values))
    ]


def sx_markov_p_same(
    history: Sequence[str],
    *,
    window: int = 24,
    prior: float = 1.0,
) -> float:
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
        return {
            name: float(getattr(self, name))
            for name in FEATURE_NAMES
        }


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
    remaining_ratio = clip(
        (total_hands - (round_index - 1.0))
        / max(1.0, total_hands)
    )
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b), 0.0, 1.0),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        sx_markov_p_same=sx_markov_p_same(seq),
        stage=float(
            current_stage(seq)
            if stage is None
            else stage
        ),
        depth=float(
            current_depth(seq)
            if depth is None
            else depth
        ),
    )


def _feature_row(record: Mapping[str, Any]) -> dict[str, float]:
    if all(name in record for name in FEATURE_NAMES):
        return {
            name: float(record[name])
            for name in FEATURE_NAMES
        }
    features = build_features(
        core_p_b=float(
            record.get(
                "core_p_b",
                record.get("core_pb", 0.5),
            )
        ),
        history=(
            record.get("history")
            or record.get("history_fingerprint")
            or ""
        ),
        estimated_total_hands=float(
            record.get("estimated_total_hands", 60.0)
            or 60.0
        ),
        stage=(
            float(record["stage"])
            if record.get("stage") is not None
            else None
        ),
        depth=(
            float(record["depth"])
            if record.get("depth") is not None
            else None
        ),
    )
    return features.as_dict()


def _parse_outcome(record: Mapping[str, Any]) -> str:
    actual = str(
        record.get("actual_outcome")
        or record.get("actual")
        or ""
    ).upper().strip()
    if actual in {"B", "P", "T"}:
        return actual

    if (
        "actual_b" in record
        and record.get("actual_b") is not None
    ):
        return "B" if float(record["actual_b"]) >= 0.5 else "P"

    raise ValueError(
        "training row must contain actual_outcome B/P/T "
        "or actual_b 0/1"
    )


def _parse_timestamp(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("missing timestamp")
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e12:
            number /= 1000.0
        if not math.isfinite(number):
            raise ValueError("invalid timestamp")
        return number

    text = str(value).strip()
    try:
        number = float(text)
        if number > 1e12:
            number /= 1000.0
        return number
    except ValueError:
        pass

    parsed = datetime.fromisoformat(
        text.replace("Z", "+00:00")
    )
    return parsed.timestamp()


@dataclass
class PreparedRow:
    shoe_id: str
    timestamp: float
    round_index: int
    outcome: str
    features: np.ndarray
    core_p_b: float
    core_raw_p_b: float
    core_logit: float
    history_fingerprint: str
    core_x_256: np.ndarray | None


def load_training_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = (
            payload.get("rows")
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(rows, list):
            raise ValueError(
                "JSON training file must be a list "
                "or {'rows': [...]} bundle"
            )
        return [
            dict(row)
            for row in rows
            if isinstance(row, Mapping)
        ]

    if path.suffix.lower() == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            return [
                dict(row)
                for row in csv.DictReader(handle)
            ]

    raise ValueError("training input must be .json or .csv")


def prepare_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[PreparedRow]:
    prepared: list[PreparedRow] = []
    seen_keys: set[tuple[str, str]] = set()

    for idx, record in enumerate(records):
        shoe_id = str(record.get("shoe_id") or "").strip()
        if not shoe_id:
            raise ValueError(
                f"row {idx}: shoe_id is required; "
                "row-level fallback is forbidden"
            )

        timestamp_source = (
            record.get("created_at")
            if record.get("created_at") is not None
            else record.get("prediction_time")
        )
        timestamp = _parse_timestamp(timestamp_source)

        row = _feature_row(record)
        vector = np.asarray(
            [float(row[name]) for name in FEATURE_NAMES],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"row {idx}: non-finite feature value"
            )

        round_index = int(round(float(row["round_index"])))
        if round_index < 1:
            raise ValueError(
                f"row {idx}: invalid round_index"
            )

        outcome = _parse_outcome(record)
        core_p_b = clip(float(row["core_p_b"]))

        raw_value = record.get(
            "core_raw_p_b",
            record.get("core_raw_pb"),
        )
        core_raw_p_b = (
            clip(float(raw_value), EPS, 1.0 - EPS)
            if raw_value is not None
            else clip(core_p_b, EPS, 1.0 - EPS)
        )
        raw_logit = record.get("core_logit")
        core_margin = (
            float(raw_logit)
            if raw_logit is not None
            and math.isfinite(float(raw_logit))
            else float(logit(core_raw_p_b))
        )

        history_fingerprint = str(
            record.get("history_fingerprint") or ""
        )
        if history_fingerprint:
            key = (shoe_id, history_fingerprint)
            if key in seen_keys:
                raise ValueError(
                    f"duplicate prediction row: {key}"
                )
            seen_keys.add(key)

        core_x = record.get("core_x_256")
        core_x_256: np.ndarray | None = None
        if isinstance(core_x, list) and len(core_x) == 256:
            arr = np.asarray(core_x, dtype=np.float32)
            if np.all(np.isfinite(arr)):
                core_x_256 = arr

        prepared.append(
            PreparedRow(
                shoe_id=shoe_id,
                timestamp=timestamp,
                round_index=round_index,
                outcome=outcome,
                features=vector,
                core_p_b=core_p_b,
                core_raw_p_b=core_raw_p_b,
                core_logit=core_margin,
                history_fingerprint=history_fingerprint,
                core_x_256=core_x_256,
            )
        )

    if not prepared:
        raise ValueError("no valid training rows")

    by_shoe: dict[str, list[PreparedRow]] = {}
    for row in prepared:
        by_shoe.setdefault(row.shoe_id, []).append(row)

    shoe_start: dict[str, float] = {}
    for shoe_id, rows in by_shoe.items():
        rows.sort(
            key=lambda item: (
                item.round_index,
                item.timestamp,
            )
        )
        shoe_start[shoe_id] = min(
            item.timestamp for item in rows
        )
        previous_round = 0
        for item in rows:
            if item.round_index <= previous_round:
                raise ValueError(
                    f"shoe {shoe_id}: round_index must "
                    "strictly increase"
                )
            previous_round = item.round_index

    ordered_shoes = sorted(
        by_shoe,
        key=lambda shoe: (
            shoe_start[shoe],
            shoe,
        ),
    )
    shoe_rank = {
        shoe: rank
        for rank, shoe in enumerate(ordered_shoes)
    }

    prepared.sort(
        key=lambda item: (
            shoe_rank[item.shoe_id],
            item.round_index,
            item.timestamp,
        )
    )
    return prepared


def ordered_shoes(rows: Sequence[PreparedRow]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.shoe_id in seen:
            continue
        seen.add(row.shoe_id)
        result.append(row.shoe_id)
    return result


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_shoes: tuple[str, ...]
    calibration_shoes: tuple[str, ...]
    test_shoes: tuple[str, ...]


def build_walk_forward_folds(
    shoes: Sequence[str],
    *,
    min_train_shoes: int,
    calibration_shoes: int,
    test_shoes: int,
    step_shoes: int,
) -> list[WalkForwardFold]:
    n = len(shoes)
    min_train = max(2, int(min_train_shoes))
    cal_n = max(1, int(calibration_shoes))
    test_n = max(1, int(test_shoes))
    step = max(1, int(step_shoes))

    if n < min_train + cal_n + test_n:
        raise ValueError(
            "not enough shoes for strict walk-forward: "
            f"need >= {min_train + cal_n + test_n}, got {n}"
        )

    starts = list(
        range(
            min_train,
            n - cal_n - test_n + 1,
            step,
        )
    )
    starts = sorted(set(starts))

    folds: list[WalkForwardFold] = []
    for fold_index, train_end in enumerate(starts, start=1):
        folds.append(
            WalkForwardFold(
                fold=fold_index,
                train_shoes=tuple(shoes[:train_end]),
                calibration_shoes=tuple(
                    shoes[train_end: train_end + cal_n]
                ),
                test_shoes=tuple(
                    shoes[
                        train_end + cal_n:
                        train_end + cal_n + test_n
                    ]
                ),
            )
        )
    return folds


def select_rows(
    rows: Sequence[PreparedRow],
    shoes: Sequence[str],
) -> list[PreparedRow]:
    allowed = set(shoes)
    return [
        row
        for row in rows
        if row.shoe_id in allowed
    ]


def non_tie_arrays(
    rows: Sequence[PreparedRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row for row in rows
        if row.outcome in {"B", "P"}
    ]
    if not selected:
        raise ValueError("split has no non-tie rows")

    x = np.stack(
        [row.features for row in selected],
        axis=0,
    ).astype(np.float32)
    y = np.asarray(
        [1 if row.outcome == "B" else 0 for row in selected],
        dtype=np.float32,
    )
    core_pb = np.asarray(
        [row.core_p_b for row in selected],
        dtype=np.float64,
    )
    core_margin = np.asarray(
        [row.core_logit for row in selected],
        dtype=np.float64,
    )
    return x, y, core_pb, core_margin


def all_outcome_arrays(
    rows: Sequence[PreparedRow],
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("empty split")
    x = np.stack(
        [row.features for row in rows],
        axis=0,
    ).astype(np.float64)
    tie = np.asarray(
        [1.0 if row.outcome == "T" else 0.0 for row in rows],
        dtype=np.float64,
    )
    return x, tie


def build_booster(
    train_rows: Sequence[PreparedRow],
    *,
    num_boost_round: int,
    random_state: int,
) -> xgb.Booster:
    x_train, y_train, _, core_margin = non_tie_arrays(
        train_rows
    )
    dtrain = xgb.DMatrix(
        x_train,
        label=y_train,
        base_margin=core_margin,
        feature_names=list(FEATURE_NAMES),
    )
    params = dict(XGB_PARAMS)
    params["seed"] = int(random_state)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=max(1, int(num_boost_round)),
    )


def booster_probability(
    booster: xgb.Booster,
    x_values: np.ndarray,
    core_margin: np.ndarray,
) -> np.ndarray:
    dmatrix = xgb.DMatrix(
        np.asarray(x_values, dtype=np.float32),
        base_margin=np.asarray(core_margin, dtype=np.float64),
        feature_names=list(FEATURE_NAMES),
    )
    probability = booster.predict(dmatrix)
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )


def adaptive_delta_limit(
    core_pb: np.ndarray,
    *,
    min_delta: float,
    max_delta: float,
    confidence_span: float,
) -> np.ndarray:
    confidence = np.clip(
        np.abs(np.asarray(core_pb, dtype=np.float64) - 0.5)
        / max(float(confidence_span), EPS),
        0.0,
        1.0,
    )
    return (
        float(min_delta)
        + (float(max_delta) - float(min_delta))
        * confidence
    )


@dataclass
class TieLogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    baseline_rate: float
    high_threshold: float
    shrink_strength: float

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_values, dtype=np.float64)
        z = (x_arr - self.mean) / self.scale
        margin = z @ self.coef + self.intercept
        return np.asarray(sigmoid(margin), dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "standardized_logistic",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "baseline_rate": float(self.baseline_rate),
            "high_threshold": float(self.high_threshold),
            "shrink_strength": float(self.shrink_strength),
        }


def fit_tie_model(
    rows: Sequence[PreparedRow],
    *,
    l2: float,
    shrink_strength: float,
) -> TieLogisticModel:
    x_values, y = all_outcome_arrays(rows)
    mean = np.mean(x_values, axis=0)
    scale = np.std(x_values, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x_values - mean) / scale

    baseline = float(np.mean(y))
    baseline_safe = clip(baseline, 1e-4, 1.0 - 1e-4)
    beta = np.zeros(z.shape[1] + 1, dtype=np.float64)
    beta[0] = float(logit(baseline_safe))

    if 0.0 < baseline < 1.0 and len(y) >= 20:
        design = np.column_stack(
            [np.ones(len(z), dtype=np.float64), z]
        )
        reg = np.eye(design.shape[1], dtype=np.float64)
        reg[0, 0] = 0.0
        reg *= float(l2)

        for _ in range(80):
            margin = design @ beta
            probability = np.asarray(sigmoid(margin))
            weight = np.clip(
                probability * (1.0 - probability),
                1e-6,
                None,
            )
            gradient = (
                design.T @ (probability - y)
                + reg @ beta
            )
            hessian = (
                design.T @ (weight[:, None] * design)
                + reg
                + np.eye(design.shape[1]) * 1e-8
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            beta -= step
            if float(np.linalg.norm(step)) < 1e-7:
                break

    model = TieLogisticModel(
        mean=mean,
        scale=scale,
        coef=beta[1:].copy(),
        intercept=float(beta[0]),
        baseline_rate=baseline,
        high_threshold=min(0.35, baseline + 0.05),
        shrink_strength=clip(
            float(shrink_strength),
            0.0,
            1.0,
        ),
    )

    train_probability = model.predict(x_values)
    q90 = float(np.quantile(train_probability, 0.90))
    model.high_threshold = max(
        baseline + 0.02,
        q90,
    )
    model.high_threshold = min(
        0.40,
        model.high_threshold,
    )
    return model


def apply_tie_shrink(
    probability_b: np.ndarray,
    tie_probability: np.ndarray,
    tie_model: TieLogisticModel,
) -> np.ndarray:
    baseline = float(tie_model.baseline_rate)
    high = max(
        baseline + 1e-6,
        float(tie_model.high_threshold),
    )
    excess = np.clip(
        (np.asarray(tie_probability, dtype=np.float64) - baseline)
        / (high - baseline),
        0.0,
        1.0,
    )
    shrink = (
        float(tie_model.shrink_strength)
        * excess
    )
    p = np.asarray(probability_b, dtype=np.float64)
    return np.clip(
        0.5 + (p - 0.5) * (1.0 - shrink),
        EPS,
        1.0 - EPS,
    )


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float

    def apply(self, probability: np.ndarray) -> np.ndarray:
        margin = np.asarray(logit(probability), dtype=np.float64)
        return np.asarray(
            sigmoid(self.slope * margin + self.intercept),
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "platt",
            "slope": float(self.slope),
            "intercept": float(self.intercept),
        }


def fit_platt(
    probability: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float = 0.10,
) -> PlattCalibration:
    p = np.asarray(probability, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if (
        len(target) < 10
        or len(np.unique(target)) < 2
    ):
        return PlattCalibration(1.0, 0.0)

    x_value = np.asarray(logit(p), dtype=np.float64)
    beta = np.asarray([1.0, 0.0], dtype=np.float64)

    for _ in range(80):
        margin = beta[0] * x_value + beta[1]
        pred = np.asarray(sigmoid(margin), dtype=np.float64)
        weight = np.clip(
            pred * (1.0 - pred),
            1e-6,
            None,
        )

        gradient = np.asarray([
            np.sum((pred - target) * x_value)
            + regularization * (beta[0] - 1.0),
            np.sum(pred - target)
            + regularization * beta[1],
        ])
        hessian = np.asarray([
            [
                np.sum(weight * x_value * x_value)
                + regularization,
                np.sum(weight * x_value),
            ],
            [
                np.sum(weight * x_value),
                np.sum(weight) + regularization,
            ],
        ]) + np.eye(2) * 1e-8

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return PlattCalibration(1.0, 0.0)

        beta -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break

    if not np.all(np.isfinite(beta)) or beta[0] <= 0.05:
        return PlattCalibration(1.0, 0.0)

    return PlattCalibration(
        slope=float(beta[0]),
        intercept=float(beta[1]),
    )


def pipeline_probability(
    booster: xgb.Booster,
    rows: Sequence[PreparedRow],
    tie_model: TieLogisticModel,
    calibration: PlattCalibration | None,
    *,
    min_delta: float,
    max_delta: float,
    confidence_span: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_values, _, core_pb, core_margin = non_tie_arrays(rows)
    xgb_probability = booster_probability(
        booster,
        x_values,
        core_margin,
    )

    raw_delta = xgb_probability - core_pb
    delta_limit = adaptive_delta_limit(
        core_pb,
        min_delta=min_delta,
        max_delta=max_delta,
        confidence_span=confidence_span,
    )
    delta = np.clip(
        raw_delta,
        -delta_limit,
        delta_limit,
    )
    residual_probability = np.clip(
        core_pb + delta,
        EPS,
        1.0 - EPS,
    )

    tie_probability = tie_model.predict(x_values)
    tie_adjusted = apply_tie_shrink(
        residual_probability,
        tie_probability,
        tie_model,
    )

    final_probability = (
        calibration.apply(tie_adjusted)
        if calibration is not None
        else tie_adjusted
    )

    return np.clip(
        final_probability,
        EPS,
        1.0 - EPS,
    ), {
        "core_probability": core_pb,
        "xgb_probability": xgb_probability,
        "raw_delta": raw_delta,
        "delta_limit": delta_limit,
        "delta": delta,
        "residual_probability": residual_probability,
        "tie_probability": tie_probability,
        "tie_adjusted_probability": tie_adjusted,
    }


def direction_accuracy(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
) -> float:
    return float(
        np.mean(
            (np.asarray(probability_b) > 0.5)
            == (np.asarray(actual_b) > 0.5)
        )
    )


def log_loss(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
) -> float:
    p = np.clip(
        np.asarray(probability_b, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    y = np.asarray(actual_b, dtype=np.float64)
    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def brier(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
) -> float:
    p = np.asarray(probability_b, dtype=np.float64)
    y = np.asarray(actual_b, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def probability_bins(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    p = np.asarray(probability_b, dtype=np.float64)
    y = np.asarray(actual_b, dtype=np.float64)
    order = np.argsort(p)
    groups = np.array_split(
        order,
        min(max(1, int(n_bins)), len(order)),
    )
    result: list[dict[str, float]] = []
    for group in groups:
        if len(group) == 0:
            continue
        pg = p[group]
        yg = y[group]
        result.append({
            "count": float(len(group)),
            "p_min": float(np.min(pg)),
            "p_max": float(np.max(pg)),
            "mean_predicted_p_b": float(np.mean(pg)),
            "actual_b_rate": float(np.mean(yg)),
            "direction_accuracy": direction_accuracy(pg, yg),
        })
    return result


def ece_equal_frequency(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    bins = probability_bins(
        probability_b,
        actual_b,
        n_bins=n_bins,
    )
    total = sum(item["count"] for item in bins)
    if total <= 0:
        return 0.0
    return float(
        sum(
            item["count"]
            / total
            * abs(
                item["mean_predicted_p_b"]
                - item["actual_b_rate"]
            )
            for item in bins
        )
    )


def binary_metrics(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
) -> dict[str, Any]:
    p = np.asarray(probability_b, dtype=np.float64)
    y = np.asarray(actual_b, dtype=np.float64)
    return {
        "samples": int(len(y)),
        "accuracy_ex_tie": direction_accuracy(p, y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "ece_equal_frequency": ece_equal_frequency(p, y),
        "probability_bins": probability_bins(p, y),
    }


def tie_metrics(
    tie_probability: np.ndarray,
    tie_actual: np.ndarray,
) -> dict[str, float]:
    p = np.asarray(tie_probability, dtype=np.float64)
    y = np.asarray(tie_actual, dtype=np.float64)
    return {
        "samples": float(len(y)),
        "actual_tie_rate": float(np.mean(y)),
        "mean_predicted_tie": float(np.mean(p)),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
    }


def audit_core_256(
    rows: Sequence[PreparedRow],
    train_shoes: Sequence[str],
) -> dict[str, Any]:
    allowed = set(train_shoes)
    selected = [
        row for row in rows
        if (
            row.shoe_id in allowed
            and row.outcome in {"B", "P"}
            and row.core_x_256 is not None
        )
    ]
    if len(selected) < 20:
        return {
            "available": False,
            "reason": "need >=20 rows with core_x_256",
            "rows": len(selected),
            "automatic_pruning": False,
        }

    x_values = np.stack(
        [row.core_x_256 for row in selected],
        axis=0,
    ).astype(np.float64)
    y = np.asarray(
        [1.0 if row.outcome == "B" else 0.0 for row in selected],
        dtype=np.float64,
    )
    core_margin = np.asarray(
        [row.core_logit for row in selected],
        dtype=np.float64,
    )

    std = np.std(x_values, axis=0)
    low_variance = np.flatnonzero(std < 1e-6).tolist()
    valid = std >= 1e-6

    high_corr: list[dict[str, Any]] = []
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) >= 2:
        corr = np.corrcoef(x_values[:, valid_idx], rowvar=False)
        for i in range(len(valid_idx)):
            for j in range(i + 1, len(valid_idx)):
                value = float(corr[i, j])
                if math.isfinite(value) and abs(value) >= 0.98:
                    high_corr.append({
                        "feature_a": int(valid_idx[i]),
                        "feature_b": int(valid_idx[j]),
                        "correlation": value,
                    })
        high_corr.sort(
            key=lambda item: abs(item["correlation"]),
            reverse=True,
        )
        high_corr = high_corr[:50]

    feature_names = [f"x{i}" for i in range(256)]
    dmatrix = xgb.DMatrix(
        x_values.astype(np.float32),
        label=y.astype(np.float32),
        base_margin=core_margin,
        feature_names=feature_names,
    )
    params = dict(XGB_PARAMS)
    params.update({
        "max_depth": 2,
        "eta": 0.03,
        "min_child_weight": 20.0,
    })
    diagnostic = xgb.train(
        params,
        dmatrix,
        num_boost_round=80,
    )
    gain = diagnostic.get_score(importance_type="gain")
    ranked = sorted(
        (
            {
                "feature": int(name[1:]),
                "gain": float(value),
            }
            for name, value in gain.items()
            if name.startswith("x")
        ),
        key=lambda item: item["gain"],
        reverse=True,
    )[:30]

    return {
        "available": True,
        "rows": len(selected),
        "low_variance_feature_indices": low_variance,
        "high_correlation_pairs": high_corr,
        "top_gain_features": ranked,
        "automatic_pruning": False,
        "note": (
            "Audit only. The 256D/V23 Core dimension is preserved; "
            "no feature is removed automatically."
        ),
    }


def _tree_leaf(
    tree: Mapping[str, Any],
    vector: Sequence[float],
) -> float:
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

        value = (
            float(np.float32(vector[index]))
            if 0 <= index < len(vector)
            else math.nan
        )
        split_condition = float(
            np.float32(
                node.get("split_condition", 0.0)
            )
        )
        next_id = (
            node.get("missing")
            if not math.isfinite(value)
            else (
                node.get("yes")
                if value < split_condition
                else node.get("no")
            )
        )
        children = node.get("children") or []
        found = next(
            (
                child
                for child in children
                if int(child.get("nodeid", -999))
                == int(next_id)
            ),
            None,
        )
        if found is None:
            return 0.0
        node = found
    return 0.0


def portable_xgb_payload(
    booster: xgb.Booster,
    reference_rows: Sequence[PreparedRow],
) -> dict[str, Any]:
    x_values, _, _, core_margin = non_tie_arrays(reference_rows)
    trees = [
        json.loads(text)
        for text in booster.get_dump(dump_format="json")
    ]

    reference = np.asarray(x_values[0], dtype=np.float64)
    tree_sum = sum(
        _tree_leaf(tree, reference)
        for tree in trees
    )
    dref = xgb.DMatrix(
        reference.reshape(1, -1).astype(np.float32),
        base_margin=np.asarray([core_margin[0]], dtype=np.float64),
        feature_names=list(FEATURE_NAMES),
    )
    native_margin = float(
        booster.predict(
            dref,
            output_margin=True,
        )[0]
    )
    margin_offset = (
        native_margin
        - float(core_margin[0])
        - tree_sum
    )

    sample_count = min(64, len(x_values))
    dsample = xgb.DMatrix(
        x_values[:sample_count].astype(np.float32),
        base_margin=core_margin[:sample_count],
        feature_names=list(FEATURE_NAMES),
    )
    native = booster.predict(
        dsample,
        output_margin=True,
    )
    for i in range(sample_count):
        portable = (
            float(core_margin[i])
            + margin_offset
            + sum(
                _tree_leaf(tree, x_values[i])
                for tree in trees
            )
        )
        if abs(portable - float(native[i])) > 2e-5:
            raise RuntimeError(
                "portable XGBoost margin mismatch: "
                f"{portable} vs {native[i]}"
            )

    return {
        "margin_offset": float(margin_offset),
        "trees": trees,
        "params": {
            "objective": "binary:logistic",
            "num_boost_round": len(trees),
            "eta": float(XGB_PARAMS["eta"]),
            "max_depth": int(XGB_PARAMS["max_depth"]),
            "min_child_weight": float(XGB_PARAMS["min_child_weight"]),
            "reg_alpha": float(XGB_PARAMS["alpha"]),
            "reg_lambda": float(XGB_PARAMS["lambda"]),
            "max_delta_step": float(XGB_PARAMS["max_delta_step"]),
        },
    }


def fit_and_evaluate_fold(
    rows: Sequence[PreparedRow],
    fold: WalkForwardFold,
    *,
    num_boost_round: int,
    random_state: int,
    min_delta: float,
    max_delta: float,
    confidence_span: float,
    tie_l2: float,
    tie_shrink_strength: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = select_rows(rows, fold.train_shoes)
    calibration_rows = select_rows(
        rows,
        fold.calibration_shoes,
    )
    test_rows = select_rows(rows, fold.test_shoes)

    booster = build_booster(
        train_rows,
        num_boost_round=num_boost_round,
        random_state=random_state,
    )
    tie_model = fit_tie_model(
        train_rows,
        l2=tie_l2,
        shrink_strength=tie_shrink_strength,
    )

    cal_x, cal_y, _, _ = non_tie_arrays(calibration_rows)
    cal_uncalibrated, _ = pipeline_probability(
        booster,
        calibration_rows,
        tie_model,
        None,
        min_delta=min_delta,
        max_delta=max_delta,
        confidence_span=confidence_span,
    )
    calibration = fit_platt(
        cal_uncalibrated,
        cal_y,
    )

    test_x, test_y, test_core_pb, _ = non_tie_arrays(test_rows)
    del test_x
    final_probability, details = pipeline_probability(
        booster,
        test_rows,
        tie_model,
        calibration,
        min_delta=min_delta,
        max_delta=max_delta,
        confidence_span=confidence_span,
    )

    test_all_x, test_tie_actual = all_outcome_arrays(test_rows)
    test_tie_probability = tie_model.predict(test_all_x)

    fold_report = {
        "fold": fold.fold,
        "train_shoes": list(fold.train_shoes),
        "calibration_shoes": list(fold.calibration_shoes),
        "test_shoes": list(fold.test_shoes),
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "test_rows": len(test_rows),
        "core": binary_metrics(test_core_pb, test_y),
        "final": binary_metrics(final_probability, test_y),
        "tie_model": tie_metrics(
            test_tie_probability,
            test_tie_actual,
        ),
        "mean_abs_delta": float(
            np.mean(np.abs(details["delta"]))
        ),
        "mean_delta_limit": float(
            np.mean(details["delta_limit"])
        ),
        "platt": calibration.as_dict(),
    }

    artifacts = {
        "booster": booster,
        "tie_model": tie_model,
        "calibration": calibration,
        "test_y": test_y,
        "test_core_pb": test_core_pb,
        "test_final_probability": final_probability,
        "test_tie_actual": test_tie_actual,
        "test_tie_probability": test_tie_probability,
        "train_rows": train_rows,
        "calibration_rows": calibration_rows,
        "test_rows": test_rows,
    }
    return fold_report, artifacts


def aggregate_oos(
    fold_artifacts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    y = np.concatenate(
        [item["test_y"] for item in fold_artifacts]
    )
    core = np.concatenate(
        [item["test_core_pb"] for item in fold_artifacts]
    )
    final = np.concatenate(
        [
            item["test_final_probability"]
            for item in fold_artifacts
        ]
    )
    tie_actual = np.concatenate(
        [item["test_tie_actual"] for item in fold_artifacts]
    )
    tie_probability = np.concatenate(
        [
            item["test_tie_probability"]
            for item in fold_artifacts
        ]
    )

    return {
        "core": binary_metrics(core, y),
        "final": binary_metrics(final, y),
        "tie_model": tie_metrics(
            tie_probability,
            tie_actual,
        ),
    }


def export_model_bundle(
    *,
    booster: xgb.Booster,
    tie_model: TieLogisticModel,
    calibration: PlattCalibration,
    reference_rows: Sequence[PreparedRow],
    output_path: Path,
    min_delta: float,
    max_delta: float,
    confidence_span: float,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "training_objective": "binary_logloss_with_core_logit_base_margin",
        "xgb": portable_xgb_payload(
            booster,
            reference_rows,
        ),
        "adaptive_clip": {
            "min_delta": float(min_delta),
            "max_delta": float(max_delta),
            "confidence_span": float(confidence_span),
            "formula": (
                "limit=min_delta+(max_delta-min_delta)"
                "*clip(abs(core_p_b-0.5)/confidence_span,0,1)"
            ),
        },
        "tie_model": tie_model.as_dict(),
        "calibration": calibration.as_dict(),
        "decision_rule": "B if calibrated_final_p_b > 0.50 else P",
        "no_pass": True,
        "evaluation": {
            "protocol": "strict_chronological_shoe_walk_forward",
            **dict(report),
        },
    }
    output_path.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    rows = prepare_rows(records)
    shoes = ordered_shoes(rows)

    folds = build_walk_forward_folds(
        shoes,
        min_train_shoes=args.min_train_shoes,
        calibration_shoes=args.calibration_shoes,
        test_shoes=args.test_shoes,
        step_shoes=args.step_shoes,
    )

    fold_reports: list[dict[str, Any]] = []
    fold_artifacts: list[dict[str, Any]] = []
    for fold in folds:
        report, artifacts = fit_and_evaluate_fold(
            rows,
            fold,
            num_boost_round=args.num_boost_round,
            random_state=args.random_state,
            min_delta=args.min_delta,
            max_delta=args.max_delta,
            confidence_span=args.confidence_span,
            tie_l2=args.tie_l2,
            tie_shrink_strength=args.tie_shrink_strength,
        )
        fold_reports.append(report)
        fold_artifacts.append(artifacts)

    aggregate = aggregate_oos(fold_artifacts)
    core_metrics = aggregate["core"]
    final_metrics = aggregate["final"]

    accepted = (
        final_metrics["log_loss"]
        <= core_metrics["log_loss"] + args.max_logloss_regression
        and final_metrics["brier"]
        <= core_metrics["brier"] + args.max_brier_regression
        and final_metrics["accuracy_ex_tie"]
        >= core_metrics["accuracy_ex_tie"] - args.max_accuracy_regression
    )

    final_fold = folds[-1]
    final_artifacts = fold_artifacts[-1]
    core_256_audit = audit_core_256(
        rows,
        final_fold.train_shoes,
    )

    full_report = {
        "protocol": {
            "split": "strict_chronological_shoe_walk_forward",
            "random_split": False,
            "primary_target": "P(B | non-tie)",
            "folds": len(folds),
            "min_train_shoes": args.min_train_shoes,
            "calibration_shoes": args.calibration_shoes,
            "test_shoes": args.test_shoes,
            "step_shoes": args.step_shoes,
        },
        "folds": fold_reports,
        "aggregate_oos": aggregate,
        "core_256_audit": core_256_audit,
        "accepted": accepted,
    }

    print(
        json.dumps(
            full_report,
            ensure_ascii=False,
            indent=2,
        )
    )

    report_path = (
        Path(args.report_output)
        if args.report_output
        else Path(args.output).with_suffix(".report.json")
    )
    report_path.write_text(
        json.dumps(
            full_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not accepted and not args.force:
        raise SystemExit(
            "walk-forward gate rejected model; "
            "use --force only for diagnostics"
        )

    export_model_bundle(
        booster=final_artifacts["booster"],
        tie_model=final_artifacts["tie_model"],
        calibration=final_artifacts["calibration"],
        reference_rows=final_artifacts["train_rows"],
        output_path=Path(args.output),
        min_delta=args.min_delta,
        max_delta=args.max_delta,
        confidence_span=args.confidence_span,
        report={
            "aggregate_oos": aggregate,
            "final_fold": fold_reports[-1],
            "core_256_audit": core_256_audit,
        },
    )
    print(f"wrote {args.output}")
    print(f"wrote {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "BBB strict walk-forward Core-margin "
            "XGBoost residual trainer"
        )
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument(
        "--output",
        default="residual_bias_model.json",
    )
    train.add_argument("--report-output", default="")
    train.add_argument(
        "--min-train-shoes",
        type=int,
        default=8,
    )
    train.add_argument(
        "--calibration-shoes",
        type=int,
        default=2,
    )
    train.add_argument(
        "--test-shoes",
        type=int,
        default=2,
    )
    train.add_argument(
        "--step-shoes",
        type=int,
        default=2,
    )
    train.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
    )
    train.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )
    train.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
    )
    train.add_argument(
        "--max-delta",
        type=float,
        default=DEFAULT_MAX_DELTA,
    )
    train.add_argument(
        "--confidence-span",
        type=float,
        default=DEFAULT_CONFIDENCE_SPAN,
    )
    train.add_argument(
        "--tie-l2",
        type=float,
        default=2.0,
    )
    train.add_argument(
        "--tie-shrink-strength",
        type=float,
        default=0.35,
    )
    train.add_argument(
        "--max-logloss-regression",
        type=float,
        default=0.0,
    )
    train.add_argument(
        "--max-brier-regression",
        type=float,
        default=0.0,
    )
    train.add_argument(
        "--max-accuracy-regression",
        type=float,
        default=0.002,
    )
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
