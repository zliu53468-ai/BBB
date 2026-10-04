#!/usr/bin/env python3
"""Direct 57D XGBoost probability layer for the frozen BBB core.

The 256D JavaScript cores remain unchanged.  Their ``Core P(B)`` is only one
input feature here; it is never added to a residual prediction.

Feature order (fixed):
    [Core P(B), progress^3] + [original 7D without duplicate Core P(B)]
    + [physics 48D] + [physics uncertainty] = 57D

Target (fixed):
    actual_B, where Player=0 and Banker=1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

try:  # Keep vector-only tests usable before requirements-xgb.txt is installed.
    from xgboost import DMatrix, XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean dev envs
    DMatrix = None  # type: ignore[assignment,misc]
    XGBClassifier = None  # type: ignore[assignment,misc]

from physics_feature_extractor import (
    PHYSICS_DIM,
    PHYSICS_FEATURE_NAMES,
    RANK_LABELS,
    SUITS,
    PhysicsFeatureExtractor,
    apply_uncertainty_calibration,
)

MODEL_TYPE = "xgb_final_probability_classifier"
RANDOM_STATE = 20260923
FEATURE_DIM = 57
PROBABILITY_BOUNDS = (0.40, 0.60)
EARLY_PROBABILITY_BOUNDS = (0.45, 0.55)
LATE_CLEAN_PROBABILITY_BOUNDS = (0.35, 0.65)
SNAPSHOT_SCHEMA_VERSION = 6
PHYSICS_PROBABILITY_TOLERANCE = 1e-4
PHYSICS_NOISE_LOW_THRESHOLD = 0.78
PHYSICS_RANK_CONSUMPTION_TOLERANCE = 0.50
DEFAULT_MIN_EV = {"early": 0.020, "middle": 0.010, "late": 0.005}
PREFERRED_SKIP_RATE_INCREASE = 0.05
MAX_SKIP_RATE_INCREASE = 0.05
CALIBRATION_MIN_BRIER_IMPROVEMENT = 1e-4
ISOTONIC_MIN_ROWS = 1000
MAX_SMOOTHING_BRIER_INCREASE = 0.0005
MAX_MODEL_BRIER_REGRESSION = 0.0015
MIN_PRIMARY_GAIN = 1e-6
EMA_ALPHA_RANGES = {"early": (0.35, 0.45), "middle": (0.50, 0.60), "late": (0.65, 0.75)}
EMA_PROFILES: dict[str, dict[str, Any]] = {
    "off": {"method": "dynamic_post_clip_ema", "profile": "off", "enabled": False},
    "stable": {"method": "dynamic_post_clip_ema", "profile": "stable", "enabled": True, "early_alpha": 0.375, "middle_alpha": 0.525, "late_alpha": 0.675, "noise_gain": 0.05},
    "balanced": {"method": "dynamic_post_clip_ema", "profile": "balanced", "enabled": True, "early_alpha": 0.40, "middle_alpha": 0.55, "late_alpha": 0.70, "noise_gain": 0.05},
    "responsive": {"method": "dynamic_post_clip_ema", "profile": "responsive", "enabled": True, "early_alpha": 0.425, "middle_alpha": 0.575, "late_alpha": 0.725, "noise_gain": 0.05},
}
MAX_NOISE_EV_PENALTY = 0.001
MIDDLE_EV_RELIEF = 0.001
LATE_EV_RELIEF = 0.001
MIDDLE_SOFT_BAND = 0.001
LATE_SOFT_BAND = 0.0015
MIN_SOFT_CONFIDENCE = 0.0005
DECISION_POLICY_PROFILES: dict[str, dict[str, Any]] = {
    # Current hard three-stage EV gate; this remains the comparison baseline.
    "hard_ev": {"profile": "hard_ev", "enabled": False},
    # More selective candidate. It is only deployed when chronological hold-out
    # gates pass, including the existing +5pp maximum Skip increase.
    "strict_selective_entry": {
        "profile": "strict_selective_entry_v1",
        "enabled": True,
        "min_skip_increase": 0.02,
        "noise_threshold": 0.68,
        "max_noise_ev_penalty": 0.006,
        "middle_relief": 0.0,
        "late_relief": 0.0,
        "middle_soft_band": 0.0,
        "late_soft_band": 0.0,
        "min_confidence": 0.001,
        "confidence_band": {
            "early": 0.028, "middle": 0.020, "late": 0.014,
            "noise_reference": 0.50, "noise_gain": 0.018,
            "clean_late_relief": 0.003, "minimum": 0.010, "maximum": 0.050,
            "strong_margin": {"early": 0.007, "middle": 0.005, "late": 0.003},
        },
        "volume_guard": {
            "enabled": True, "window": 16, "min_history": 8,
            "target_action_rate": {"early": 0.30, "middle": 0.36, "late": 0.42},
            "expected_correct_floor": 0.95, "band_relief": 0.006, "ev_relief": 0.004,
        },
    },
}
LEGACY_XGB_PARAMETERS = {"n_estimators": 320, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.82, "reg_alpha": 0.35, "reg_lambda": 12.0}
XGB_TUNING_CANDIDATES: tuple[dict[str, Any], ...] = (
    LEGACY_XGB_PARAMETERS,
    {"n_estimators": 420, "max_depth": 2, "learning_rate": 0.020, "min_child_weight": 18, "subsample": 0.80, "colsample_bytree": 0.75, "reg_alpha": 0.50, "reg_lambda": 16.0},
    {"n_estimators": 600, "max_depth": 2, "learning_rate": 0.012, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.25, "reg_lambda": 12.0},
    {"n_estimators": 520, "max_depth": 2, "learning_rate": 0.015, "min_child_weight": 24, "subsample": 0.90, "colsample_bytree": 0.70, "reg_alpha": 0.75, "reg_lambda": 20.0},
    {"n_estimators": 360, "max_depth": 3, "learning_rate": 0.020, "min_child_weight": 18, "subsample": 0.80, "colsample_bytree": 0.75, "reg_alpha": 0.75, "reg_lambda": 18.0},
    {"n_estimators": 480, "max_depth": 3, "learning_rate": 0.015, "min_child_weight": 24, "subsample": 0.90, "colsample_bytree": 0.80, "reg_alpha": 1.00, "reg_lambda": 24.0},
    {"n_estimators": 700, "max_depth": 1, "learning_rate": 0.015, "min_child_weight": 10, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.25, "reg_lambda": 12.0},
)
ORIGINAL_7D_FEATURE_NAMES = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "sx_markov_p_same",
    "stage",
    "depth",
)

FEATURE_NAMES: tuple[str, ...] = (
    ("core_p_b_external", "shoe_progress_weight")
    + tuple(f"original7_{name}" for name in ORIGINAL_7D_FEATURE_NAMES[1:])
    + PHYSICS_FEATURE_NAMES
    + ("physics_noise_score",)
)
assert len(FEATURE_NAMES) == FEATURE_DIM

NEXT_CARD_COUNT_LABELS = ("4", "5", "6")
_NEXT_CARD_COUNT_NAMES = tuple(f"cards_p{label}" for label in NEXT_CARD_COUNT_LABELS)
_NEXT_RANK_CONSUMPTION_NAMES = tuple(f"next_rank_expected_{label}" for label in RANK_LABELS)
_NEXT_SUIT_RATIO_NAMES = tuple(f"next_suit_ratio_{suit}" for suit in SUITS)
_PHYSICS_INDEX = {name: index for index, name in enumerate(PHYSICS_FEATURE_NAMES)}
assert all(name in _PHYSICS_INDEX for name in _NEXT_CARD_COUNT_NAMES + _NEXT_RANK_CONSUMPTION_NAMES + _NEXT_SUIT_RATIO_NAMES)


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


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _probability_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("probability_bounds must contain exactly (min, max)")
    lo, hi = (float(bounds[0]), float(bounds[1]))
    if not (0.0 <= lo <= 0.5 <= hi <= 1.0):
        raise ValueError("probability bounds must satisfy 0 <= min <= .5 <= max <= 1")
    return lo, hi


def _physics_vector(physics_48d: Sequence[float]) -> np.ndarray:
    """Validate the already-produced 48D physical forecast without simulating it."""
    physics = np.asarray(physics_48d, dtype=np.float32).reshape(-1)
    if physics.size != PHYSICS_DIM:
        raise ValueError(f"physics_48d must contain exactly {PHYSICS_DIM} values")
    if not np.all(np.isfinite(physics)):
        raise ValueError("physics_48d must contain only finite values")
    return physics


def unpack_physics_forecast(physics_48d: Sequence[float]) -> dict[str, Any]:
    """Decode next-hand physical estimates embedded in the existing 48D vector.

    This is a pure view of the supplied feature block: it does not invoke, train,
    or alter any simulation/MCMC component.  Suit ratios are also converted to
    expected cards by multiplying them by the decoded next-hand card expectation.
    """
    physics = _physics_vector(physics_48d)
    value = lambda name: float(physics[_PHYSICS_INDEX[name]])

    card_count_probabilities = {
        f"{label}_cards": value(f"cards_p{label}")
        for label in NEXT_CARD_COUNT_LABELS
    }
    expected_next_card_count = sum(
        int(label) * card_count_probabilities[f"{label}_cards"]
        for label in NEXT_CARD_COUNT_LABELS
    )
    rank_expected_consumption = {
        label: value(f"next_rank_expected_{label}")
        for label in RANK_LABELS
    }
    suit_consumption_ratios = {
        suit: value(f"next_suit_ratio_{suit}")
        for suit in SUITS
    }
    suit_expected_consumption = {
        suit: expected_next_card_count * ratio
        for suit, ratio in suit_consumption_ratios.items()
    }
    return {
        "next_card_count_probabilities": card_count_probabilities,
        "expected_next_card_count": float(expected_next_card_count),
        "next_rank_expected_consumption": rank_expected_consumption,
        "next_suit_consumption_ratios": suit_consumption_ratios,
        "next_suit_expected_consumption": suit_expected_consumption,
    }


def physics_integrity_report(physics_48d: Sequence[float]) -> dict[str, Any]:
    """Report whether the supplied 48D physical forecast is self-consistent.

    The bridge deliberately does not alter or simulate the physics block.  This
    report is diagnostic metadata for snapshots and model validation: card-count
    and suit distributions should normalise to one, while the A-K consumption
    total should be close to the decoded expected next-hand card count.
    """
    forecast = unpack_physics_forecast(physics_48d)
    card_probability_sum = float(sum(forecast["next_card_count_probabilities"].values()))
    suit_ratio_sum = float(sum(forecast["next_suit_consumption_ratios"].values()))
    expected_card_count = float(forecast["expected_next_card_count"])
    rank_expected_total = float(sum(forecast["next_rank_expected_consumption"].values()))
    rank_total_gap = abs(rank_expected_total - expected_card_count)
    rank_tolerance = max(
        PHYSICS_RANK_CONSUMPTION_TOLERANCE,
        expected_card_count * 0.10,
    )
    checks = {
        "card_count_distribution": abs(card_probability_sum - 1.0) <= PHYSICS_PROBABILITY_TOLERANCE,
        "suit_ratio_distribution": abs(suit_ratio_sum - 1.0) <= PHYSICS_PROBABILITY_TOLERANCE,
        "rank_consumption_total": rank_total_gap <= rank_tolerance,
    }
    return {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "card_count_probability_sum": card_probability_sum,
        "expected_next_card_count": expected_card_count,
        "rank_expected_consumption_total": rank_expected_total,
        "rank_expected_total_gap": rank_total_gap,
        "rank_expected_total_tolerance": rank_tolerance,
        "suit_ratio_sum": suit_ratio_sum,
    }


def physics_noise_score(
    physics_48d: Sequence[float],
    round_index: float = 70.0,
    calibration: Mapping[str, Any] | None = None,
) -> float:
    """Compressed physics uncertainty, optionally calibrated to empirical 48D residuals."""
    physics = _physics_vector(physics_48d).astype(np.float64)

    def entropy(block: np.ndarray) -> float:
        probability = np.clip(block, 1e-8, None)
        probability /= probability.sum()
        return float(-np.sum(probability * np.log(probability)) / math.log(len(probability)))

    winner = np.sort(physics[23:26])[::-1]
    winner_gap = float(winner[0] - winner[1])
    density_gap = abs(_clip(physics[44]) - _clip(physics[45]))
    density_ambiguity = 1.0 - min(1.0, density_gap / 0.25)
    raw = _clip(
        0.10 * entropy(physics[0:3])
        + 0.075 * entropy(physics[3:13])
        + 0.075 * entropy(physics[13:23])
        + 0.15 * entropy(physics[23:26])
        + 0.25 * entropy(physics[26:39])
        + 0.15 * entropy(physics[39:43])
        + 0.075 * (1.0 - winner_gap)
        + 0.125 * density_ambiguity
    )
    compressed = 0.50 + 0.35 * math.tanh((raw - 0.75) / 0.20)
    influence = 0.35 if round_index <= 40 else 0.65 if round_index <= 50 else 1.0
    proxy = _clip(0.50 + (compressed - 0.50) * influence)
    return float(apply_uncertainty_calibration([proxy], calibration)[0])


def dynamic_probability_bounds(round_index: float, noise_score: float) -> tuple[float, float]:
    if round_index <= 40:
        return EARLY_PROBABILITY_BOUNDS
    if round_index > 50 and noise_score <= PHYSICS_NOISE_LOW_THRESHOLD:
        return LATE_CLEAN_PROBABILITY_BOUNDS
    return PROBABILITY_BOUNDS


def build_56d_feature_matrix(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    noise_calibration: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Build the compatibility-named fixed feature bridge as ``(1, 57)``.

    ``original_7d`` is intentionally not recalculated or reordered.  The
    physics block is supplied by the caller so no MCMC/simulation work happens
    in this bridge function.
    """
    core = np.asarray(core_pb, dtype=np.float32).reshape(-1)
    original = np.asarray(original_7d, dtype=np.float32).reshape(-1)
    physics = _physics_vector(physics_48d)

    if core.size != 1:
        raise ValueError("core_pb must contain exactly one probability")
    if not 0.0 <= float(core[0]) <= 1.0:
        raise ValueError("core_pb must be within [0, 1]")
    if original.size != 7:
        raise ValueError("original_7d must contain exactly 7 values")
    if not np.all(np.isfinite(original)):
        raise ValueError("feature blocks must contain only finite values")

    progress_w = np.float32((float(original[1]) / 70.0) ** 3)
    noise = np.float32(physics_noise_score(physics, float(original[1]), noise_calibration))
    merged = np.hstack((core, [progress_w], original[1:], physics, [noise])).astype(np.float32, copy=False)
    if merged.size != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {merged.size}")
    return merged.reshape(1, FEATURE_DIM)


def build_xgboost_classifier(
    *,
    random_state: int = RANDOM_STATE,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Conservative direct-classification configuration; no residual target."""
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required; install requirements-xgb.txt")
    parameters: dict[str, Any] = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=420,
        max_depth=2,
        learning_rate=0.02,
        min_child_weight=18,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.50,
        reg_lambda=16.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    parameters.update(overrides or {})
    return XGBClassifier(**parameters)


def _positive_class_probability(model: Any, features: np.ndarray) -> float:
    probabilities = _positive_probabilities(model, features)
    calibrated = apply_probability_calibration(probabilities, getattr(model, "bbb_calibration_", None))
    return _clip(float(calibrated[0]))


def _direct_prediction_payload(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Create the direct XGBoost result and decode its 48D physical forecast."""
    physics = _physics_vector(physics_48d)
    noise_calibration=getattr(xgboost_model, "bbb_physics_noise_calibration_", None)
    features = build_56d_feature_matrix(core_pb, original_7d, physics, noise_calibration=noise_calibration)
    raw_pb = _positive_class_probability(xgboost_model, features)
    round_index = float(np.asarray(original_7d, dtype=np.float32).reshape(-1)[1])
    noise_score = float(features[0, -1])
    lo, hi = dynamic_probability_bounds(round_index, noise_score)
    final_pb = _clip(raw_pb, lo, hi)
    p_tie = _clip(float(physics[_PHYSICS_INDEX["winner_p_t"]]))
    p_player = 1.0 - final_pb
    ev_banker = final_pb * 0.95 - p_player
    ev_player = p_player - final_pb
    ev_thresholds = dict(DEFAULT_MIN_EV)
    ev_thresholds.update(getattr(xgboost_model, "bbb_ev_thresholds_", {}) or {})
    policy_config = dict(getattr(xgboost_model, "bbb_decision_policy_", {}) or {})
    policy_enabled = bool(policy_config.get("enabled", False))
    min_ev, activation_ev, soft_band = decision_policy_value(
        round_index, noise_score, ev_thresholds, enabled=policy_enabled, policy_config=policy_config
    )
    confidence_band, strong_margin = confidence_band_arrays(np.asarray([round_index]), np.asarray([noise_score]), policy_config if policy_enabled else None)
    probability_distance=abs(final_pb-.5)
    direction = "B" if probability_distance >= confidence_band[0] and ev_banker > activation_ev and ev_banker > ev_player else "P" if probability_distance >= confidence_band[0] and ev_player > activation_ev and ev_player > ev_banker else "Skip"
    final_direction = {"B": "莊 B", "P": "閒 P", "Skip": "觀望 Skip"}[direction]
    entry_tier = "skip" if direction == "Skip" else "strong" if probability_distance >= confidence_band[0] + strong_margin[0] else "weak"
    confidence = soft_confidence(
        ev_banker if direction == "B" else ev_player,
        min_ev,
        activation_ev,
        soft_band,
        min_confidence=float(policy_config.get("min_confidence", MIN_SOFT_CONFIDENCE)),
    ) if direction != "Skip" else 0.0
    return {
        "core_p_b": float(core_pb),
        "raw_p_b": raw_pb,
        "final_p_b": final_pb,
        "p_tie": p_tie,
        "p_player": p_player,
        "ev_banker": ev_banker,
        "ev_player": ev_player,
        "min_ev": min_ev,
        "activation_ev": activation_ev,
        "soft_band": soft_band,
        "confidence_band": float(confidence_band[0]),
        "effective_confidence_band": float(confidence_band[0]),
        "volume_guard_active": False,
        "entry_tier": entry_tier,
        "stake_multiplier": 0.0 if entry_tier == "skip" else 1.0 if entry_tier == "strong" else 0.5,
        "decision_policy_enabled": policy_enabled,
        "direction": direction,
        "final_direction": final_direction,
        "confidence": confidence,
        "probability_bounds": {"min": lo, "max": hi},
        "shoe_progress_weight": float(features[0, 1]),
        "physics_noise_score": noise_score,
        "features": features,
        "physics_forecast": unpack_physics_forecast(physics),
        "physics_integrity": physics_integrity_report(physics),
    }


def predict_final_probability(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Return final P(B) and unpacked next-hand physical estimates.

    The former ``Core P(B) + Delta`` operation does not exist in this path.
    ``[0.40, 0.60]`` is the direct-probability equivalent of the previous
    neutral-centred +/-0.10 safety envelope.  ``final_p_b`` is obtained from
    ``xgboost_model.predict_proba(features)[:, 1]`` (the Banker class).
    """
    return _direct_prediction_payload(
        core_pb,
        original_7d,
        physics_48d,
        xgboost_model=xgboost_model,
        probability_bounds=probability_bounds,
    )


def predict_final_result(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Compatibility name for the complete direct probability result payload."""
    return predict_final_probability(
        core_pb,
        original_7d,
        physics_48d,
        xgboost_model=xgboost_model,
        probability_bounds=probability_bounds,
    )


def _actual_b(record: Mapping[str, Any]) -> int:
    if record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    outcome = str(record.get("actual_outcome") or record.get("actual") or "").upper()
    if outcome == "B":
        return 1
    if outcome == "P":
        return 0
    raise ValueError("only directional B/P rows are valid training targets")


def _history(record: Mapping[str, Any]) -> str | Sequence[str]:
    return record.get("history") or record.get("history_fingerprint") or ""


def _core_pb(record: Mapping[str, Any]) -> float:
    value = record.get("core_p_b", record.get("core_pb"))
    if value is None:
        raise ValueError("missing core_p_b")
    return _clip(float(value))


def _history_tokens(history: str | Sequence[str]) -> list[str]:
    values = history.upper() if isinstance(history, str) else history
    return [str(value).strip().upper() for value in values if str(value).strip().upper() in {"B", "P", "T"}]


def _transition_tokens(history: Sequence[str]) -> list[str]:
    directional = [value for value in history if value in {"B", "P"}]
    return ["S" if directional[index] == directional[index - 1] else "X" for index in range(1, len(directional))]


def _current_stage(history: Sequence[str]) -> int:
    directional = [value for value in history if value in {"B", "P"}]
    if not directional:
        return 0
    side, count = directional[-1], 1
    for value in reversed(directional[:-1]):
        if value != side:
            break
        count += 1
    return count


def _current_depth(history: Sequence[str]) -> int:
    transitions = _transition_tokens(history)
    if not transitions:
        return 0
    token, count = transitions[-1], 1
    for value in reversed(transitions[:-1]):
        if value != token:
            break
        count += 1
    return count


def _sx_markov_p_same(history: Sequence[str]) -> float:
    transitions = _transition_tokens(history)
    if not transitions:
        return 0.5
    current = transitions[-1]
    start = max(0, len(transitions) - 1 - 24)
    same = switched = 0
    for index in range(start, len(transitions) - 1):
        if transitions[index] != current:
            continue
        if transitions[index + 1] == "S":
            same += 1
        else:
            switched += 1
    return _clip((same + 1.0) / (same + switched + 2.0))


def _rebuild_original_7d(record: Mapping[str, Any], core_pb: float) -> np.ndarray:
    history = _history_tokens(_history(record))
    total_hands = _clip(float(record.get("estimated_total_hands", 60) or 60), 40.0, 90.0)
    round_index = float(max(1, min(70, len(history) + 1)))
    stage = float(record["stage"]) if record.get("stage") is not None else float(_current_stage(history))
    depth = float(record["depth"]) if record.get("depth") is not None else float(_current_depth(history))
    return np.asarray([
        core_pb,
        round_index,
        total_hands,
        _clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands)),
        _sx_markov_p_same(history),
        stage,
        depth,
    ], dtype=np.float32)


def _original_7d(record: Mapping[str, Any], core_pb: float) -> np.ndarray:
    if all(record.get(name) is not None for name in ORIGINAL_7D_FEATURE_NAMES):
        return np.asarray([float(record[name]) for name in ORIGINAL_7D_FEATURE_NAMES], dtype=np.float32)
    return _rebuild_original_7d(record, core_pb)


def _physics_48d(record: Mapping[str, Any], extractor: PhysicsFeatureExtractor | None) -> np.ndarray:
    supplied = record.get("physics_48d")
    if supplied is not None:
        return np.asarray(supplied, dtype=np.float32).reshape(-1)
    if extractor is None:
        raise ValueError("physics_48d is missing and no physics extractor was supplied")
    return extractor.predict_features(_history(record))


def _snapshot_56d(record: Mapping[str, Any]) -> np.ndarray | None:
    """Accept new 57D snapshots and migrate legacy 56D snapshots in-place."""
    snapshot = record.get("features_57d", record.get("feature_snapshot_57d"))
    if snapshot is not None:
        vector = np.asarray(snapshot, dtype=np.float32).reshape(-1)
        if vector.size != FEATURE_DIM or not np.all(np.isfinite(vector)):
            raise ValueError("features_57d must contain exactly 57 finite values")
        supplied_physics = record.get("physics_48d")
        if supplied_physics is not None:
            vector[-1] = physics_noise_score(supplied_physics, float(vector[2]))
        return vector.reshape(1, FEATURE_DIM)

    legacy = record.get("features_56d", record.get("feature_snapshot_56d"))
    if legacy is None:
        return None
    vector = np.asarray(legacy, dtype=np.float32).reshape(-1)
    if vector.size != 56 or not np.all(np.isfinite(vector)):
        raise ValueError("legacy features_56d must contain exactly 56 finite values")
    physics = vector[8:]
    migrated = np.hstack((
        vector[0],
        (float(vector[2]) / 70.0) ** 3,
        vector[2:8],
        physics,
        physics_noise_score(physics, float(vector[2])),
    )).astype(np.float32, copy=False)
    return migrated.reshape(1, FEATURE_DIM)


def make_training_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    physics_extractor: PhysicsFeatureExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    """Build direct-label training data and retain the matching chronological rows.

    New browser records provide the exact 57D vector that was used at
    prediction-time.  Older records remain supported by rebuilding the feature
    blocks only when no snapshot is available.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    used_records: list[Mapping[str, Any]] = []
    for record in records:
        try:
            x = _snapshot_56d(record)
            if x is None:
                pb = _core_pb(record)
                x = build_56d_feature_matrix(
                    pb,
                    _original_7d(record, pb),
                    _physics_48d(record, physics_extractor),
                )
            y = _actual_b(record)
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(x[0])
        labels.append(y)
        used_records.append(record)
    if not rows:
        raise ValueError("no valid B/P training rows")
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int8),
        used_records,
    )


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    physics_extractor: PhysicsFeatureExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological 57D inputs with absolute binary labels (B=1/P=0)."""
    x, y, _ = make_training_dataset(records, physics_extractor=physics_extractor)
    return x, y


def chronological_validation_mask(row_count: int, *, fraction: float = 0.20) -> np.ndarray:
    """Strict walk-forward split: earlier rows train, final rows validate."""
    if row_count < 2:
        raise ValueError("at least two chronological rows are required")
    fraction = min(0.50, max(0.05, float(fraction)))
    cut = max(1, min(row_count - 1, int(math.floor(row_count * (1.0 - fraction)))))
    mask = np.zeros(row_count, dtype=bool)
    mask[cut:] = True
    return mask


def shoe_level_validation_mask(
    records: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.20,
) -> np.ndarray:
    """Hold out complete latest shoes, preserving the supplied chronological order."""
    if len(records) < 2:
        raise ValueError("at least two chronological rows are required")
    fraction = min(0.50, max(0.05, float(fraction)))
    shoe_keys: list[str] = []
    ordered_shoes: list[str] = []
    seen: set[str] = set()
    for record in records:
        value = record.get("shoe_id")
        shoe_id = str(value).strip() if value is not None else ""
        if not shoe_id:
            raise ValueError("strict shoe-level validation requires a non-empty shoe_id")
        shoe_keys.append(shoe_id)
        if shoe_id not in seen:
            ordered_shoes.append(shoe_id)
            seen.add(shoe_id)
    if len(ordered_shoes) < 2:
        raise ValueError("strict shoe-level validation requires at least two shoes")
    validation_shoe_count = max(
        1,
        min(len(ordered_shoes) - 1, int(math.ceil(len(ordered_shoes) * fraction))),
    )
    validation_shoes = set(ordered_shoes[-validation_shoe_count:])
    return np.asarray([shoe_id in validation_shoes for shoe_id in shoe_keys], dtype=bool)


def validate_chronological_shoes(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Reject interleaved shoes or backwards rounds before any time split."""
    closed: set[str] = set()
    previous_shoe = ""
    last_round: dict[str, float] = {}
    shoes: set[str] = set()
    for index, record in enumerate(records):
        shoe_id = str(record.get("shoe_id") or "").strip()
        if not shoe_id:
            raise ValueError("chronological validation requires a non-empty shoe_id")
        if shoe_id != previous_shoe:
            if shoe_id in closed:
                raise ValueError(f"shoe_id {shoe_id!r} reappears after another shoe at row {index}")
            if previous_shoe:
                closed.add(previous_shoe)
            previous_shoe = shoe_id
        shoes.add(shoe_id)

        raw_round = record.get("round_index")
        if raw_round is None:
            snapshot = record.get("features_57d")
            if isinstance(snapshot, Sequence) and not isinstance(snapshot, (str, bytes)):
                flattened = np.asarray(snapshot, dtype=np.float64).reshape(-1)
                raw_round = flattened[2] if flattened.size == FEATURE_DIM else None
        if raw_round is None:
            continue
        round_index = float(raw_round)
        if not math.isfinite(round_index):
            raise ValueError(f"round_index must be finite at row {index}")
        if shoe_id in last_round and round_index < last_round[shoe_id]:
            raise ValueError(f"round_index moves backwards inside shoe_id {shoe_id!r} at row {index}")
        last_round[shoe_id] = round_index
    return {"rows": len(records), "shoes": len(shoes)}


def three_way_shoe_masks(
    records: Sequence[Mapping[str, Any]],
    *,
    calibration_fraction: float = 0.15,
    holdout_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological whole-shoe train/calibration/strict-holdout split."""
    validate_chronological_shoes(records)
    shoe_keys = [str(record.get("shoe_id") or "").strip() for record in records]
    if any(not key for key in shoe_keys):
        raise ValueError("three-way shoe split requires a non-empty shoe_id")
    ordered_shoes = list(dict.fromkeys(shoe_keys))
    if len(ordered_shoes) < 3:
        raise ValueError("three-way shoe split requires at least three shoes")
    calibration_fraction = min(0.40, max(0.05, float(calibration_fraction)))
    holdout_fraction = min(0.40, max(0.05, float(holdout_fraction)))
    holdout_count = max(1, int(math.ceil(len(ordered_shoes) * holdout_fraction)))
    calibration_count = max(1, int(math.ceil(len(ordered_shoes) * calibration_fraction)))
    calibration_count = min(calibration_count, len(ordered_shoes) - holdout_count - 1)
    holdout_shoes = set(ordered_shoes[-holdout_count:])
    calibration_shoes = set(ordered_shoes[-(holdout_count + calibration_count):-holdout_count])
    holdout = np.asarray([key in holdout_shoes for key in shoe_keys], dtype=bool)
    calibration = np.asarray([key in calibration_shoes for key in shoe_keys], dtype=bool)
    train = ~(calibration | holdout)
    return train, calibration, holdout


def nested_tuning_masks(
    records: Sequence[Mapping[str, Any]],
    training: np.ndarray,
    *,
    fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the latest whole shoes inside training for hyperparameter selection."""
    keys = np.asarray([str(record.get("shoe_id") or "").strip() for record in records])
    ordered = list(dict.fromkeys(keys[np.asarray(training, dtype=bool)].tolist()))
    if len(ordered) < 2:
        raise ValueError("nested tuning split requires at least two training shoes")
    count = max(1, min(len(ordered) - 1, int(math.ceil(len(ordered) * min(0.30, max(0.05, float(fraction)))))))
    tuning_shoes = set(ordered[-count:])
    tuning = np.asarray(training, dtype=bool) & np.asarray([key in tuning_shoes for key in keys], dtype=bool)
    fit = np.asarray(training, dtype=bool) & ~tuning
    return fit, tuning


def balanced_sample_weights(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Class-balanced weights focused on 50-70, low-noise and likely-actionable rows."""
    labels=np.asarray(y,dtype=np.int8).reshape(-1);features=np.asarray(x,dtype=np.float64)
    rounds=features[:,2];weights=np.ones(len(labels),dtype=np.float64)
    for label in (0,1):
        mask=labels==label
        if np.any(mask): weights[mask]*=len(labels)/(2.0*float(np.sum(mask)))
    # Keep every stage represented without letting the small late slice dominate.
    stage_factor=np.ones(len(labels),dtype=np.float64)
    stage_factor[(rounds>40)&(rounds<=50)]=1.05
    stage_factor[(rounds>50)&(rounds<=70)]=1.20
    stage_factor[rounds>70]=1.05
    uncertainty=np.clip(features[:,-1],0.0,1.0)
    low_noise_factor=1.0+0.15*(1.0-uncertainty)
    core_probability=np.clip(features[:,0],0.0,1.0)
    core_edge=np.maximum(core_probability*.95-(1.0-core_probability),(1.0-core_probability)-core_probability)
    action_proxy=np.clip((core_edge-_stage_min_ev(rounds,DEFAULT_MIN_EV)+.01)/.02,0.0,1.0)
    action_factor=1.0+0.10*action_proxy
    weights*=stage_factor*low_noise_factor*action_factor
    weights=np.clip(weights,0.35,2.75)
    return (weights/np.mean(weights)).astype(np.float32)


def recalibrate_noise_feature(
    x: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> np.ndarray:
    """Recompute only feature 56 from each stored 48D block; all other 57D values stay fixed."""
    out=np.asarray(x,dtype=np.float32).copy()
    if not calibration: return out
    for index,record in enumerate(records):
        physics=record.get("physics_48d")
        if isinstance(physics,Sequence) and not isinstance(physics,(str,bytes)):
            values=np.asarray(physics,dtype=np.float64).reshape(-1)
            if values.size==PHYSICS_DIM and np.all(np.isfinite(values)):
                out[index,-1]=np.float32(physics_noise_score(values,float(out[index,2]),calibration))
                continue
        out[index,-1]=np.float32(apply_uncertainty_calibration([float(out[index,-1])],calibration)[0])
    return out


def legacy_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Recreate the pre-upgrade integrity-only noise feature for release comparison."""
    legacy=np.asarray(x,dtype=np.float32).copy(); physics=legacy[:,8:56]
    legacy[:,-1]=np.abs(np.sum(physics[:,0:3],axis=1)-1.0)+np.abs(np.sum(physics[:,39:43],axis=1)-1.0)
    return legacy


def _positive_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", (0, 1)))
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise RuntimeError("classifier has no Banker class")
    return np.clip(probabilities[:, int(positive[0])], 1e-7, 1.0 - 1e-7)


def fit_probability_calibration(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    shoe_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose identity/Platt/isotonic on a chronological calibration tail."""
    labels = np.asarray(y, dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return {"method": "identity", "slope": 1.0, "intercept": 0.0}
    probability = _positive_probabilities(model, x)

    def fit_platt(p: np.ndarray, target: np.ndarray, weight: np.ndarray | None) -> LogisticRegression:
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        fitted = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_STATE)
        fitted.fit(logits, target, sample_weight=weight)
        return fitted

    if shoe_ids is not None and len(shoe_ids)==len(labels):
        shoes=np.asarray([str(value) for value in shoe_ids]); ordered=list(dict.fromkeys(shoes.tolist()))
        split_shoes=max(1,min(len(ordered)-1,int(math.floor(len(ordered)*.70))))
        fit_shoes=set(ordered[:split_shoes]); probe_fit=np.asarray([shoe in fit_shoes for shoe in shoes],dtype=bool)
    else:
        split=max(1,min(len(labels)-1,int(math.floor(len(labels)*.70)))); probe_fit=np.arange(len(labels))<split
    probe_test=~probe_fit
    can_compare = len(labels) >= 500 and len(np.unique(labels[probe_fit])) == 2 and len(np.unique(labels[probe_test])) == 2
    method = "platt"
    selection: dict[str, Any] = {}
    if can_compare:
        fit_weight = sample_weight[probe_fit] if sample_weight is not None else None
        platt_probe = fit_platt(probability[probe_fit], labels[probe_fit], fit_weight)
        probe_logits = np.log(probability[probe_test] / (1.0 - probability[probe_test])).reshape(-1, 1)
        platt_probability = platt_probe.predict_proba(probe_logits)[:, 1]
        selection = {
            "identity_brier": _brier(probability[probe_test], labels[probe_test]),
            "platt_brier": _brier(platt_probability, labels[probe_test]),
        }
        isotonic_allowed = len(labels) >= ISOTONIC_MIN_ROWS and len(np.unique(probability[probe_fit])) >= 32
        if isotonic_allowed:
            isotonic_probe = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic_probe.fit(probability[probe_fit], labels[probe_fit], sample_weight=fit_weight)
            isotonic_probability = isotonic_probe.predict(probability[probe_test])
            selection["isotonic_brier"] = _brier(isotonic_probability, labels[probe_test])
        calibrated_scores = [
            (name.removesuffix("_brier"), score)
            for name, score in selection.items()
            if name != "identity_brier" and name.endswith("_brier")
        ]
        best_calibrated = min(calibrated_scores, key=lambda item: item[1])
        method = best_calibrated[0] if selection["identity_brier"] - best_calibrated[1] >= CALIBRATION_MIN_BRIER_IMPROVEMENT else "identity"
        selection.update({
            "selected_method": method,
            "minimum_brier_improvement": CALIBRATION_MIN_BRIER_IMPROVEMENT,
            "isotonic_allowed": isotonic_allowed,
        })
    if method == "identity":
        return {"method": "identity", "slope": 1.0, "intercept": 0.0, "selection": selection}
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(probability, labels, sample_weight=sample_weight)
        return {"method": method, "x_thresholds": calibrator.X_thresholds_.tolist(), "y_thresholds": calibrator.y_thresholds_.tolist(), "selection": selection}
    calibrator = fit_platt(probability, labels, sample_weight)
    return {"method": method, "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]), "selection": selection}


def apply_probability_calibration(probability: np.ndarray, calibration: Mapping[str, Any] | None) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if not calibration:
        return values
    if calibration.get("method") == "isotonic":
        x_thresholds=np.asarray(calibration.get("x_thresholds") or [],dtype=np.float64)
        y_thresholds=np.asarray(calibration.get("y_thresholds") or [],dtype=np.float64)
        return np.clip(np.interp(values,x_thresholds,y_thresholds),1e-7,1.0-1e-7) if len(x_thresholds)>1 else values
    if calibration.get("method") != "platt": return values
    logits = np.log(values / (1.0 - values))
    adjusted = float(calibration.get("slope", 1.0)) * logits + float(calibration.get("intercept", 0.0))
    return 1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0)))


def feature_importance_report(
    model: Any,
    x: np.ndarray | None = None,
    *,
    top_n: int = 15,
) -> dict[str, Any]:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    ranked: list[tuple[str, float]] = []
    for key, value in gain.items():
        index = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else FEATURE_NAMES.index(key)
        if 0 <= index < FEATURE_DIM:
            ranked.append((FEATURE_NAMES[index], float(value)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in ranked) or 1.0
    report: dict[str, Any] = {
        "top_gain": [{"feature": name, "share": value / total} for name, value in ranked[:top_n]],
        "unused_feature_count": FEATURE_DIM - len(ranked),
    }
    if x is not None and DMatrix is not None and len(x):
        sample = np.asarray(x[: min(2048, len(x))], dtype=np.float32)
        contributions = np.asarray(booster.predict(DMatrix(sample), pred_contribs=True), dtype=np.float64)
        mean_absolute = np.mean(np.abs(contributions[:, :FEATURE_DIM]), axis=0)
        shap_total = float(np.sum(mean_absolute)) or 1.0
        order = np.argsort(mean_absolute)[::-1][:top_n]
        report["top_mean_abs_shap"] = [
            {"feature": FEATURE_NAMES[int(index)], "share": float(mean_absolute[index] / shap_total)}
            for index in order
        ]
    return report


def _stage_min_ev(rounds: np.ndarray, thresholds: Mapping[str, float]) -> np.ndarray:
    values = np.full(len(rounds), float(thresholds["middle"]), dtype=np.float64)
    values[rounds <= 40] = float(thresholds["early"])
    values[rounds > 50] = float(thresholds["late"])
    return values


def decision_policy_arrays(
    rounds: np.ndarray,
    noise_scores: np.ndarray | None,
    thresholds: Mapping[str, float],
    *,
    enabled: bool,
    policy_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the selected browser-deployable EV/noise policy without changing 57D features."""
    rounds = np.asarray(rounds, dtype=np.float64)
    base = _stage_min_ev(rounds, thresholds)
    if not enabled:
        return base, base.copy(), np.zeros(len(rounds), dtype=np.float64)
    config = dict(policy_config or {})
    noise_threshold = _clip(float(config.get("noise_threshold", PHYSICS_NOISE_LOW_THRESHOLD)))
    max_noise_penalty = max(0.0, float(config.get("max_noise_ev_penalty", MAX_NOISE_EV_PENALTY)))
    middle_relief = max(0.0, float(config.get("middle_relief", MIDDLE_EV_RELIEF)))
    late_relief = max(0.0, float(config.get("late_relief", LATE_EV_RELIEF)))
    middle_soft_band = max(0.0, float(config.get("middle_soft_band", MIDDLE_SOFT_BAND)))
    late_soft_band = max(0.0, float(config.get("late_soft_band", LATE_SOFT_BAND)))
    noise = np.full(len(rounds), 0.5) if noise_scores is None else np.clip(np.asarray(noise_scores, dtype=np.float64), 0.0, 1.0)
    if len(noise) != len(rounds):
        raise ValueError("noise_scores and rounds must be aligned")
    low_noise = noise <= noise_threshold
    middle = (rounds > 40) & (rounds <= 50); late = rounds > 50
    relief = np.zeros(len(rounds), dtype=np.float64)
    relief[middle & low_noise] = middle_relief; relief[late & low_noise] = late_relief
    penalty = np.minimum(max_noise_penalty, np.maximum(0.0, noise - noise_threshold) * (max_noise_penalty / max(1e-9, 1.0 - noise_threshold)))
    min_ev = np.maximum(0.0, base - relief + penalty)
    soft_band = np.zeros(len(rounds), dtype=np.float64)
    soft_band[middle & low_noise] = middle_soft_band; soft_band[late & low_noise] = late_soft_band
    return min_ev, np.maximum(0.0, min_ev - soft_band), soft_band


def decision_policy_value(
    round_index: float,
    noise_score: float,
    thresholds: Mapping[str, float],
    *,
    enabled: bool,
    policy_config: Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    values = decision_policy_arrays(
        np.asarray([round_index]), np.asarray([noise_score]), thresholds, enabled=enabled, policy_config=policy_config
    )
    return tuple(float(item[0]) for item in values)


def soft_confidence(edge: float, min_ev: float, activation_ev: float, soft_band: float, *, min_confidence: float = MIN_SOFT_CONFIDENCE) -> float:
    if edge <= activation_ev:
        return 0.0
    premium = max(0.0, edge - min_ev)
    if soft_band <= 0.0:
        return premium
    return max(max(0.0, min_confidence), premium, 0.5 * min(soft_band, edge - activation_ev))


def confidence_band_arrays(
    rounds: np.ndarray,
    noise_scores: np.ndarray | None,
    policy_config: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-from-0.50 gate; returns (band, strong_margin) without altering 57D features."""
    rounds = np.asarray(rounds, dtype=np.float64)
    config = dict(policy_config or {}); band_config = dict(config.get("confidence_band") or {})
    if not band_config or not bool(config.get("enabled", False)):
        return np.zeros(len(rounds), dtype=np.float64), np.zeros(len(rounds), dtype=np.float64)
    noise = np.full(len(rounds), 0.5) if noise_scores is None else np.clip(np.asarray(noise_scores, dtype=np.float64), 0.0, 1.0)
    stage = np.where(rounds <= 40, "early", np.where(rounds <= 50, "middle", "late"))
    base = np.asarray([float(band_config.get(str(name), 0.0)) for name in stage], dtype=np.float64)
    strong_config = dict(band_config.get("strong_margin") or {})
    strong = np.asarray([float(strong_config.get(str(name), 0.0)) for name in stage], dtype=np.float64)
    ref = _clip(float(band_config.get("noise_reference", 0.50)))
    gain = max(0.0, float(band_config.get("noise_gain", 0.0)))
    clean_late = (rounds > 50) & (noise <= PHYSICS_NOISE_LOW_THRESHOLD)
    band = base + gain * np.maximum(0.0, noise - ref)
    band[clean_late] -= max(0.0, float(band_config.get("clean_late_relief", 0.0)))
    minimum = max(0.0, float(band_config.get("minimum", 0.0)))
    maximum = max(minimum, float(band_config.get("maximum", 0.50)))
    return np.clip(band, minimum, maximum), np.maximum(0.0, strong)


def _stage_value(round_index: float, mapping: Mapping[str, Any], fallback: float) -> float:
    key = "early" if round_index <= 40 else "middle" if round_index <= 50 else "late"
    return float(mapping.get(key, fallback))


def decision_returns(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
    rounds: np.ndarray,
    thresholds: Mapping[str, float],
    noise_scores: np.ndarray | None = None,
    *,
    policy_enabled: bool = False,
    policy_config: Mapping[str, Any] | None = None,
    shoe_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probability_b = np.asarray(probability_b, dtype=np.float64)
    actual_b = np.asarray(actual_b, dtype=np.int8)
    rounds = np.asarray(rounds, dtype=np.float64)
    probability_p = 1.0 - probability_b
    ev_banker = probability_b * 0.95 - probability_p
    ev_player = probability_p - probability_b
    _, activation_ev, _ = decision_policy_arrays(
        rounds, noise_scores, thresholds, enabled=policy_enabled, policy_config=policy_config
    )
    band, _ = confidence_band_arrays(rounds, noise_scores, policy_config if policy_enabled else None)
    distance = np.abs(probability_b - 0.5)
    banker = (ev_banker > activation_ev) & (ev_banker > ev_player) & (distance >= band)
    player = (ev_player > activation_ev) & (ev_player > ev_banker) & (distance >= band)
    config = dict(policy_config or {}); volume = dict(config.get("volume_guard") or {})
    if policy_enabled and bool(volume.get("enabled", False)) and shoe_ids is not None:
        shoes = [str(value) for value in shoe_ids]
        if len(shoes) != len(probability_b):
            raise ValueError("shoe_ids and probability_b must be aligned")
        window=max(1,int(volume.get("window",16))); minimum=max(0,int(volume.get("min_history",8)))
        floor=_clip(float(volume.get("expected_correct_floor",.95)))
        band_relief=max(0.0,float(volume.get("band_relief",0.0))); ev_relief=max(0.0,float(volume.get("ev_relief",0.0)))
        rates=dict(volume.get("target_action_rate") or {}); band_floor=max(0.0,float((config.get("confidence_band") or {}).get("minimum",0.0)))
        state: dict[str,list[tuple[float,float,float]]] = {}
        for index, shoe in enumerate(shoes):
            prior=state.setdefault(shoe,[])
            if len(prior)>=minimum:
                actions=sum(item[0] for item in prior)/len(prior)
                expected=sum(item[1] for item in prior); baseline=sum(item[2] for item in prior)
                target=_stage_value(rounds[index],rates,0.0)
                if actions < target or (baseline>0.0 and expected+1e-12 < floor*baseline):
                    relaxed_band=max(band_floor,band[index]-band_relief); relaxed_ev=max(0.0,activation_ev[index]-ev_relief)
                    banker[index]=(ev_banker[index]>relaxed_ev and ev_banker[index]>ev_player[index] and distance[index]>=relaxed_band)
                    player[index]=(ev_player[index]>relaxed_ev and ev_player[index]>ev_banker[index] and distance[index]>=relaxed_band)
            base_action=(ev_banker[index]>activation_ev[index] and ev_banker[index]>ev_player[index]) or (ev_player[index]>activation_ev[index] and ev_player[index]>ev_banker[index])
            action=float(banker[index] or player[index]); expected=max(probability_b[index],probability_p[index]) if action else 0.0
            baseline=max(probability_b[index],probability_p[index]) if base_action else 0.0
            prior.append((action,expected,baseline))
            if len(prior)>window: del prior[0]
    wagered = banker | player
    realised = np.zeros(len(probability_b), dtype=np.float64)
    realised[banker] = np.where(actual_b[banker] == 1, 0.95, -1.0)
    realised[player] = np.where(actual_b[player] == 0, 1.0, -1.0)
    return realised, wagered


def decision_metrics(realised: np.ndarray, wagered: np.ndarray) -> dict[str, float]:
    wagers = int(np.sum(wagered)); rows = len(wagered)
    return {
        "rows": float(rows),
        "wagers": float(wagers),
        "action_rate": float(wagers / rows) if rows else 0.0,
        "skip_rate": float(1.0 - wagers / rows) if rows else 0.0,
        "hit_rate": float(np.mean(realised[wagered] > 0.0)) if wagers else 0.0,
        "absolute_correct_bets": float(np.sum(realised[wagered] > 0.0)) if wagers else 0.0,
        "realized_ev_per_row": float(np.mean(realised)) if rows else 0.0,
        "realized_ev_per_bet": float(np.sum(realised) / wagers) if wagers else 0.0,
    }


def optimize_ev_thresholds(
    probability_b: np.ndarray,
    actual_b: np.ndarray,
    rounds: np.ndarray,
    noise_scores: np.ndarray | None = None,
    *,
    policy_enabled: bool = False,
    policy_config: Mapping[str, Any] | None = None,
    shoe_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Tune the existing three numbers under accuracy/EV and +5% Skip guardrails."""
    thresholds = dict(DEFAULT_MIN_EV); noise=np.full(len(rounds),.5) if noise_scores is None else np.asarray(noise_scores,dtype=np.float64)
    shoes = None if shoe_ids is None else np.asarray([str(value) for value in shoe_ids])
    if shoes is not None and len(shoes) != len(rounds):
        raise ValueError("shoe_ids and rounds must be aligned")
    grids = {
        "early": np.arange(0.010, 0.0451, 0.0025),
        "middle": np.arange(0.005, 0.0301, 0.0025),
        "late": np.arange(0.000, 0.0251, 0.00125),
    }
    masks = {
        "early": np.asarray(rounds) <= 40,
        "middle": (np.asarray(rounds) > 40) & (np.asarray(rounds) <= 50),
        "late": np.asarray(rounds) > 50,
    }
    stages: dict[str, Any] = {}
    for stage, mask in masks.items():
        count = int(np.sum(mask))
        if not count:
            stages[stage] = {"rows": 0, "min_ev": thresholds[stage]}
            continue
        baseline_realised,baseline_wagered=decision_returns(probability_b[mask],actual_b[mask],np.asarray(rounds)[mask],DEFAULT_MIN_EV,noise[mask],policy_enabled=policy_enabled,policy_config=policy_config,shoe_ids=None if shoes is None else shoes[mask])
        baseline=decision_metrics(baseline_realised,baseline_wagered)
        minimum_wagers=max(1,int(math.ceil(max(0.0,baseline["action_rate"]-MAX_SKIP_RATE_INCREASE)*count)))
        best: tuple[float, float, float, float] | None = None
        candidates = sorted(set(float(value) for value in grids[stage]) | {float(DEFAULT_MIN_EV[stage])})
        for candidate in candidates:
            trial = dict(thresholds)
            trial[stage] = candidate
            realised, wagered = decision_returns(probability_b[mask], actual_b[mask], np.asarray(rounds)[mask], trial, noise[mask], policy_enabled=policy_enabled, policy_config=policy_config, shoe_ids=None if shoes is None else shoes[mask])
            wagers = int(np.sum(wagered))
            if wagers < minimum_wagers:
                continue
            metrics=decision_metrics(realised,wagered); skip_delta=metrics["skip_rate"]-baseline["skip_rate"]
            if skip_delta > MAX_SKIP_RATE_INCREASE + 1e-12: continue
            if metrics["hit_rate"] + 1e-12 < baseline["hit_rate"] or metrics["realized_ev_per_bet"] + 1e-12 < baseline["realized_ev_per_bet"] or metrics["absolute_correct_bets"] + 1e-12 < .95*baseline["absolute_correct_bets"]: continue
            score=metrics["realized_ev_per_bet"]+.25*metrics["realized_ev_per_row"]+.10*metrics["hit_rate"]
            result=(score,metrics["realized_ev_per_bet"],metrics["realized_ev_per_row"],-candidate)
            if best is None or result > best:
                best = result
        if best is not None:
            thresholds[stage] = float(-best[3])
        realised, wagered = decision_returns(probability_b[mask], actual_b[mask], np.asarray(rounds)[mask], thresholds, noise[mask], policy_enabled=policy_enabled, policy_config=policy_config, shoe_ids=None if shoes is None else shoes[mask])
        tuned=decision_metrics(realised,wagered)
        stages[stage] = {
            "rows": count,
            "wagers": int(tuned["wagers"]),
            "action_rate": tuned["action_rate"],
            "skip_rate": tuned["skip_rate"],
            "baseline_skip_rate": baseline["skip_rate"],
            "baseline_hit_rate": baseline["hit_rate"],
            "baseline_realized_ev_per_bet": baseline["realized_ev_per_bet"],
            "skip_rate_delta": tuned["skip_rate"]-baseline["skip_rate"],
            "hit_rate": tuned["hit_rate"],
            "realized_ev_per_bet": tuned["realized_ev_per_bet"],
            "absolute_correct_bets": tuned["absolute_correct_bets"], "baseline_absolute_correct_bets": baseline["absolute_correct_bets"],
            "quality_constraint_passed": bool(tuned["hit_rate"]+1e-12>=baseline["hit_rate"] and tuned["realized_ev_per_bet"]+1e-12>=baseline["realized_ev_per_bet"] and tuned["absolute_correct_bets"]+1e-12>=.95*baseline["absolute_correct_bets"]),
            "min_ev": thresholds[stage],
        }
    baseline_realised,baseline_wagered=decision_returns(probability_b,actual_b,rounds,DEFAULT_MIN_EV,noise,policy_enabled=policy_enabled,policy_config=policy_config,shoe_ids=shoes)
    tuned_realised,tuned_wagered=decision_returns(probability_b,actual_b,rounds,thresholds,noise,policy_enabled=policy_enabled,policy_config=policy_config,shoe_ids=shoes)
    baseline=decision_metrics(baseline_realised,baseline_wagered); tuned=decision_metrics(tuned_realised,tuned_wagered)
    skip_delta=tuned["skip_rate"]-baseline["skip_rate"]
    quality_passed=tuned["hit_rate"]+1e-12>=baseline["hit_rate"] and tuned["realized_ev_per_bet"]+1e-12>=baseline["realized_ev_per_bet"] and tuned["absolute_correct_bets"]+1e-12>=.95*baseline["absolute_correct_bets"]
    if skip_delta > MAX_SKIP_RATE_INCREASE + 1e-12 or not quality_passed:
        thresholds=dict(DEFAULT_MIN_EV); tuned=baseline; skip_delta=0.0
        quality_passed=True
    return {"thresholds":thresholds,"stages":stages,"baseline":baseline,"tuned":tuned,"skip_rate_delta":skip_delta,"decision_policy_enabled":bool(policy_enabled),
            "quality_constraint":{"requires_non_decreasing_hit_rate":True,"requires_non_decreasing_ev_per_bet":True,"requires_absolute_correct_bets_at_least_95pct":True,"passed":bool(quality_passed)},
            "skip_constraint":{"preferred_max_increase":PREFERRED_SKIP_RATE_INCREASE,"hard_max_increase":MAX_SKIP_RATE_INCREASE,"passed":skip_delta<=MAX_SKIP_RATE_INCREASE+1e-12}}


def optimize_smoothing_and_thresholds(
    calibrated_probability: np.ndarray,
    actual_b: np.ndarray,
    x: np.ndarray,
    shoe_ids: Sequence[str],
    *,
    profiles: Sequence[str] = tuple(EMA_PROFILES),
    decision_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select EMA + entry policy using chronological tuning shoes and hard quality gates."""
    features=np.asarray(x,dtype=np.float64);rounds=features[:,2];noise=features[:,-1]
    clipped=bounded_from_calibrated(calibrated_probability,features)
    candidates=["off"]+[str(value) for value in profiles if str(value)!="off"]
    candidates=list(dict.fromkeys(candidates))
    unknown=[name for name in candidates if name not in EMA_PROFILES]
    if unknown: raise ValueError(f"unknown EMA profiles: {unknown}")
    policies = dict(decision_policies or DECISION_POLICY_PROFILES)
    if "hard_ev" not in policies:
        raise ValueError("decision_policies must retain hard_ev as the baseline")
    reports: list[dict[str, Any]] = []; baseline: dict[str, float] | None = None
    best: tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for name in candidates:
        smoothing=dict(EMA_PROFILES[name])
        final=dynamic_ema_by_shoe(clipped,features,shoe_ids,smoothing)
        for policy_name, policy_config in policies.items():
            policy=dict(policy_config); policy_enabled=bool(policy.get("enabled",False))
            tuning = optimize_ev_thresholds(final,actual_b,rounds,noise,policy_enabled=policy_enabled,policy_config=policy,shoe_ids=shoe_ids)
            realised,wagered=decision_returns(final,actual_b,rounds,tuning["thresholds"],noise,policy_enabled=policy_enabled,policy_config=policy,shoe_ids=shoe_ids)
            metrics=decision_metrics(realised,wagered);brier=_brier(final,actual_b)
            if baseline is None: baseline={**metrics,"brier":brier}
            skip_delta=metrics["skip_rate"]-baseline["skip_rate"]
            ev_delta=metrics["realized_ev_per_bet"]-baseline["realized_ev_per_bet"]
            hit_delta=metrics["hit_rate"]-baseline["hit_rate"]
            absolute_delta=metrics["absolute_correct_bets"]-baseline["absolute_correct_bets"]
            brier_delta=brier-baseline["brier"]
            min_skip_increase=max(0.0,float(policy.get("min_skip_increase",0.0)))
            absolute_passed=metrics["absolute_correct_bets"]+1e-12>=.95*baseline["absolute_correct_bets"]
            eligible=(skip_delta<=MAX_SKIP_RATE_INCREASE+1e-12 and skip_delta+1e-12>=min_skip_increase and
                      ev_delta>=-1e-12 and hit_delta>=-1e-12 and absolute_passed and brier_delta<=MAX_SMOOTHING_BRIER_INCREASE)
            score=brier-.02*metrics["realized_ev_per_bet"]-.01*metrics["hit_rate"]-.05*metrics["realized_ev_per_row"]+.05*metrics["skip_rate"]
            report={"profile":name,"smoothing":smoothing,"decision_policy_profile":policy_name,"decision_policy":policy,"decision_policy_enabled":policy_enabled,"score":score,"brier":brier,**metrics,
                    "skip_rate_delta":skip_delta,"ev_per_bet_delta":ev_delta,"hit_rate_delta":hit_delta,"absolute_correct_bets_delta":absolute_delta,"absolute_correct_bets_constraint_passed":absolute_passed,"minimum_skip_increase":min_skip_increase,"brier_delta":brier_delta,
                    "guardrail_passed":eligible,"ev_thresholds":tuning["thresholds"]}
            reports.append(report)
            if eligible and (best is None or score<best[0]): best=(score,smoothing,policy,tuning)
    assert baseline is not None and best is not None
    return {"method":"dynamic_post_clip_ema","smoothing":best[1],"decision_policy":best[2],"ev_tuning":best[3],
            "baseline": baseline, "candidates": reports,
            "guardrail": {"max_skip_increase": MAX_SKIP_RATE_INCREASE,
                          "requires_non_decreasing_hit_rate": True,
                          "requires_non_decreasing_ev_per_bet": True,
                          "requires_absolute_correct_bets_at_least_95pct": True,
                          "max_brier_increase": MAX_SMOOTHING_BRIER_INCREASE}}


def shoe_bootstrap_ci(
    values: np.ndarray,
    shoe_ids: Sequence[str],
    *,
    samples: int = 1000,
    random_state: int = RANDOM_STATE,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    shoes = np.asarray([str(value) for value in shoe_ids])
    unique = np.unique(shoes)
    if len(values) != len(shoes) or not len(values):
        raise ValueError("bootstrap values and shoe_ids must be non-empty and aligned")
    if len(unique) < 2 or samples < 2:
        point = float(np.mean(values))
        return [point, point]
    rng = np.random.default_rng(random_state)
    estimates = np.empty(int(samples), dtype=np.float64)
    indices = {shoe: np.flatnonzero(shoes == shoe) for shoe in unique}
    for index in range(int(samples)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[shoe] for shoe in selected])
        estimates[index] = float(np.mean(values[rows]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def shoe_bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    shoe_ids: Sequence[str],
    *,
    samples: int = 1000,
    random_state: int = RANDOM_STATE,
) -> list[float]:
    numerator=np.asarray(numerator,dtype=np.float64); denominator=np.asarray(denominator,dtype=np.float64)
    shoes=np.asarray([str(value) for value in shoe_ids]); unique=np.unique(shoes)
    if len(numerator)!=len(denominator) or len(numerator)!=len(shoes):
        raise ValueError("bootstrap ratio inputs must be aligned")
    rng=np.random.default_rng(random_state); estimates=[]; indices={shoe:np.flatnonzero(shoes==shoe) for shoe in unique}
    for _ in range(max(1,int(samples))):
        selected=rng.choice(unique,size=len(unique),replace=True); rows=np.concatenate([indices[shoe] for shoe in selected])
        total=float(np.sum(denominator[rows]))
        if total>0: estimates.append(float(np.sum(numerator[rows])/total))
    point=float(np.sum(numerator)/np.sum(denominator)) if np.sum(denominator)>0 else 0.0
    return [float(value) for value in np.quantile(estimates,[.025,.975])] if len(estimates)>1 else [point,point]


def _accuracy(probability_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((probability_b > 0.50) == (actual_b > 0)))


def _brier(probability_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((probability_b.astype(float) - actual_b.astype(float)) ** 2))


def dynamic_ema_alpha(round_index: float, noise_score: float, config: Mapping[str, Any]) -> float:
    stage="early" if round_index<=40 else "late" if round_index>50 else "middle"
    lo,hi=EMA_ALPHA_RANGES[stage]
    base=float(config.get(f"{stage}_alpha",(lo+hi)/2.0)); gain=max(0.0,float(config.get("noise_gain",0.05)))
    return _clip(base+gain*(0.5-_clip(noise_score,0.0,1.0)),lo,hi)


def dynamic_ema_by_shoe(
    clipped_probability_b: np.ndarray,
    x: np.ndarray,
    shoe_ids: Sequence[str],
    config: Mapping[str, Any] | None,
) -> np.ndarray:
    """動態 Clip 後，以同靴過去值做 causal EMA；新靴首局不平滑。"""
    values=np.asarray(clipped_probability_b,dtype=np.float64).reshape(-1);features=np.asarray(x,dtype=np.float64)
    shoes=[str(value) for value in shoe_ids]; smoothing=dict(config or EMA_PROFILES["off"])
    if len(values)!=len(shoes) or len(values)!=len(features): raise ValueError("EMA inputs must be aligned")
    if not smoothing.get("enabled",False): return values.copy()
    output=np.empty_like(values);previous:dict[str,float]={}
    for index,(current,row,shoe_id) in enumerate(zip(values,features,shoes)):
        lo,hi=dynamic_probability_bounds(float(row[2]),float(row[-1]));current=_clip(float(current),lo,hi)
        alpha=dynamic_ema_alpha(float(row[2]),float(row[-1]),smoothing)
        smoothed=current if shoe_id not in previous else alpha*current+(1.0-alpha)*previous[shoe_id]
        output[index]=previous[shoe_id]=_clip(smoothed,lo,hi)
    return output


def bounded_from_calibrated(
    probability_b: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        _clip(pb, *dynamic_probability_bounds(float(row[2]), float(row[-1])))
        for pb, row in zip(np.asarray(probability_b,dtype=np.float64),np.asarray(x))
    ], dtype=np.float64)


def bounded_probabilities(
    model: Any,
    x: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncalibrated = _positive_probabilities(model, x)
    calibrated = apply_probability_calibration(uncalibrated, calibration)
    bounded=bounded_from_calibrated(calibrated,x)
    return uncalibrated, calibrated, bounded


def evaluate(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    probability_bounds: Sequence[float],
    calibration: Mapping[str, Any] | None = None,
    ev_thresholds: Mapping[str, float] | None = None,
    smoothing_config: Mapping[str, Any] | None = None,
    decision_policy_enabled: bool = False,
    decision_policy_config: Mapping[str, Any] | None = None,
    shoe_ids: Sequence[str] | None = None,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    shoe_ids = list(shoe_ids) if shoe_ids is not None else [str(index) for index in range(len(y))]
    uncalibrated,raw,unsmoothed=bounded_probabilities(model,x,calibration)
    smoothing=dict(smoothing_config or EMA_PROFILES["off"])
    final=dynamic_ema_by_shoe(unsmoothed,x,shoe_ids,smoothing)
    rounds = np.asarray(x[:, 2], dtype=np.float64)
    noise = np.asarray(x[:, -1], dtype=np.float64)
    thresholds = dict(DEFAULT_MIN_EV)
    thresholds.update(ev_thresholds or {})
    realised,wagered=decision_returns(final,y,rounds,thresholds,noise,policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=shoe_ids)
    pre_realised,pre_wagered=decision_returns(unsmoothed,y,rounds,thresholds,noise,policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=shoe_ids)
    baseline_realised,baseline_wagered=decision_returns(final,y,rounds,DEFAULT_MIN_EV,noise,policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=shoe_ids)
    hard_realised,hard_wagered=decision_returns(final,y,rounds,thresholds,noise,policy_enabled=False)
    current_realised,current_wagered=decision_returns(unsmoothed,y,rounds,thresholds,noise,policy_enabled=False)
    decision=decision_metrics(realised,wagered); pre=decision_metrics(pre_realised,pre_wagered); baseline=decision_metrics(baseline_realised,baseline_wagered)
    hard=decision_metrics(hard_realised,hard_wagered);current=decision_metrics(current_realised,current_wagered)
    skip_delta=decision["skip_rate"]-baseline["skip_rate"]
    policy_skip_delta=decision["skip_rate"]-hard["skip_rate"]
    policy_ev_delta=decision["realized_ev_per_bet"]-hard["realized_ev_per_bet"]
    policy_hit_delta=decision["hit_rate"]-hard["hit_rate"]
    policy_absolute_delta=decision["absolute_correct_bets"]-hard["absolute_correct_bets"]
    policy_absolute_passed=decision["absolute_correct_bets"]+1e-12>=.95*hard["absolute_correct_bets"]
    minimum_policy_skip=max(0.0,float((decision_policy_config or {}).get("min_skip_increase",0.0))) if decision_policy_enabled else 0.0
    policy_guard=(policy_skip_delta<=MAX_SKIP_RATE_INCREASE+1e-12 and policy_skip_delta+1e-12>=minimum_policy_skip and
                  policy_ev_delta>=-1e-12 and policy_hit_delta>=-1e-12 and policy_absolute_passed)
    correct = ((final > 0.50) == (y > 0)).astype(np.float64)
    pre_correct = ((unsmoothed > 0.50) == (y > 0)).astype(np.float64)
    smoothing_skip_delta=decision["skip_rate"]-pre["skip_rate"]
    smoothing_ev_delta=decision["realized_ev_per_bet"]-pre["realized_ev_per_bet"]
    smoothing_hit_delta=decision["hit_rate"]-pre["hit_rate"]
    smoothing_brier_delta=_brier(final,y)-_brier(unsmoothed,y)
    smoothing_guard=(smoothing_skip_delta<=1e-12 and smoothing_ev_delta>=-1e-12 and smoothing_hit_delta>=-1e-12 and smoothing_brier_delta<=MAX_SMOOTHING_BRIER_INCREASE)
    upgrade_delta={"realized_ev_per_bet":decision["realized_ev_per_bet"]-current["realized_ev_per_bet"],
                   "hit_rate":decision["hit_rate"]-current["hit_rate"],"skip_rate":decision["skip_rate"]-current["skip_rate"],
                   "absolute_correct_bets":decision["absolute_correct_bets"]-current["absolute_correct_bets"]}
    upgrade_guard=(upgrade_delta["realized_ev_per_bet"]>=-1e-12 and upgrade_delta["hit_rate"]>=-1e-12 and
                   upgrade_delta["skip_rate"]<=MAX_SKIP_RATE_INCREASE+1e-12 and decision["absolute_correct_bets"]+1e-12>=.95*current["absolute_correct_bets"])
    stage_masks={"early":rounds<=40,"middle":(rounds>40)&(rounds<=50),"late":rounds>50,"late_50_70":(rounds>50)&(rounds<=70)}; stage_report={}
    for stage,mask in stage_masks.items():
        stage_shoes=np.asarray(shoe_ids)[mask]
        stage_realised,stage_wagered=decision_returns(final[mask],y[mask],rounds[mask],thresholds,noise[mask],policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=stage_shoes)
        pre_stage_realised,pre_stage_wagered=decision_returns(unsmoothed[mask],y[mask],rounds[mask],thresholds,noise[mask],policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=stage_shoes)
        base_realised,base_wagered=decision_returns(final[mask],y[mask],rounds[mask],DEFAULT_MIN_EV,noise[mask],policy_enabled=decision_policy_enabled,policy_config=decision_policy_config,shoe_ids=stage_shoes)
        hard_stage_realised,hard_stage_wagered=decision_returns(final[mask],y[mask],rounds[mask],thresholds,noise[mask],policy_enabled=False)
        tuned_stage=decision_metrics(stage_realised,stage_wagered); pre_stage=decision_metrics(pre_stage_realised,pre_stage_wagered); base_stage=decision_metrics(base_realised,base_wagered)
        hard_stage=decision_metrics(hard_stage_realised,hard_stage_wagered)
        stage_rows=int(np.sum(mask))
        stage_report[stage]={**tuned_stage,
                             "overall_accuracy":_accuracy(final[mask],y[mask]) if stage_rows else None,
                             "brier":_brier(final[mask],y[mask]) if stage_rows else None,
                             "pre_smoothing_overall_accuracy":_accuracy(unsmoothed[mask],y[mask]) if stage_rows else None,
                             "pre_smoothing_brier":_brier(unsmoothed[mask],y[mask]) if stage_rows else None,
                             "baseline_skip_rate":base_stage["skip_rate"],"skip_rate_delta":tuned_stage["skip_rate"]-base_stage["skip_rate"],
                             "hard_policy_ev_per_bet":hard_stage["realized_ev_per_bet"],"hard_policy_skip_rate":hard_stage["skip_rate"],
                             "policy_ev_per_bet_delta":tuned_stage["realized_ev_per_bet"]-hard_stage["realized_ev_per_bet"],
                             "policy_hit_rate_delta":tuned_stage["hit_rate"]-hard_stage["hit_rate"],
                             "policy_skip_rate_delta":tuned_stage["skip_rate"]-hard_stage["skip_rate"],
                             "pre_smoothing_ev_per_bet":pre_stage["realized_ev_per_bet"],"pre_smoothing_skip_rate":pre_stage["skip_rate"],
                             "smoothing_ev_per_bet_delta":tuned_stage["realized_ev_per_bet"]-pre_stage["realized_ev_per_bet"],
                             "smoothing_hit_rate_delta":tuned_stage["hit_rate"]-pre_stage["hit_rate"],
                             "smoothing_skip_rate_delta":tuned_stage["skip_rate"]-pre_stage["skip_rate"]}
    return {
        "samples": float(len(y)),
        "uncalibrated_brier": _brier(uncalibrated, y),
        "raw_accuracy": _accuracy(raw, y),
        "raw_brier": _brier(raw, y),
        "overall_accuracy": _accuracy(final, y),
        "bounded_accuracy": _accuracy(final, y),
        "bounded_accuracy_ci95": shoe_bootstrap_ci(correct, shoe_ids, samples=bootstrap_samples),
        "bounded_brier": _brier(final, y),
        "bounded_brier_ci95": shoe_bootstrap_ci((final - y) ** 2, shoe_ids, samples=bootstrap_samples),
        "realized_ev_per_row": decision["realized_ev_per_row"],
        "realized_ev_per_bet": decision["realized_ev_per_bet"],
        "hit_rate_on_bets": decision["hit_rate"],
        "bet_accuracy": decision["hit_rate"],
        "absolute_correct_bets": decision["absolute_correct_bets"],
        "hit_rate_on_bets_ci95": shoe_bootstrap_ratio_ci((realised>0.0).astype(float),wagered,shoe_ids,samples=bootstrap_samples),
        "realized_ev_per_bet_ci95": shoe_bootstrap_ratio_ci(realised, wagered, shoe_ids, samples=bootstrap_samples),
        "realized_ev_per_row_ci95": shoe_bootstrap_ci(realised, shoe_ids, samples=bootstrap_samples),
        "action_rate": decision["action_rate"],
        "skip_rate": decision["skip_rate"],
        "skip_rate_ci95": shoe_bootstrap_ci((~wagered).astype(float),shoe_ids,samples=bootstrap_samples),
        "baseline_skip_rate": baseline["skip_rate"],
        "skip_rate_delta_vs_default": skip_delta,
        "skip_constraint_passed": bool(skip_delta <= MAX_SKIP_RATE_INCREASE + 1e-12),
        "baseline_realized_ev_per_bet": baseline["realized_ev_per_bet"],
        "stage_decision_metrics": stage_report,
        "evaluated_min_ev": thresholds,
        "upgrade_comparison": {"guardrail_passed":bool(upgrade_guard),"before":current,"after":decision,"delta":upgrade_delta,
                               "absolute_correct_bets_constraint_passed":bool(decision["absolute_correct_bets"]+1e-12>=.95*current["absolute_correct_bets"])},
        "decision_policy": {
            "enabled": bool(decision_policy_enabled),
            "config": dict(decision_policy_config or {}),
            "guardrail_passed": bool(policy_guard),
            "before": hard,
            "after": decision,
            "delta": {"realized_ev_per_bet": policy_ev_delta, "hit_rate": policy_hit_delta, "skip_rate": policy_skip_delta, "absolute_correct_bets": policy_absolute_delta},
            "minimum_skip_increase": minimum_policy_skip,
            "absolute_correct_bets_constraint_passed": bool(policy_absolute_passed),
        },
        "smoothing": {
            **smoothing,"guardrail_passed":bool(smoothing_guard),
            "before":{"accuracy":_accuracy(unsmoothed,y),"accuracy_ci95":shoe_bootstrap_ci(pre_correct,shoe_ids,samples=bootstrap_samples),
                      "brier":_brier(unsmoothed,y),"brier_ci95":shoe_bootstrap_ci((unsmoothed-y)**2,shoe_ids,samples=bootstrap_samples),
                      **pre,"ev_per_bet_ci95":shoe_bootstrap_ratio_ci(pre_realised,pre_wagered,shoe_ids,samples=bootstrap_samples),
                      "skip_rate_ci95":shoe_bootstrap_ci((~pre_wagered).astype(float),shoe_ids,samples=bootstrap_samples)},
            "after":{"accuracy":_accuracy(final,y),"brier":_brier(final,y),**decision},
            "delta":{"accuracy":_accuracy(final,y)-_accuracy(unsmoothed,y),"brier":smoothing_brier_delta,
                     "realized_ev_per_bet":smoothing_ev_delta,"hit_rate":smoothing_hit_delta,"skip_rate":smoothing_skip_delta},
        },
    }


def select_xgboost_model(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    trials: int = len(XGB_TUNING_CANDIDATES),
    random_state: int = RANDOM_STATE,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Primary search: bet hit-rate + EV + Brier, with a hard +5% Skip ceiling."""
    limit = max(1, min(int(trials), len(XGB_TUNING_CANDIDATES)))
    reports: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    baseline_metrics: dict[str, float] | None = None
    weights = balanced_sample_weights(y[train], x[train])
    rounds = np.asarray(x[validation, 2], dtype=np.float64)
    for index, parameters in enumerate(XGB_TUNING_CANDIDATES[:limit]):
        model = build_xgboost_classifier(random_state=random_state + index, overrides=parameters)
        model.fit(x[train], y[train], sample_weight=weights)
        _, _, probability = bounded_probabilities(model, x[validation])
        realised, wagered = decision_returns(probability, y[validation], rounds, DEFAULT_MIN_EV)
        brier = _brier(probability, y[validation])
        metrics=decision_metrics(realised,wagered)
        accuracy=_accuracy(probability,y[validation])
        if baseline_metrics is None: baseline_metrics={**metrics,"accuracy":accuracy,"brier":brier}
        skip_delta=metrics["skip_rate"]-baseline_metrics["skip_rate"]
        brier_passed=brier<=baseline_metrics["brier"]+MAX_MODEL_BRIER_REGRESSION+1e-12
        quality_passed=(metrics["hit_rate"]+1e-12>=baseline_metrics["hit_rate"] and
                        metrics["realized_ev_per_bet"]+1e-12>=baseline_metrics["realized_ev_per_bet"] and brier_passed)
        skip_passed=skip_delta<=MAX_SKIP_RATE_INCREASE+1e-12
        eligible=skip_passed and quality_passed
        score=brier-.04*metrics["realized_ev_per_bet"]-.02*metrics["hit_rate"]-.05*metrics["realized_ev_per_row"]+.02*max(0.0,skip_delta)
        report = {
            "candidate": index,
            "score": score,
            "bounded_brier": brier,
            "bounded_accuracy": accuracy,
            "hit_rate_on_bets": metrics["hit_rate"],
            "realized_ev_per_row": metrics["realized_ev_per_row"],
            "realized_ev_per_bet": metrics["realized_ev_per_bet"],
            "action_rate": metrics["action_rate"],
            "skip_rate": metrics["skip_rate"],
            "skip_rate_delta_vs_legacy": skip_delta,
            "skip_constraint_passed": skip_passed,
            "brier_constraint_passed": brier_passed,
            "quality_constraint_passed": quality_passed,
            "candidate_eligible": eligible,
            "parameters": dict(parameters),
        }
        reports.append(report)
        if eligible and (best is None or score < best[0]):
            best = (score, model, dict(parameters))
    assert best is not None
    return best[1], best[2], reports


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node: Mapping[str, Any] = tree
    for _ in range(256):
        if "leaf" in node:
            return float(node.get("leaf", 0.0))
        split = str(node.get("split", ""))
        index = int(split[1:]) if split.startswith("f") and split[1:].isdigit() else FEATURE_NAMES.index(split)
        value = float(np.float32(vector[index])) if 0 <= index < len(vector) else math.nan
        threshold = float(np.float32(node.get("split_condition", 0.0)))
        next_id = node.get("missing") if not math.isfinite(value) else (node.get("yes") if value < threshold else node.get("no"))
        node = next((child for child in node.get("children") or [] if int(child.get("nodeid", -1)) == int(next_id)), {})
        if not node:
            return 0.0
    return 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def _logit(probability: float) -> float:
    probability = _clip(probability, 1e-7, 1.0 - 1e-7)
    return math.log(probability / (1.0 - probability))


def export_browser_bundle(
    model: Any,
    reference_x: np.ndarray,
    output_path: str | Path,
    *,
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
    ev_thresholds: Mapping[str, float] | None = None,
    smoothing_config: Mapping[str, Any] | None = None,
    decision_policy: Mapping[str, Any] | None = None,
    physics_noise_calibration: Mapping[str, Any] | None = None,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Export sigmoid-linked classifier trees for the static browser runtime."""
    lo, hi = _probability_bounds(probability_bounds)
    trees = [json.loads(item) for item in model.get_booster().get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=np.float32)
    tree_sum = sum(_tree_leaf(tree, reference) for tree in trees)
    native_pb = float(_positive_probabilities(model, reference.reshape(1, -1))[0])
    base_margin = _logit(native_pb) - tree_sum

    for vector in np.asarray(reference_x[: min(128, len(reference_x))], dtype=np.float32):
        portable_pb = _sigmoid(base_margin + sum(_tree_leaf(tree, vector) for tree in trees))
        native_pb = float(_positive_probabilities(model, vector.reshape(1, -1))[0])
        if abs(portable_pb - native_pb) > 2e-5:
            raise RuntimeError(f"portable probability mismatch {portable_pb} vs {native_pb}")

    bundle = {
        "schema_version": 2,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "link": "sigmoid",
        "base_margin": float(base_margin),
        "probability_bounds": [lo, hi],
        "calibration": dict(calibration or {"method": "identity", "slope": 1.0, "intercept": 0.0}),
        "ev_thresholds": dict(ev_thresholds or DEFAULT_MIN_EV),
        "smoothing": dict(smoothing_config or EMA_PROFILES["off"]),
        "decision_policy": dict(decision_policy or {"enabled": False}),
        "trees": trees,
        "training": {
            "target": "actual_b_binary",
            "label_mapping": {"P": 0, "B": 1},
            "residual": False,
            "noise_score_version": 4,
            "physics_noise_calibration": dict(physics_noise_calibration or {"method":"identity","x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.0]}),
            "skip_guardrail": {"preferred_max_increase": PREFERRED_SKIP_RATE_INCREASE, "hard_max_increase": MAX_SKIP_RATE_INCREASE},
            "feature_snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "metrics": dict(metrics),
        },
    }
    Path(output_path).write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    extractor = PhysicsFeatureExtractor.load(args.physics_model) if args.physics_model else None
    records = load_training_records(Path(args.input))
    x, y, training_records = make_training_dataset(records, physics_extractor=extractor)
    physics_noise_calibration=dict(getattr(extractor,"uncertainty_calibration",{}) or {}) if extractor is not None else {}
    if physics_noise_calibration:
        x=recalibrate_noise_feature(x,training_records,physics_noise_calibration)
    if len(x) < args.min_samples:
        raise SystemExit(f"need {args.min_samples} rows; got {len(x)}")
    shoe_count = len({str(record.get("shoe_id") or "").strip() for record in training_records})
    if shoe_count < args.min_shoes:
        raise SystemExit(f"need {args.min_shoes} shoes; got {shoe_count}")

    train, calibration_rows, holdout = three_way_shoe_masks(
        training_records,
        calibration_fraction=args.calibration_fraction,
        holdout_fraction=args.validation_fraction,
    )
    search_train, tuning_rows = nested_tuning_masks(
        training_records,
        train,
        fraction=args.tuning_fraction,
    )
    probability_calibration_rows,ev_tuning_rows=nested_tuning_masks(
        training_records,
        calibration_rows,
        fraction=args.ev_tuning_fraction,
    )
    _, best_parameters, search_report = select_xgboost_model(
        x,
        y,
        search_train,
        tuning_rows,
        trials=args.tune_trials,
        random_state=args.random_state,
    )
    model = build_xgboost_classifier(random_state=args.random_state, overrides=best_parameters)
    model.fit(x[train], y[train], sample_weight=balanced_sample_weights(y[train], x[train]))
    calibration = fit_probability_calibration(
        model,
        x[probability_calibration_rows],
        y[probability_calibration_rows],
        shoe_ids=[str(record["shoe_id"]) for record,selected in zip(training_records,probability_calibration_rows) if selected],
    )
    _, calibration_probability, _ = bounded_probabilities(model, x[ev_tuning_rows], calibration)
    ev_tuning_shoe_ids=[str(record["shoe_id"]) for record,selected in zip(training_records,ev_tuning_rows) if selected]
    smoothing_tuning=optimize_smoothing_and_thresholds(
        calibration_probability,y[ev_tuning_rows],x[ev_tuning_rows],ev_tuning_shoe_ids,
        profiles=args.smoothing_profiles,
        decision_policies={name: DECISION_POLICY_PROFILES[name] for name in args.decision_policy_profiles},
    )
    ev_tuning=smoothing_tuning["ev_tuning"]
    smoothing_config=dict(smoothing_tuning["smoothing"])
    decision_policy=dict(smoothing_tuning["decision_policy"])
    decision_policy_enabled=bool(decision_policy["enabled"])
    holdout_records = [record for record, selected in zip(training_records, holdout) if selected]
    metrics = evaluate(
        model,
        x[holdout],
        y[holdout],
        probability_bounds=args.probability_bounds,
        calibration=calibration,
        ev_thresholds=ev_tuning["thresholds"],
        smoothing_config=smoothing_config,
        decision_policy_enabled=decision_policy_enabled,
        decision_policy_config=decision_policy,
        shoe_ids=[str(record["shoe_id"]) for record in holdout_records],
        bootstrap_samples=args.bootstrap_samples,
    )
    legacy_x=legacy_feature_matrix(x)
    legacy_model=build_xgboost_classifier(random_state=args.random_state,overrides=LEGACY_XGB_PARAMETERS)
    legacy_model.fit(legacy_x[train],y[train])
    _,_,legacy_probability=bounded_probabilities(legacy_model,legacy_x[holdout])
    legacy_realised,legacy_wagered=decision_returns(legacy_probability,y[holdout],legacy_x[holdout,2],DEFAULT_MIN_EV)
    legacy_decision=decision_metrics(legacy_realised,legacy_wagered)
    legacy_accuracy=_accuracy(legacy_probability,y[holdout]);legacy_brier=_brier(legacy_probability,y[holdout])
    deployment_skip_delta=metrics["skip_rate"]-legacy_decision["skip_rate"]
    hit_delta=metrics["hit_rate_on_bets"]-legacy_decision["hit_rate"]
    ev_delta=metrics["realized_ev_per_bet"]-legacy_decision["realized_ev_per_bet"]
    correct_delta=metrics["absolute_correct_bets"]-legacy_decision["absolute_correct_bets"]
    deployment_brier_passed=metrics["bounded_brier"]<=legacy_brier+MAX_MODEL_BRIER_REGRESSION+1e-12
    deployment_primary_non_regressing=(hit_delta>=-1e-12 and ev_delta>=-1e-12 and metrics["absolute_correct_bets"]+1e-12>=.95*legacy_decision["absolute_correct_bets"])
    deployment_primary_improved=(hit_delta>MIN_PRIMARY_GAIN or ev_delta>MIN_PRIMARY_GAIN)
    deployment_quality_passed=deployment_primary_non_regressing and deployment_primary_improved and deployment_brier_passed
    deployment_smoothing_passed=bool(metrics["smoothing"]["guardrail_passed"])
    deployment_policy_passed=bool(metrics["decision_policy"]["guardrail_passed"])
    deployment_upgrade_passed=bool(metrics["upgrade_comparison"]["guardrail_passed"])
    deployment_skip_passed=bool(deployment_skip_delta<=MAX_SKIP_RATE_INCREASE+1e-12)
    deployment_min_skip_increase=max(0.0,float(decision_policy.get("min_skip_increase",0.0))) if decision_policy_enabled else 0.0
    deployment_skip_clear_passed=bool(deployment_skip_delta+1e-12>=deployment_min_skip_increase)
    deployment_absolute_correct_passed=bool(metrics["absolute_correct_bets"]+1e-12>=.95*legacy_decision["absolute_correct_bets"])
    retraining_success=bool(deployment_quality_passed and deployment_skip_passed and deployment_skip_clear_passed and deployment_absolute_correct_passed and deployment_smoothing_passed and deployment_policy_passed and deployment_upgrade_passed)
    metrics.update({
        "validation_strategy": "chronological_shoe_train_tune_calibrate_ev_tune_strict_holdout",
        "training_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, train) if selected})),
        "tuning_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, tuning_rows) if selected})),
        "calibration_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, calibration_rows) if selected})),
        "probability_calibration_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, probability_calibration_rows) if selected})),
        "ev_tuning_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, ev_tuning_rows) if selected})),
        "holdout_shoes": int(len({str(record["shoe_id"]) for record, selected in zip(training_records, holdout) if selected})),
        "calibration": calibration,
        "recommended_min_ev": ev_tuning,
        "selected_decision_policy": decision_policy,
        "smoothing_selection": smoothing_tuning,
        "hyperparameter_search": search_report,
        "selected_parameters": best_parameters,
        "feature_importance": feature_importance_report(model, x[calibration_rows]),
        "chronology_validation": validate_chronological_shoes(training_records),
        "legacy_baseline_accuracy": legacy_accuracy,
        "legacy_baseline_brier": legacy_brier,
        "legacy_baseline_hit_rate_on_bets": legacy_decision["hit_rate"],
        "legacy_baseline_absolute_correct_bets": legacy_decision["absolute_correct_bets"],
        "legacy_baseline_skip_rate": legacy_decision["skip_rate"],
        "legacy_baseline_realized_ev_per_bet": legacy_decision["realized_ev_per_bet"],
        "skip_rate_delta_vs_legacy": deployment_skip_delta,
        "deployment_skip_constraint_passed": deployment_skip_passed,
        "deployment_minimum_skip_increase": deployment_min_skip_increase,
        "deployment_skip_clear_constraint_passed": deployment_skip_clear_passed,
        "deployment_absolute_correct_bets_constraint_passed": deployment_absolute_correct_passed,
        "deployment_quality_constraint_passed": bool(deployment_quality_passed),
        "deployment_brier_constraint_passed": bool(deployment_brier_passed),
        "primary_hit_rate_delta_vs_legacy": hit_delta,
        "primary_ev_per_bet_delta_vs_legacy": ev_delta,
        "absolute_correct_bets_delta_vs_legacy": correct_delta,
        "retraining_success": retraining_success,
        "physics_noise_calibration": physics_noise_calibration,
        "ci_smoke_override": bool(args.ci_smoke),
        "deployment_smoothing_constraint_passed": deployment_smoothing_passed,
        "deployment_decision_policy_constraint_passed": deployment_policy_passed,
        "deployment_upgrade_constraint_passed": deployment_upgrade_passed,
    })
    deployment_failures: list[str] = []
    if not metrics["deployment_skip_constraint_passed"]:
        deployment_failures.append(f"holdout Skip increased by {deployment_skip_delta:.4f} (> {MAX_SKIP_RATE_INCREASE:.4f})")
    if not metrics["deployment_skip_clear_constraint_passed"]:
        deployment_failures.append(f"holdout Skip increased by {deployment_skip_delta:.4f} (< required {deployment_min_skip_increase:.4f})")
    if not metrics["deployment_absolute_correct_bets_constraint_passed"]:
        deployment_failures.append("holdout absolute correct bets fell below 95% of the legacy baseline")
    if not metrics["deployment_quality_constraint_passed"]:
        deployment_failures.append("primary holdout bet hit-rate/EV did not improve, correct-bet volume fell, or Brier regressed")
    if not metrics["deployment_smoothing_constraint_passed"]:
        deployment_failures.append("smoothing reduced holdout hit-rate/EV, worsened Brier, or increased Skip")
    if not metrics["deployment_decision_policy_constraint_passed"]:
        deployment_failures.append("soft EV policy reduced holdout hit-rate/EV or increased Skip")
    if not metrics["deployment_upgrade_constraint_passed"]:
        deployment_failures.append("combined upgrade reduced holdout hit-rate/EV or increased Skip")
    metrics["deployment_blocked"] = bool(deployment_failures)
    metrics["deployment_block_reasons"] = deployment_failures
    report_payload={"holdout":metrics}
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    if args.report_output:
        Path(args.report_output).write_text(json.dumps(report_payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if deployment_failures and not args.ci_smoke:
        raise SystemExit("deployment blocked: " + "; ".join(deployment_failures))
    if deployment_failures:
        print(json.dumps({"ci_smoke_override": True, "deployment_blocked": True, "reasons": deployment_failures}, ensure_ascii=False))

    # Deploy the exact model evaluated above; refitting on calibration rows would
    # invalidate both its probability calibrator and the strict holdout report.
    deploy = train
    final_model = model
    final_model.bbb_calibration_ = calibration
    final_model.bbb_ev_thresholds_ = ev_tuning["thresholds"]
    final_model.bbb_smoothing_ = smoothing_config
    final_model.bbb_decision_policy_ = decision_policy
    final_model.bbb_physics_noise_calibration_ = physics_noise_calibration
    if args.joblib_output:
        joblib.dump(final_model, args.joblib_output)
    export_browser_bundle(
        final_model,
        x[deploy],
        args.output,
        metrics=metrics,
        calibration=calibration,
        ev_thresholds=ev_tuning["thresholds"],
        smoothing_config=smoothing_config,
        decision_policy=decision_policy,
        physics_noise_calibration=physics_noise_calibration,
        probability_bounds=args.probability_bounds,
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the direct 57D final-probability XGBoost model")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--physics-model", default="")
    train.add_argument("--output", default="final_probability_model.json")
    train.add_argument("--report-output", default="")
    train.add_argument("--joblib-output", default="")
    train.add_argument("--min-samples", type=int, default=12000)
    train.add_argument("--min-shoes", type=int, default=1000)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--calibration-fraction", type=float, default=0.15)
    train.add_argument("--tuning-fraction", type=float, default=0.15)
    train.add_argument("--ev-tuning-fraction", type=float, default=0.30)
    train.add_argument("--smoothing-profiles",nargs="+",choices=tuple(EMA_PROFILES),default=list(EMA_PROFILES))
    train.add_argument("--decision-policy-profiles", nargs="+", choices=tuple(DECISION_POLICY_PROFILES), default=list(DECISION_POLICY_PROFILES))
    train.add_argument("--bootstrap-samples", type=int, default=1000)
    train.add_argument("--tune-trials", type=int, default=len(XGB_TUNING_CANDIDATES))
    train.add_argument("--ci-smoke", action="store_true", help="continue artifact/runtime smoke validation even when deployment quality gates fail; never use for production promotion")
    train.add_argument("--probability-bounds", nargs=2, type=float, default=PROBABILITY_BOUNDS, metavar=("MIN", "MAX"))
    train.add_argument("--random-state", type=int, default=RANDOM_STATE)
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
