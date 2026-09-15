#!/usr/bin/env python3
"""BBB XGBoost residual-bias predictor with 8/4 S/X defense and core-streak sizing.

The deterministic 256D/V23 Frozen Base remains untouched. XGBRegressor learns
only the calibration residual of the core B probability:

    residual = actual_B - core_p_B

Local regime features intentionally avoid half-shoe B/P totals:
- sx_volatility_8: normalized population std of the latest eight S/X tokens.
  Ties are removed before S/X construction, so T never changes road geometry.
- sx_micro_run_4: signed suffix-run strength inside the latest four S/X tokens.
- tie_density: T count in the latest eight raw outcomes divided by 8.
- core_streak: signed consecutive correctness streak of the *Frozen Base*
  direction before residual correction; wins are positive, losses are negative.

Production decision:

    delta = clip(xgb_residual, -0.12, +0.12)
    final_p_B = 0.50 + ((core_p_B - 0.50) + delta) * damper
    direction = "B" if final_p_B > 0.50 else "P"

There is never a PASS state. Bet weight is driven by final edge and core_streak.
Severe tie/oscillation defense forces bottom weight; otherwise core_streak >= 2
immediately unlocks full attack weight 1.0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from xgboost import XGBRegressor

FEATURE_NAMES: tuple[str, ...] = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "stage",
    "depth",
    "sx_volatility_8",
    "sx_micro_run_4",
    "tie_density",
    "core_streak",
)
MODEL_TYPE = "xgb_residual_regressor"
SCHEMA_VERSION = 7
DEFAULT_MAX_DELTA = 0.12
DEFAULT_RANDOM_STATE = 20260915

DEFAULT_VOLATILITY_WINDOW = 8
DEFAULT_VOLATILITY_MIN_TOKENS = 4
DEFAULT_MICRO_WINDOW = 4
DEFAULT_TIE_WINDOW = 8
DEFAULT_CORE_STREAK_CAP = 8

DEFAULT_REGULAR_VOLATILITY_MAX = 0.35
DEFAULT_HIGH_VOLATILITY_START = 0.85
DEFAULT_HIGH_DAMPER_MAX = 0.40
DEFAULT_HIGH_DAMPER_MIN = 0.20
DEFAULT_FAST_UNLOCK_RUN = 0.75
DEFAULT_TIE_DAMPING_POWER = 3.20
DEFAULT_TIE_DAMPER_MIN = 0.20

DEFAULT_BET_WEIGHT_MIN = 0.20
DEFAULT_BET_REFERENCE_EDGE = 0.08
DEFAULT_ATTACK_STREAK_TRIGGER = 2
DEFAULT_ATTACK_SINGLE_WIN_MULTIPLIER = 1.35
DEFAULT_DEFENSE_TIE_DENSITY = 0.25
DEFAULT_DEFENSE_VOLATILITY = 0.90
DEFAULT_DEFENSE_DAMPER = 0.40


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def clip_signed(value: float, limit: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    limit = abs(float(limit))
    return max(-limit, min(limit, value))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    values: Iterable[Any]
    if isinstance(history, str):
        values = [x for x in history.upper() if x in {"B", "P", "T"}]
    else:
        values = history
    out: list[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out


def transition_sequence(history: Sequence[str]) -> list[str]:
    """Build S/X only after removing all T outcomes.

    SAME means consecutive non-tie outcomes are equal; SWITCH means they differ.
    T therefore never opens a new S/X token and cannot distort road geometry.
    """
    values = [x for x in history if x in {"B", "P"}]
    return ["S" if values[i] == values[i - 1] else "X" for i in range(1, len(values))]


def sx_volatility(
    history: Sequence[str],
    *,
    window: int = DEFAULT_VOLATILITY_WINDOW,
    min_tokens: int = DEFAULT_VOLATILITY_MIN_TOKENS,
) -> float:
    """Normalized rolling population std of recent S/X in [0, 1]."""
    tokens = transition_sequence(history)[-max(2, int(window)) :]
    if len(tokens) < max(2, int(min_tokens)):
        return 0.0
    values = [1.0 if token == "S" else 0.0 for token in tokens]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return clip(math.sqrt(variance) / 0.5)


def sx_micro_run(history: Sequence[str], *, window: int = DEFAULT_MICRO_WINDOW) -> float:
    """Signed suffix-run strength of latest four S/X states.

    +1 is four SAME states; -1 is four SWITCH states. Three identical suffix
    states produce +/-0.75 and qualify for fast damper unlock.
    """
    width = max(2, int(window))
    tokens = transition_sequence(history)[-width:]
    if len(tokens) < width:
        return 0.0
    latest = tokens[-1]
    run = 1
    for token in reversed(tokens[:-1]):
        if token != latest:
            break
        run += 1
    sign = 1.0 if latest == "S" else -1.0
    return clip_signed(sign * (run / width))


def tie_density(history: Sequence[str], *, window: int = DEFAULT_TIE_WINDOW) -> float:
    """T density in latest eight raw outcomes, using the requested fixed /8 base."""
    width = max(1, int(window))
    recent = list(history)[-width:]
    return clip(sum(outcome == "T" for outcome in recent) / float(width))


def current_stage(history: Sequence[str]) -> int:
    values = [x for x in history if x in {"B", "P"}]
    if not values:
        return 0
    side = values[-1]
    length = 1
    for i in range(len(values) - 2, -1, -1):
        if values[i] != side:
            break
        length += 1
    return length


def current_depth(history: Sequence[str]) -> int:
    tokens = transition_sequence(history)
    if not tokens:
        return 0
    token = tokens[-1]
    depth = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != token:
            break
        depth += 1
    return depth


def update_core_streak(previous: int | float, core_direction: str, actual_outcome: str) -> int:
    """Update signed Frozen-Base correctness streak; T is neutral."""
    previous_i = int(previous or 0)
    direction = str(core_direction or "").upper().strip()
    actual = str(actual_outcome or "").upper().strip()
    if actual == "T" or direction not in {"B", "P"} or actual not in {"B", "P"}:
        return previous_i
    correct = direction == actual
    if correct:
        return previous_i + 1 if previous_i > 0 else 1
    return previous_i - 1 if previous_i < 0 else -1


def _regime_damper(
    volatility_8: float,
    micro_run_4: float,
    *,
    regular_max: float = DEFAULT_REGULAR_VOLATILITY_MAX,
    high_start: float = DEFAULT_HIGH_VOLATILITY_START,
    high_damper_max: float = DEFAULT_HIGH_DAMPER_MAX,
    high_damper_min: float = DEFAULT_HIGH_DAMPER_MIN,
    fast_unlock_run: float = DEFAULT_FAST_UNLOCK_RUN,
) -> float:
    v = clip(volatility_8)
    if abs(clip_signed(micro_run_4)) >= clip(fast_unlock_run):
        return 1.0

    regular_max = clip(regular_max, 0.0, 0.95)
    high_start = clip(high_start, regular_max + 1e-6, 0.999999)
    high_damper_max = clip(high_damper_max, 0.20, 1.0)
    high_damper_min = clip(high_damper_min, 0.0, high_damper_max)

    if v <= regular_max:
        return 1.0
    if v < high_start:
        t = (v - regular_max) / (high_start - regular_max)
        return clip(1.0 - (1.0 - high_damper_max) * (t * t), high_damper_max, 1.0)
    t = (v - high_start) / (1.0 - high_start)
    return clip(
        high_damper_max - (high_damper_max - high_damper_min) * (t * t),
        high_damper_min,
        high_damper_max,
    )


def _tie_damper(
    density: float,
    *,
    power: float = DEFAULT_TIE_DAMPING_POWER,
    minimum: float = DEFAULT_TIE_DAMPER_MIN,
) -> float:
    """Independent tie defense. Small density is mild; high density collapses fast."""
    d = clip(density)
    return clip(1.0 - max(0.0, float(power)) * (d * d), clip(minimum), 1.0)


def dynamic_damper(
    volatility_8: float,
    micro_run_4: float = 0.0,
    tie_density_value: float = 0.0,
) -> float:
    """Combine S/X regime damping and tie-density defense.

    A fresh 3-4 state micro pattern can unlock the S/X component immediately,
    while a genuinely high tie density can still cap risk defensively.
    """
    regime = _regime_damper(volatility_8, micro_run_4)
    tie_guard = _tie_damper(tie_density_value)
    return min(regime, tie_guard)


def bet_weight_from_probability(
    final_p_b: float,
    *,
    core_streak: float = 0.0,
    tie_density_value: float = 0.0,
    volatility_8: float = 0.0,
    damper: float = 1.0,
    minimum: float = DEFAULT_BET_WEIGHT_MIN,
    reference_edge: float = DEFAULT_BET_REFERENCE_EDGE,
) -> tuple[float, str, str]:
    """Edge sizing plus explicit attack/defense gates.

    Defense has priority when the environment is severely noisy. Outside severe
    defense, a Frozen-Base win streak >=2 immediately sets weight to 1.0.
    """
    minimum = clip(minimum)
    edge = abs(clip(final_p_b) - 0.5)
    reference_edge = max(1e-6, float(reference_edge))
    weight = minimum + (1.0 - minimum) * clip(edge / reference_edge)

    severe_defense = (
        clip(tie_density_value) >= DEFAULT_DEFENSE_TIE_DENSITY
        or clip(volatility_8) >= DEFAULT_DEFENSE_VOLATILITY
        or clip(damper) <= DEFAULT_DEFENSE_DAMPER
    )
    streak = int(round(float(core_streak)))

    if severe_defense:
        weight = minimum
        mode = "DEFENSE"
    elif streak >= DEFAULT_ATTACK_STREAK_TRIGGER:
        weight = 1.0
        mode = "ATTACK"
    elif streak == 1:
        weight = clip(weight * DEFAULT_ATTACK_SINGLE_WIN_MULTIPLIER, minimum, 1.0)
        mode = "ATTACK_WARMUP"
    elif streak <= -2:
        weight = max(minimum, weight * 0.60)
        mode = "LOSS_DEFENSE"
    elif streak == -1:
        weight = max(minimum, weight * 0.80)
        mode = "LOSS_CAUTION"
    else:
        mode = "EDGE"

    tier = "LOW" if weight < 0.40 else "MEDIUM" if weight < 0.70 else "HIGH"
    return float(weight), tier, mode


@dataclass(frozen=True)
class ResidualFeatures:
    core_p_b: float
    round_index: float
    estimated_total_hands: float
    remaining_ratio: float
    stage: float
    depth: float
    sx_volatility_8: float
    sx_micro_run_4: float
    tie_density: float
    core_streak: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float32)


def build_features(
    *,
    core_p_b: float,
    history: str | Sequence[str],
    estimated_total_hands: float = 60.0,
    stage: float | None = None,
    depth: float | None = None,
    core_streak: float = 0.0,
) -> ResidualFeatures:
    seq = normalize_history(history)
    total_hands = clip(float(estimated_total_hands), 40.0, 90.0)
    round_index = float(max(1, min(70, len(seq) + 1)))
    remaining_ratio = clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    streak_value = clip_signed(core_streak, DEFAULT_CORE_STREAK_CAP)
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b)),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        stage=float(current_stage(seq) if stage is None else abs(float(stage))),
        depth=float(current_depth(seq) if depth is None else depth),
        sx_volatility_8=sx_volatility(seq),
        sx_micro_run_4=sx_micro_run(seq),
        tie_density=tie_density(seq),
        core_streak=float(streak_value),
    )


def build_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=240,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=0.20,
        reg_lambda=8.0,
        gamma=0.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


class TieStreakResidualBiasPredictor:
    """Residual XGB + 8/4 local regime defense + tie feature + core streak sizing."""

    def __init__(
        self,
        model: XGBRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
        bet_weight_min: float = DEFAULT_BET_WEIGHT_MIN,
        bet_reference_edge: float = DEFAULT_BET_REFERENCE_EDGE,
    ) -> None:
        self.model = model or build_regressor()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)
        self.bet_weight_min = float(bet_weight_min)
        self.bet_reference_edge = float(bet_reference_edge)

    def assemble_features(self, **kwargs: Any) -> ResidualFeatures:
        return build_features(**kwargs)

    def fit(self, feature_rows: np.ndarray, actual_b: Sequence[int]) -> "TieStreakResidualBiasPredictor":
        x = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(actual_b, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"feature_rows must have {len(FEATURE_NAMES)} columns")
        if len(x) != len(y):
            raise ValueError("feature_rows and actual_b must have equal length")
        residual = y - x[:, FEATURE_NAMES.index("core_p_b")]
        self.model.fit(x, residual)
        return self

    def predict_delta(self, feature_row: Sequence[float]) -> float:
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"feature_row must have {len(FEATURE_NAMES)} values")
        raw = float(self.model.predict(x)[0])
        return clip(raw, -self.max_delta, self.max_delta)

    def correct(self, feature_row: Sequence[float]) -> dict[str, Any]:
        x = np.asarray(feature_row, dtype=np.float32)
        if x.size != len(FEATURE_NAMES):
            raise ValueError(f"feature_row must have {len(FEATURE_NAMES)} values")

        core_pb = clip(float(x[FEATURE_NAMES.index("core_p_b")]))
        volatility = clip(float(x[FEATURE_NAMES.index("sx_volatility_8")]))
        micro = clip_signed(float(x[FEATURE_NAMES.index("sx_micro_run_4")]))
        ties = clip(float(x[FEATURE_NAMES.index("tie_density")]))
        streak = clip_signed(float(x[FEATURE_NAMES.index("core_streak")]), DEFAULT_CORE_STREAK_CAP)

        delta = self.predict_delta(x)
        damper = dynamic_damper(volatility, micro, ties)
        undamped_pb = clip(core_pb + delta)
        final_pb = clip(0.5 + ((core_pb - 0.5) + delta) * damper)
        bet_weight, bet_tier, bet_mode = bet_weight_from_probability(
            final_pb,
            core_streak=streak,
            tie_density_value=ties,
            volatility_8=volatility,
            damper=damper,
            minimum=self.bet_weight_min,
            reference_edge=self.bet_reference_edge,
        )
        return {
            "core_p_b": core_pb,
            "delta": delta,
            "sx_volatility_8": volatility,
            "sx_micro_run_4": micro,
            "tie_density": ties,
            "core_streak": streak,
            "damper": damper,
            "undamped_p_b": undamped_pb,
            "final_p_b": final_pb,
            "direction": "B" if final_pb > 0.5 else "P",
            "bet_weight": bet_weight,
            "bet_weight_tier": bet_tier,
            "bet_weight_mode": bet_mode,
        }

    def predict_from_context(self, **kwargs: Any) -> dict[str, Any]:
        features = self.assemble_features(**kwargs)
        result = self.correct(features.as_vector())
        result["features"] = features.as_dict()
        return result


DynamicResidualBiasPredictor = TieStreakResidualBiasPredictor
ResidualBiasPredictor = TieStreakResidualBiasPredictor


def _parse_actual_b(record: Mapping[str, Any]) -> int:
    if "actual_b" in record and record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").upper().strip()
    if actual == "B":
        return 1
    if actual == "P":
        return 0
    raise ValueError("training row must contain actual_outcome B/P or actual_b 0/1")


def _feature_row(record: Mapping[str, Any], *, reconstructed_core_streak: float = 0.0) -> dict[str, float]:
    schema = int(float(record.get("schema_version", 0) or 0))
    if schema >= SCHEMA_VERSION and all(name in record for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}

    history = record.get("history") or record.get("history_fingerprint") or ""
    features = build_features(
        core_p_b=float(record.get("core_p_b", record.get("core_pb", 0.5))),
        history=history,
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
        core_streak=reconstructed_core_streak,
    ).as_dict()
    for name in ("core_p_b", "round_index", "estimated_total_hands", "remaining_ratio", "stage", "depth"):
        if name in record and record.get(name) is not None:
            features[name] = float(record[name])
    return features


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


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    residuals: list[float] = []
    actuals: list[int] = []
    shoes: list[str] = []
    streak_by_shoe: dict[str, int] = {}

    for idx, record in enumerate(records):
        shoe = str(record.get("shoe_id") or "legacy")
        prior_streak = streak_by_shoe.get(shoe, 0)
        try:
            actual_b = _parse_actual_b(record)
            row = _feature_row(record, reconstructed_core_streak=prior_streak)
            vector = [float(row[name]) for name in FEATURE_NAMES]
            if not all(math.isfinite(v) for v in vector):
                continue
            core_p_b = clip(row["core_p_b"])
        except (TypeError, ValueError, KeyError):
            continue

        vectors.append(vector)
        residuals.append(float(actual_b) - core_p_b)
        actuals.append(actual_b)
        shoes.append(str(record.get("shoe_id") or f"row_{idx}"))

        actual = "B" if actual_b else "P"
        core_direction = str(record.get("core_direction") or "").upper().strip()
        streak_by_shoe[shoe] = update_core_streak(prior_streak, core_direction, actual)

    if not vectors:
        raise ValueError("no valid B/P training rows")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(actuals, dtype=np.int8),
        shoes,
    )


def deterministic_validation_mask(shoes: Sequence[str], *, fraction: float = 0.20) -> np.ndarray:
    threshold = int(256 * clip(fraction, 0.05, 0.50))
    result = np.asarray(
        [hashlib.sha256(str(shoe).encode("utf-8")).digest()[0] < threshold for shoe in shoes],
        dtype=bool,
    )
    if result.all() or (~result).all():
        cut = max(1, int(round(len(result) * (1.0 - fraction))))
        result[:] = False
        result[cut:] = True
    return result


def direction_accuracy(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b > 0.5) == (actual_b > 0)))


def brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def _apply_final_formula(
    core_pb: np.ndarray,
    delta: np.ndarray,
    volatility: np.ndarray,
    micro: np.ndarray,
    ties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dampers = np.asarray(
        [dynamic_damper(float(v), float(m), float(t)) for v, m, t in zip(volatility, micro, ties, strict=False)],
        dtype=float,
    )
    final_pb = np.clip(0.5 + ((core_pb - 0.5) + delta) * dampers, 0.0, 1.0)
    return final_pb, dampers


def evaluate(model: XGBRegressor, x: np.ndarray, actual_b: np.ndarray, *, max_delta: float) -> dict[str, float]:
    raw_delta = np.asarray(model.predict(x), dtype=float)
    delta = np.clip(raw_delta, -max_delta, max_delta)
    core_pb = x[:, FEATURE_NAMES.index("core_p_b")].astype(float)
    volatility = x[:, FEATURE_NAMES.index("sx_volatility_8")].astype(float)
    micro = x[:, FEATURE_NAMES.index("sx_micro_run_4")].astype(float)
    ties = x[:, FEATURE_NAMES.index("tie_density")].astype(float)
    streak = x[:, FEATURE_NAMES.index("core_streak")].astype(float)
    residual_pb = np.clip(core_pb + delta, 0.0, 1.0)
    final_pb, dampers = _apply_final_formula(core_pb, delta, volatility, micro, ties)
    return {
        "samples": float(len(x)),
        "core_accuracy": direction_accuracy(core_pb, actual_b),
        "residual_accuracy": direction_accuracy(residual_pb, actual_b),
        "corrected_accuracy": direction_accuracy(final_pb, actual_b),
        "core_brier": brier(core_pb, actual_b),
        "residual_brier": brier(residual_pb, actual_b),
        "corrected_brier": brier(final_pb, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "mean_volatility_8": float(np.mean(volatility)) if len(volatility) else 0.0,
        "mean_tie_density": float(np.mean(ties)) if len(ties) else 0.0,
        "mean_abs_core_streak": float(np.mean(np.abs(streak))) if len(streak) else 0.0,
        "mean_damper": float(np.mean(dampers)) if len(dampers) else 1.0,
    }


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node: Mapping[str, Any] = tree
    for _ in range(256):
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


def export_portable_bundle(
    model: XGBRegressor,
    *,
    reference_x: np.ndarray,
    output_path: Path,
    max_delta: float,
    metrics: Mapping[str, Any],
    training_rows: int,
) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(text) for text in booster.get_dump(dump_format="json")]
    reference = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf(tree, reference) for tree in trees)
    model_reference = float(model.predict(reference.reshape(1, -1))[0])
    base_score = model_reference - tree_sum

    for vector in np.asarray(reference_x[: min(64, len(reference_x))], dtype=float):
        portable = base_score + sum(_tree_leaf(tree, vector) for tree in trees)
        native = float(model.predict(vector.reshape(1, -1))[0])
        if abs(portable - native) > 1e-5:
            raise RuntimeError(f"portable export mismatch: {portable} vs {native}")

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "base_score": float(base_score),
        "max_delta": float(max_delta),
        "short_memory": {
            "volatility_feature": "sx_volatility_8",
            "volatility_window": DEFAULT_VOLATILITY_WINDOW,
            "micro_feature": "sx_micro_run_4",
            "micro_window": DEFAULT_MICRO_WINDOW,
            "tie_feature": "tie_density",
            "tie_window": DEFAULT_TIE_WINDOW,
            "core_streak_feature": "core_streak",
        },
        "damper": {
            "regular_volatility_max": DEFAULT_REGULAR_VOLATILITY_MAX,
            "high_volatility_start": DEFAULT_HIGH_VOLATILITY_START,
            "high_damper_max": DEFAULT_HIGH_DAMPER_MAX,
            "high_damper_min": DEFAULT_HIGH_DAMPER_MIN,
            "fast_unlock_run": DEFAULT_FAST_UNLOCK_RUN,
            "tie_damping_power": DEFAULT_TIE_DAMPING_POWER,
            "tie_damper_min": DEFAULT_TIE_DAMPER_MIN,
        },
        "bet_weight": {
            "minimum": DEFAULT_BET_WEIGHT_MIN,
            "reference_edge": DEFAULT_BET_REFERENCE_EDGE,
            "attack_streak_trigger": DEFAULT_ATTACK_STREAK_TRIGGER,
            "attack_single_win_multiplier": DEFAULT_ATTACK_SINGLE_WIN_MULTIPLIER,
            "defense_tie_density": DEFAULT_DEFENSE_TIE_DENSITY,
            "defense_volatility": DEFAULT_DEFENSE_VOLATILITY,
            "defense_damper": DEFAULT_DEFENSE_DAMPER,
        },
        "trees": trees,
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "final_probability": "0.50 + ((core_p_B - 0.50) + delta) * damper",
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    x, residual, actual_b, shoes = make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need at least {args.min_samples} valid rows; got {len(x)}")

    validation = deterministic_validation_mask(shoes, fraction=args.validation_fraction)
    train = ~validation
    model = build_regressor(random_state=args.random_state)
    model.fit(x[train], residual[train])
    metrics = evaluate(model, x[validation], actual_b[validation], max_delta=args.max_delta)
    accepted = (
        metrics["corrected_brier"] <= metrics["core_brier"] + args.max_brier_regression
        and metrics["corrected_accuracy"] >= metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected tie/streak residual model; use --force only for diagnostics")

    final_model = build_regressor(random_state=args.random_state)
    final_model.fit(x, residual)
    export_portable_bundle(
        final_model,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=metrics,
        training_rows=len(x),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB XGBoost residual bias + 8/4 tie/streak dynamic damper trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train XGBRegressor and export browser model JSON")
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
