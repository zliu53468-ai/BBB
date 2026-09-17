#!/usr/bin/env python3
"""Deterministic continuation/reversal features from baccarat Big Road + derived roads.

Raw red/blue lower-road markers are used only internally to build structural
continuation/reversal states. XGBoost receives probabilities and state flags,
not raw color identity.

For each lower road:
- continue_now: newest marker continued the previous marker color.
- turn_now: newest marker flipped from the previous marker color.
- p_continue: Bayesian-smoothed probability that the current structural run continues.
- p_turn: complement of p_continue.

The Big Road receives the same continuation/turn probability treatment using
B/P streaks. Cross-road means summarize the three lower roads.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "big_eye_continue_now",
    "big_eye_turn_now",
    "big_eye_p_continue",
    "big_eye_p_turn",
    "small_road_continue_now",
    "small_road_turn_now",
    "small_road_p_continue",
    "small_road_p_turn",
    "cockroach_continue_now",
    "cockroach_turn_now",
    "cockroach_p_continue",
    "cockroach_p_turn",
    "big_road_p_continue",
    "big_road_p_turn",
    "derived_p_continue",
    "derived_p_turn",
)

PRIOR = 1.0
LOCAL_WINDOW = 12


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    values: Iterable[Any] = history.upper() if isinstance(history, str) else history
    out: list[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P"}:
            out.append(value)
    return out


def derived_markers(history: str | Sequence[Any] | None, offset: int) -> list[int]:
    """Return standard lower-road markers as +1/-1 for internal structure analysis."""
    offset = int(offset)
    if offset not in {1, 2, 3}:
        raise ValueError("derived-road offset must be 1, 2, or 3")
    seq = normalize_bp(history)
    if not seq:
        return []

    runs: list[int] = []
    markers: list[int] = []
    previous = ""
    for side in seq:
        if side == previous and runs:
            runs[-1] += 1
            current_col = len(runs) - 1
            row = runs[-1] - 1
            if current_col >= offset:
                reference_depth = runs[current_col - offset]
                markers.append(-1 if reference_depth == row else 1)
        else:
            runs.append(1)
            current_col = len(runs) - 1
            if current_col >= offset + 1:
                left_depth = runs[current_col - 1]
                compare_depth = runs[current_col - 1 - offset]
                markers.append(1 if left_depth == compare_depth else -1)
            previous = side
    return markers


def _run_lengths(values: Sequence[Any]) -> list[int]:
    if not values:
        return []
    runs: list[int] = []
    last = None
    for value in values:
        if runs and value == last:
            runs[-1] += 1
        else:
            runs.append(1)
            last = value
    return runs


def recent_continue_rate(values: Sequence[Any], window: int = LOCAL_WINDOW, prior: float = PRIOR) -> float:
    """Smoothed local frequency of same-state transitions."""
    if len(values) < 2:
        return 0.5
    recent = list(values[-max(2, int(window) + 1):])
    cont = sum(1 for a, b in zip(recent, recent[1:]) if a == b)
    turn = max(0, len(recent) - 1 - cont)
    return float((cont + prior) / (cont + turn + 2.0 * prior))


def survival_continue_prob(values: Sequence[Any], prior: float = PRIOR) -> float:
    """Estimate P(current run continues one more step) from previous run lengths.

    At current depth d, previous completed runs that reached depth d are eligible.
    Runs longer than d are continuations; runs ending exactly at d are turns.
    Falls back to the local transition rate when history is sparse.
    """
    if len(values) < 2:
        return 0.5
    runs = _run_lengths(values)
    if not runs:
        return 0.5
    depth = runs[-1]
    completed = runs[:-1]
    eligible = [length for length in completed if length >= depth]
    if len(eligible) < 2:
        return recent_continue_rate(values)
    continued = sum(1 for length in eligible if length > depth)
    stopped = len(eligible) - continued
    return float((continued + prior) / (continued + stopped + 2.0 * prior))


def continuation_state(values: Sequence[Any]) -> dict[str, float]:
    if len(values) < 2:
        return {
            "continue_now": 0.0,
            "turn_now": 0.0,
            "p_continue": 0.5,
            "p_turn": 0.5,
            "available": 0.0,
        }
    continued_now = 1.0 if values[-1] == values[-2] else 0.0
    turned_now = 1.0 - continued_now
    p_continue = max(0.0, min(1.0, survival_continue_prob(values)))
    return {
        "continue_now": continued_now,
        "turn_now": turned_now,
        "p_continue": p_continue,
        "p_turn": 1.0 - p_continue,
        "available": 1.0,
    }


def road_continuation_state(history: str | Sequence[Any] | None, offset: int) -> dict[str, float]:
    return continuation_state(derived_markers(history, offset))


def big_road_continuation_state(history: str | Sequence[Any] | None) -> dict[str, float]:
    return continuation_state(normalize_bp(history))


def build_derived_road_features(history: str | Sequence[Any] | None) -> dict[str, float]:
    big_eye = road_continuation_state(history, 1)
    small = road_continuation_state(history, 2)
    cockroach = road_continuation_state(history, 3)
    big_road = big_road_continuation_state(history)

    states = (big_eye, small, cockroach)
    available_states = [state for state in states if state["available"] > 0]
    if available_states:
        derived_p_continue = sum(state["p_continue"] for state in available_states) / len(available_states)
    else:
        derived_p_continue = 0.5
    derived_p_turn = 1.0 - derived_p_continue

    return {
        "big_eye_continue_now": big_eye["continue_now"],
        "big_eye_turn_now": big_eye["turn_now"],
        "big_eye_p_continue": big_eye["p_continue"],
        "big_eye_p_turn": big_eye["p_turn"],
        "small_road_continue_now": small["continue_now"],
        "small_road_turn_now": small["turn_now"],
        "small_road_p_continue": small["p_continue"],
        "small_road_p_turn": small["p_turn"],
        "cockroach_continue_now": cockroach["continue_now"],
        "cockroach_turn_now": cockroach["turn_now"],
        "cockroach_p_continue": cockroach["p_continue"],
        "cockroach_p_turn": cockroach["p_turn"],
        "big_road_p_continue": big_road["p_continue"],
        "big_road_p_turn": big_road["p_turn"],
        "derived_p_continue": float(derived_p_continue),
        "derived_p_turn": float(derived_p_turn),
    }
