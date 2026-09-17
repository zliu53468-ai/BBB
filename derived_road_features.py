#!/usr/bin/env python3
"""Continuation/reversal features from Baccarat Big Road + three derived roads.

Raw lower-road red/blue markers are used only internally as structural signals.
XGBoost is not given raw color identity or color distribution.

For each derived road the exported features are:
- continue_now / turn_now: whether the newest derived marker continued or flipped.
- p_bigroad_continue / p_bigroad_turn: causal, Bayesian-smoothed historical
  probability that the *next Big Road B/P result* continues or turns when this
  derived road is in a comparable current structural state.

Big Road also gets its own run-survival P(continue)/P(turn). Cross-road features
average the three derived-road conditional probabilities.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "big_eye_continue_now",
    "big_eye_turn_now",
    "big_eye_p_bigroad_continue",
    "big_eye_p_bigroad_turn",
    "small_road_continue_now",
    "small_road_turn_now",
    "small_road_p_bigroad_continue",
    "small_road_p_bigroad_turn",
    "cockroach_continue_now",
    "cockroach_turn_now",
    "cockroach_p_bigroad_continue",
    "cockroach_p_bigroad_turn",
    "big_road_p_continue",
    "big_road_p_turn",
    "derived_p_bigroad_continue",
    "derived_p_bigroad_turn",
)

PRIOR = 1.0
LOCAL_WINDOW = 12
CONDITIONAL_WINDOW = 24
MIN_EXACT_MATCHES = 3


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
    if len(values) < 2:
        return 0.5
    recent = list(values[-max(2, int(window) + 1):])
    cont = sum(1 for a, b in zip(recent, recent[1:]) if a == b)
    turn = max(0, len(recent) - 1 - cont)
    return float((cont + prior) / (cont + turn + 2.0 * prior))


def survival_continue_prob(values: Sequence[Any], prior: float = PRIOR) -> float:
    """Estimate P(current run continues one more result) from prior run lengths."""
    if len(values) < 2:
        return 0.5
    runs = _run_lengths(values)
    depth = runs[-1]
    completed = runs[:-1]
    eligible = [length for length in completed if length >= depth]
    if len(eligible) < 2:
        return recent_continue_rate(values)
    continued = sum(1 for length in eligible if length > depth)
    stopped = len(eligible) - continued
    return float((continued + prior) / (continued + stopped + 2.0 * prior))


def marker_continue_turn(markers: Sequence[int]) -> tuple[float, float]:
    if len(markers) < 2:
        return 0.0, 0.0
    continued = 1.0 if int(markers[-1]) == int(markers[-2]) else 0.0
    return continued, 1.0 - continued


def _smoothed_binary_prob(outcomes: Sequence[int], prior: float = PRIOR) -> float:
    if not outcomes:
        return 0.5
    positives = sum(1 for value in outcomes if int(value) == 1)
    negatives = len(outcomes) - positives
    return float((positives + prior) / (positives + negatives + 2.0 * prior))


def derived_to_bigroad_continue_prob(
    history: str | Sequence[Any] | None,
    offset: int,
    *,
    window: int = CONDITIONAL_WINDOW,
) -> float:
    """Estimate P(next Big Road continues | current derived-road structural state).

    The current raw marker color is never exported. It is used internally to find
    comparable historical prefixes. We first match both current marker identity
    and whether that derived road just continued/turned. If too few exact matches
    exist, we back off to marker identity only, then to the Big Road base rate.
    """
    seq = normalize_bp(history)
    if len(seq) < 2:
        return 0.5

    current_markers = derived_markers(seq, offset)
    if not current_markers:
        return survival_continue_prob(seq)

    current_signal = int(current_markers[-1])
    current_continue, _ = marker_continue_turn(current_markers)
    current_transition = int(current_continue) if len(current_markers) >= 2 else None

    exact: list[int] = []
    signal_only: list[int] = []

    for t in range(1, len(seq)):
        prefix = seq[:t]
        markers = derived_markers(prefix, offset)
        if not markers or int(markers[-1]) != current_signal:
            continue

        bigroad_continued = 1 if seq[t] == seq[t - 1] else 0
        signal_only.append(bigroad_continued)

        if current_transition is not None and len(markers) >= 2:
            continued, _ = marker_continue_turn(markers)
            if int(continued) == current_transition:
                exact.append(bigroad_continued)

    exact = exact[-max(1, int(window)):]
    signal_only = signal_only[-max(1, int(window)):]

    if len(exact) >= MIN_EXACT_MATCHES:
        return _smoothed_binary_prob(exact)
    if len(signal_only) >= 2:
        return _smoothed_binary_prob(signal_only)
    return survival_continue_prob(seq)


def road_probability_state(history: str | Sequence[Any] | None, offset: int) -> dict[str, float]:
    markers = derived_markers(history, offset)
    continue_now, turn_now = marker_continue_turn(markers)
    p_bigroad_continue = max(0.0, min(1.0, derived_to_bigroad_continue_prob(history, offset)))
    return {
        "continue_now": continue_now,
        "turn_now": turn_now,
        "p_bigroad_continue": p_bigroad_continue,
        "p_bigroad_turn": 1.0 - p_bigroad_continue,
        "available": 1.0 if markers else 0.0,
    }


def big_road_continuation_state(history: str | Sequence[Any] | None) -> dict[str, float]:
    seq = normalize_bp(history)
    p_continue = max(0.0, min(1.0, survival_continue_prob(seq)))
    continue_now = 0.0
    turn_now = 0.0
    if len(seq) >= 2:
        continue_now = 1.0 if seq[-1] == seq[-2] else 0.0
        turn_now = 1.0 - continue_now
    return {
        "continue_now": continue_now,
        "turn_now": turn_now,
        "p_continue": p_continue,
        "p_turn": 1.0 - p_continue,
        "available": 1.0 if len(seq) >= 2 else 0.0,
    }


def build_derived_road_features(history: str | Sequence[Any] | None) -> dict[str, float]:
    big_eye = road_probability_state(history, 1)
    small = road_probability_state(history, 2)
    cockroach = road_probability_state(history, 3)
    big_road = big_road_continuation_state(history)

    states = (big_eye, small, cockroach)
    available_states = [state for state in states if state["available"] > 0]
    if available_states:
        derived_p_continue = sum(state["p_bigroad_continue"] for state in available_states) / len(available_states)
    else:
        derived_p_continue = big_road["p_continue"]
    derived_p_turn = 1.0 - derived_p_continue

    return {
        "big_eye_continue_now": big_eye["continue_now"],
        "big_eye_turn_now": big_eye["turn_now"],
        "big_eye_p_bigroad_continue": big_eye["p_bigroad_continue"],
        "big_eye_p_bigroad_turn": big_eye["p_bigroad_turn"],
        "small_road_continue_now": small["continue_now"],
        "small_road_turn_now": small["turn_now"],
        "small_road_p_bigroad_continue": small["p_bigroad_continue"],
        "small_road_p_bigroad_turn": small["p_bigroad_turn"],
        "cockroach_continue_now": cockroach["continue_now"],
        "cockroach_turn_now": cockroach["turn_now"],
        "cockroach_p_bigroad_continue": cockroach["p_bigroad_continue"],
        "cockroach_p_bigroad_turn": cockroach["p_bigroad_turn"],
        "big_road_p_continue": big_road["p_continue"],
        "big_road_p_turn": big_road["p_turn"],
        "derived_p_bigroad_continue": float(derived_p_continue),
        "derived_p_bigroad_turn": float(derived_p_turn),
    }
