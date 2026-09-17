#!/usr/bin/env python3
"""Deterministic baccarat derived-road turning features for BBB XGBoost.

The three lower roads are still generated from standard Big Road structure, but
XGBoost is NOT given raw red/blue direction as a feature.  Red/blue markers are
used only internally to detect structural turns.

For each derived road we export:
- turn_now: 1 when the newest marker changed color from the previous marker.
- steps_since_turn: length of the current same-color segment (1 immediately after a turn).
- turn_rate_6: fraction of color changes across the most recent 6 markers.

A tenth feature, derived_turn_sync, measures how many currently available lower
roads turned on the newest marker at the same time.

Big Eye Boy uses offset 1, Small Road offset 2, Cockroach Pig offset 3.
Ties do not create a new Big Road cell and are ignored for derived-road geometry.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "big_eye_turn_now",
    "big_eye_steps_since_turn",
    "big_eye_turn_rate_6",
    "small_road_turn_now",
    "small_road_steps_since_turn",
    "small_road_turn_rate_6",
    "cockroach_turn_now",
    "cockroach_steps_since_turn",
    "cockroach_turn_rate_6",
    "derived_turn_sync",
)


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
    """Return standard lower-road markers as +1/-1 for internal turn detection."""
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


def turn_now(markers: Sequence[int]) -> float:
    if len(markers) < 2:
        return 0.0
    return 1.0 if int(markers[-1]) != int(markers[-2]) else 0.0


def steps_since_turn(markers: Sequence[int]) -> int:
    if not markers:
        return 0
    current = int(markers[-1])
    steps = 1
    for value in reversed(markers[:-1]):
        if int(value) != current:
            break
        steps += 1
    return steps


def turn_rate(markers: Sequence[int], window: int = 6) -> float:
    if len(markers) < 2:
        return 0.0
    recent = list(markers[-max(2, int(window)):])
    turns = sum(1 for a, b in zip(recent, recent[1:]) if int(a) != int(b))
    return float(turns / max(1, len(recent) - 1))


def road_turn_state(history: str | Sequence[Any] | None, offset: int) -> dict[str, float]:
    markers = derived_markers(history, offset)
    return {
        "turn_now": turn_now(markers),
        "steps_since_turn": float(steps_since_turn(markers)),
        "turn_rate_6": float(turn_rate(markers, 6)),
        "available": 1.0 if len(markers) >= 2 else 0.0,
    }


def build_derived_road_features(history: str | Sequence[Any] | None) -> dict[str, float]:
    big_eye = road_turn_state(history, 1)
    small = road_turn_state(history, 2)
    cockroach = road_turn_state(history, 3)

    states = (big_eye, small, cockroach)
    available = sum(state["available"] for state in states)
    turn_sync = (
        sum(state["turn_now"] for state in states) / available
        if available > 0
        else 0.0
    )

    return {
        "big_eye_turn_now": big_eye["turn_now"],
        "big_eye_steps_since_turn": big_eye["steps_since_turn"],
        "big_eye_turn_rate_6": big_eye["turn_rate_6"],
        "small_road_turn_now": small["turn_now"],
        "small_road_steps_since_turn": small["steps_since_turn"],
        "small_road_turn_rate_6": small["turn_rate_6"],
        "cockroach_turn_now": cockroach["turn_now"],
        "cockroach_steps_since_turn": cockroach["steps_since_turn"],
        "cockroach_turn_rate_6": cockroach["turn_rate_6"],
        "derived_turn_sync": float(turn_sync),
    }
