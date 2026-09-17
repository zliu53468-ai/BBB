#!/usr/bin/env python3
"""Deterministic baccarat derived-road features for BBB XGBoost residual input.

The three lower roads are derived from the B/P Big Road structure; they are not
independent outcome labels. Marker encoding used here:

    red  = +1  (structure repeated / matched)
    blue = -1  (structure broke / differed)

Big Eye Boy uses offset 1, Small Road offset 2, Cockroach Pig offset 3.
Ties do not create a new Big Road cell and are ignored for the derived roads.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "big_eye_color",
    "big_eye_run",
    "big_eye_switch_rate_6",
    "small_road_color",
    "small_road_run",
    "small_road_switch_rate_6",
    "cockroach_color",
    "cockroach_run",
    "cockroach_switch_rate_6",
    "derived_road_agreement",
)


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    values: Iterable[Any]
    if isinstance(history, str):
        values = history.upper()
    else:
        values = history
    out: list[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P"}:
            out.append(value)
    return out


def derived_markers(history: str | Sequence[Any] | None, offset: int) -> list[int]:
    """Return red(+1)/blue(-1) markers for one standard derived road.

    The implementation uses unbounded Big Road streak depths. This is equivalent
    to treating dragon tails as if the streak continued downward, which avoids
    display-grid geometry affecting the road calculation.
    """
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
            row = runs[-1] - 1  # zero-based depth in the unbounded Big Road column
            if current_col >= offset:
                reference_depth = runs[current_col - offset]
                # Compare the reference cell at this row with the cell above it.
                # They differ only when the reference column ends exactly at row.
                markers.append(-1 if reference_depth == row else 1)
        else:
            runs.append(1)
            current_col = len(runs) - 1
            # A new column compares the two completed columns separated by the
            # road's offset: Big Eye=adjacent, Small=skip 1, Cockroach=skip 2.
            if current_col >= offset + 1:
                left_depth = runs[current_col - 1]
                compare_depth = runs[current_col - 1 - offset]
                markers.append(1 if left_depth == compare_depth else -1)
            previous = side

    return markers


def current_marker_run(markers: Sequence[int]) -> int:
    if not markers:
        return 0
    color = int(markers[-1])
    run = 1
    for value in reversed(markers[:-1]):
        if int(value) != color:
            break
        run += 1
    return run


def switch_rate(markers: Sequence[int], window: int = 6) -> float:
    if len(markers) < 2:
        return 0.0
    recent = list(markers[-max(2, int(window)):])
    changes = sum(1 for a, b in zip(recent, recent[1:]) if int(a) != int(b))
    return float(changes / max(1, len(recent) - 1))


def road_state(history: str | Sequence[Any] | None, offset: int) -> dict[str, float]:
    markers = derived_markers(history, offset)
    return {
        "color": float(markers[-1]) if markers else 0.0,
        "run": float(current_marker_run(markers)),
        "switch_rate_6": float(switch_rate(markers, 6)),
    }


def build_derived_road_features(history: str | Sequence[Any] | None) -> dict[str, float]:
    big_eye = road_state(history, 1)
    small = road_state(history, 2)
    cockroach = road_state(history, 3)
    agreement = (big_eye["color"] + small["color"] + cockroach["color"]) / 3.0
    return {
        "big_eye_color": big_eye["color"],
        "big_eye_run": big_eye["run"],
        "big_eye_switch_rate_6": big_eye["switch_rate_6"],
        "small_road_color": small["color"],
        "small_road_run": small["run"],
        "small_road_switch_rate_6": small["switch_rate_6"],
        "cockroach_color": cockroach["color"],
        "cockroach_run": cockroach["run"],
        "cockroach_switch_rate_6": cockroach["switch_rate_6"],
        "derived_road_agreement": float(agreement),
    }
