"""Offline candidate trainer for exported point-score hand histories.

This tool is intentionally separate from the browser runtime. It reads the
licensed UI's exported ``bgs_observed_hands_*.json`` file, creates
chronological next-hand examples, holds out entire shoes, and writes a
review-only logistic-regression candidate. It never changes deployed weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

OUTCOMES = ("B", "P", "T")
WINDOW_OUTCOMES = (8, 32)
POINT_WINDOW = 32
FEATURE_NAMES = (
    "shoe_progress",
    *(f"outcome_{side}_last{window}" for window in WINDOW_OUTCOMES for side in OUTCOMES),
    "same_transition_last16", "switch_transition_last16", "point_coverage_last32",
    *(f"player_point_{value}_last32" for value in range(10)),
    *(f"banker_point_{value}_last32" for value in range(10)),
    "card_count_4_last32", "card_count_5_last32", "card_count_6_last32",
    "point_diff_signed_mean_last32", "point_diff_abs_mean_last32", "point_diff_std_last32",
    "last_player_point", "last_banker_point", "last_card_count",
)


def _optional_int(value: Any, low: int, high: int, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result != value and str(result) != str(value):
        raise ValueError(f"{field} must be an integer")
    if not low <= result <= high:
        raise ValueError(f"{field} must be in {low}..{high}")
    return result


def load_observations(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read the browser's macro-observation export and validate shoe order."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = raw.get("rows") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must contain a non-empty rows array")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    point_rows = 0
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("each row must be an object")
        shoe = str(item.get("shoe_id") or "").strip()
        outcome = str(item.get("outcome") or "").upper()
        if not shoe or outcome not in OUTCOMES:
            raise ValueError("each row requires shoe_id and B/P/T outcome")
        round_index = _optional_int(item.get("round_index"), 1, 500, "round_index")
        if round_index is None:
            raise ValueError("round_index is required")
        key = (shoe, round_index)
        if key in seen:
            raise ValueError(f"duplicate shoe/round: {shoe}/{round_index}")
        seen.add(key)
        player = _optional_int(item.get("player_score"), 0, 9, "player_score")
        banker = _optional_int(item.get("banker_score"), 0, 9, "banker_score")
        cards = _optional_int(item.get("card_count"), 4, 6, "card_count")
        if (player is None) != (banker is None):
            raise ValueError(f"partial score observation at {shoe}/{round_index}")
        if player is not None:
            point_rows += 1
            expected = "B" if banker > player else "P" if player > banker else "T"
            if outcome != expected:
                raise ValueError(f"score/outcome mismatch at {shoe}/{round_index}")
        grouped.setdefault(shoe, []).append({
            "shoe_id": shoe, "round_index": round_index, "outcome": outcome,
            "player_score": player, "banker_score": banker, "card_count": cards,
        })
    if point_rows == 0:
        raise ValueError("no complete player_score/banker_score observations found")
    for shoe, hands in grouped.items():
        hands.sort(key=lambda row: row["round_index"])
        rounds = [row["round_index"] for row in hands]
        if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
            raise ValueError(f"round_index must be continuous within shoe {shoe}")
    return grouped


def _features(history: list[dict[str, Any]], total_hands: int) -> list[float]:
    values: list[float] = [min(1.0, len(history) / max(1, total_hands))]
    for window in WINDOW_OUTCOMES:
        tail = history[-window:]
        denominator = max(1, len(tail))
        values.extend(sum(row["outcome"] == side for row in tail) / denominator for side in OUTCOMES)
    directional = [row["outcome"] for row in history if row["outcome"] in ("B", "P")][-17:]
    transitions = [directional[i] == directional[i - 1] for i in range(1, len(directional))]
    values.extend((
        sum(transitions) / max(1, len(transitions)),
        sum(not item for item in transitions) / max(1, len(transitions)),
    ))
    tail = history[-POINT_WINDOW:]
    score_rows = [row for row in tail if row