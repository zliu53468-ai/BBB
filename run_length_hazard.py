"""Quant256 V6: deterministic count-conditioned run survival/hazard features.

The Python twin of bayes_markov_runtime.js/countSurvivalHazard. It consumes
fully observed 0..9 card points if available. With B/P/T alone, risk fields
remain research diagnostics and no follow/break action is authorized.
The 'hazard_rate' is a bounded stress index, NOT a calibrated probability.
Any 15-25% fraction is simulated only; actionable bet_fraction is zero.
"""
from __future__ import annotations
from typing import Any, Dict, List
import math


def _clip(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def count_survival_hazard(
    history: List[str],
    true_count: float = 0.0,
    zero_ratio: float = 1.0,
    count_source: str = "estimated",
) -> Dict[str, Any]:
    if not isinstance(history, list) or any(x not in ("B", "P", "T") for x in history):
        raise ValueError("invalid_survival_history")
    if not math.isfinite(true_count) or not math.isfinite(zero_ratio) or zero_ratio < 0:
        raise ValueError("invalid_survival_shoe")
    tc, z = _clip(true_count, -8, 8), _clip(zero_ratio, 0, 2)
    observed = count_source == "observed"
    side, n = None, 0
    for outcome in reversed(history):
        if outcome == "T":
            continue
        if side is None:
            side, n = outcome, 1
        elif side == outcome:
            n += 1
        else:
            break
    active = n >= 4 and side is not None
    sign = 1 if side == "B" else -1 if side == "P" else 0
    aligned = sign * tc
    conflict = bool(active and aligned <= -3)
    supportive = bool(active and aligned >= 1.5 and z >= .75)
    blurry = abs(tc) <= 1.5
    follow, break_factor = 1.0, 1.0
    if active and observed and aligned > 3:
        follow = 1.5
    if active and observed and aligned < -2.5:
        follow, break_factor = .4, 2.0
    depth = _clip((n - 3) / 9)
    advantage = _clip((aligned - 1.5) / 4)
    cpsi = (_clip((.2 + .8 * advantage) *
                  (.6 + .4 * _clip(z))) if active else 0.0)
    hazard = (_clip(.55 + .35 * depth - .4 * cpsi +
                    (.30 if conflict else 0)) if active else 0.0)
    direction, reason = "Skip", "inactive"
    if not active:
        reason = "run_length_below_4"
    elif not observed:
        reason = "no_observed_card_points"
    elif conflict:
        direction = "P" if side == "B" else "B"
        reason = "observed_count_conflict_break_scenario"
    elif blurry:
        reason = "count_ambiguous_skip"
    elif supportive:
        direction, reason = side, "observed_count_aligned_follow_scenario"
    else:
        reason = "insufficient_count_or_zero_density"
    cont = direction != "Skip" and direction == side
    fraction = (0.0 if direction == "Skip" else
                _clip(.15 + .01 * (n - 4), .15, .25) if cont else .15)
    return {
        "version": "BBB_CPSI_HAZARD_V6", "active": bool(active),
        "run_length": n, "run_side": side,
        "observed_card_evidence": observed, "count_source": count_source,
        "tc": tc, "zero_ratio": z, "aligned_tc": aligned,
        "cpsi": cpsi, "hazard_rate": hazard,
        "follow_multiplier": follow, "break_multiplier": break_factor,
        "direction": direction, "direction_text":
            "觀望" if direction == "Skip" else "莊" if direction == "B" else "閒",
        "reason": reason, "uncalibrated": True, "simulated_only": True,
        "bet_fraction": 0, "simulated_bet_fraction": fraction,
        "road_override_allowed": False,
        "pf_weighting_allowed": bool(active and observed),
        "risk_is_probability": False,
    }


def survival_features_16(signal: Dict[str, Any]) -> List[float]:
    """Fixed index map for slots 16..31 and 128..143 in the JS 256D panel."""
    side = signal["run_side"]
    return [
        signal["cpsi"], signal["hazard_rate"],
        _clip(signal["run_length"] / 16),
        1.0 if side == "B" else -1.0 if side == "P" else 0.0,
        _clip(signal["aligned_tc"] / 8, -1, 1),
        _clip(signal["zero_ratio"] / 2),
        float(signal["observed_card_evidence"]),
        float(signal["active"]), float(signal["direction"] != "Skip"),
        float(signal["direction"] == side and side is not None),
        float(signal["direction"] not in ("Skip", side)),
        signal["follow_multiplier"] - 1,
        signal["break_multiplier"] - 1,
        _clip(signal["simulated_bet_fraction"] / .25),
        float(signal["pf_weighting_allowed"]),
        _clip(signal["run_length"] / 70),
    ]


if __name__ == "__main__":
    import json, sys
    data = json.load(sys.stdin)
    print(json.dumps({
        "survival": count_survival_hazard(**data),
        "features": survival_features_16(count_survival_hazard(**data)),
    }, ensure_ascii=False))
