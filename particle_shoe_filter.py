#!/usr/bin/env python3
"""Deterministic B/P/T-only particle shoe estimator for baccarat.

This module never claims to reconstruct unseen cards exactly.  It maintains a
posterior over plausible eight-deck rank compositions, conditions each
simulated hand on the observed B/P/T result, and derives a rule-consistent
48D next-hand physics estimate including 4/5/6-card consumption.

It is intentionally a post-MLP physics layer: the production 213D -> 48D MLP,
57D bridge, Frozen Core and XGBoost classifier remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

DECKS = 8
RANKS = 13
INITIAL_PER_RANK = DECKS * 4
TOTAL_CARDS = 52 * DECKS
PLAYABLE_CARDS_ESTIMATE = TOTAL_CARDS - 60
PHYSICS_DIM = 48
LEGACY_PARTICLE_COUNT = 64
PARTICLE_MIN = 512
PARTICLE_MID = 1024
PARTICLE_MAX = 2000
PARTICLE_TEACHER = 2000
# Kept as a public compatibility alias.  New callers should select a budget
# through ``select_particle_budget`` instead of treating this as a fixed count.
PARTICLE_COUNT = PARTICLE_MID
PARTICLE_FILTER_VERSION = 3
PARTICLE_POLICY = "adaptive_512_1024_2000"
ESS_RESAMPLE_THRESHOLD = .50
REJUVENATION_RATE = .03
REJUVENATION_UNIQUE_RATIO = .55
LIKELIHOOD_DRAWS = 4
FALLBACK_DRAWS = 12
FORECAST_DRAWS = 2
RB_INITIAL_STATE_DRAWS = 16
RANDOM_STATE = 20261005
PHYSICS_HAND_STATE_VERSION = 2
PHYSICS_HAND_STATE_POLICY = "current_hand_posterior_plus_rb_next_hand"
THIRD_CARD_NONE = 10
EARLY35_VERSION = 1
EARLY35_BRIDGE_START = 35.0
EARLY35_BRIDGE_END = 40.0
EARLY35_EVIDENCE_ANCHORS = (
    (1.0, .15), (5.0, .22), (10.0, .32), (15.0, .45),
    (20.0, .55), (25.0, .64), (30.0, .72), (35.0, .78),
)
EARLY35_EV_RELIABILITY_ANCHORS = (
    (1.0, .20), (5.0, .28), (10.0, .38), (15.0, .48),
    (20.0, .58), (25.0, .66), (30.0, .73), (35.0, .79),
)
SHOE_ERROR_CORRECTION_VERSION = 1
SHOE_ERROR_CORRECTION_POLICY = "posterior_health_only_no_observation_reweight"
SHOE_ERROR_CORRECTION_MEMORY = 10

_OUTCOMES = ("B", "P", "T")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalise(block: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(block, dtype=np.float64), 0.0, None)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full(len(x), 1.0 / max(1, len(x)), dtype=np.float64)
    return x / total


def _normalise_log_weights(log_weights: Sequence[float]) -> np.ndarray:
    """Stable log-sum-exp normalisation for sequential likelihood updates."""
    values=np.asarray(log_weights,dtype=np.float64).reshape(-1)
    finite=np.isfinite(values)
    if values.size==0:
        return values
    if not np.any(finite):
        return np.full(values.size,1.0/values.size,dtype=np.float64)
    maximum=float(np.max(values[finite]))
    shifted=np.zeros(values.size,dtype=np.float64)
    shifted[finite]=np.exp(np.clip(values[finite]-maximum,-745.0,0.0))
    total=float(shifted.sum())
    return shifted/total if total>1e-300 else np.full(values.size,1.0/values.size,dtype=np.float64)


def _log_weights_from_normalised(weights: Sequence[float]) -> np.ndarray:
    normalised=_normalise_log_weights(np.log(np.clip(np.asarray(weights,dtype=np.float64),1e-300,None)))
    return np.log(np.clip(normalised,1e-300,None))


def _composition_instability(diagnostics: dict[str, Any] | None) -> float:
    return _clip(float((diagnostics or {}).get("composition_spread", .12))/.20)


def particle_quality(diagnostics: dict[str, Any] | None) -> float:
    """Posterior quality, deliberately independent from the raw particle count."""
    diagnostics=diagnostics or {}
    ess=_clip(float(diagnostics.get("recent_ess_ratio", .5)))
    uncertainty=_clip(float(diagnostics.get("posterior_uncertainty", .5)))
    instability=_composition_instability(diagnostics)
    return _clip(.45*ess+.35*(1.0-uncertainty)+.20*(1.0-instability))


def _desired_particle_budget(diagnostics: dict[str, Any] | None) -> int:
    diagnostics=diagnostics or {}
    uncertainty=_clip(float(diagnostics.get("posterior_uncertainty", .5)))
    ess=_clip(float(diagnostics.get("recent_ess_ratio", .5)))
    instability=_composition_instability(diagnostics)
    if uncertainty<=.30 and ess>=.70 and instability<=.60:
        return PARTICLE_MIN
    if uncertainty<=.55 and ess>=.45 and instability<=.85:
        return PARTICLE_MID
    return PARTICLE_MAX


def select_particle_budget(
    diagnostics: dict[str, Any] | None,
    *,
    mode: str = "runtime",
    previous_budget: int | None = None,
) -> int:
    """Select a bounded runtime budget with one-tier hysteresis.

    The previous-hand diagnostics choose this hand's budget.  A direct
    low/high transition moves through 1024 first so one noisy hand cannot
    oscillate 512 -> 2000 -> 512.
    """
    if str(mode).lower()=="teacher":
        return PARTICLE_TEACHER
    desired=_desired_particle_budget(diagnostics)
    previous=int(previous_budget or 0)
    if previous not in (PARTICLE_MIN,PARTICLE_MID,PARTICLE_MAX) or desired==previous:
        return desired
    if {previous,desired}=={PARTICLE_MIN,PARTICLE_MAX}:
        return PARTICLE_MID
    return desired


def should_resample(ess_ratio: float) -> bool:
    return _clip(ess_ratio) < ESS_RESAMPLE_THRESHOLD


def _interp(value: float, anchors: Sequence[tuple[float, float]]) -> float:
    xs=np.asarray([item[0] for item in anchors],dtype=np.float64)
    ys=np.asarray([item[1] for item in anchors],dtype=np.float64)
    return float(np.interp(float(value),xs,ys))


def _blend(start: float, end: float, value: float) -> float:
    return float(start + (end - start) * _clip(value))


def _tokens(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    values = list(history.upper()) if isinstance(history, str) else list(history)
    return [str(value).strip().upper() for value in values if str(value).strip().upper() in _OUTCOMES]


def _rank_value(rank_index: int) -> int:
    rank = int(rank_index) + 1
    return rank if rank <= 9 else 0


def _banker_draws(total: int, player_third: int | None) -> bool:
    if player_third is None:
        return total <= 5
    if total <= 2:
        return True
    if total == 3:
        return player_third != 8
    if total == 4:
        return 2 <= player_third <= 7
    if total == 5:
        return 4 <= player_third <= 7
    if total == 6:
        return 6 <= player_third <= 7
    return False


@dataclass(frozen=True)
class SimulatedHand:
    outcome: str
    player_point: int
    banker_point: int
    ranks: tuple[int, ...]
    player_initial_total: int = 0
    banker_initial_total: int = 0
    natural_8_9: bool = False
    player_third_value: int | None = None
    banker_third_value: int | None = None

    @property
    def card_count(self) -> int:
        return len(self.ranks)


@dataclass(frozen=True)
class ParticlePhysicsEstimate:
    physics_48d: np.ndarray
    diagnostics: dict[str, Any]


def _normalised_entropy(values: Sequence[float]) -> float:
    probabilities = _normalise(np.asarray(values, dtype=np.float64))
    if probabilities.size <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))) / np.log(probabilities.size))


def _empty_hand_state_posterior() -> dict[str, Any]:
    """Stable diagnostics shape before a B/P/T observation is available."""
    return {
        "available": False,
        "total_weight": 0.0,
        "natural_probability": 0.0,
        "player_draw_probability": 0.0,
        "banker_draw_probability": 0.0,
        "cards_distribution": [0.0, 0.0, 0.0],
        "player_third_value_distribution": [0.0] * (THIRD_CARD_NONE + 1),
        "banker_third_value_distribution": [0.0] * (THIRD_CARD_NONE + 1),
        "player_final_point_distribution": [0.0] * 10,
        "banker_final_point_distribution": [0.0] * 10,
        "winner_distribution": [0.0] * 3,
        "winning_point_distribution": [0.0] * 10,
        "expected_cards_consumed": 0.0,
        "expected_point_diff_norm": 0.0,
        "expected_abs_point_diff_norm": 0.0,
        "draw_state_uncertainty": 0.0,
    }


def _new_hand_state_accumulator() -> dict[str, Any]:
    return {
        "weight": 0.0,
        "natural": 0.0,
        "player_draw": 0.0,
        "banker_draw": 0.0,
        "cards": np.zeros(3, dtype=np.float64),
        "player_third": np.zeros(THIRD_CARD_NONE + 1, dtype=np.float64),
        "banker_third": np.zeros(THIRD_CARD_NONE + 1, dtype=np.float64),
        "player_points": np.zeros(10, dtype=np.float64),
        "banker_points": np.zeros(10, dtype=np.float64),
        "winner": np.zeros(3, dtype=np.float64),
        "winning_points": np.zeros(10, dtype=np.float64),
        "cards_consumed": 0.0,
        "point_diff_norm": 0.0,
        "abs_point_diff_norm": 0.0,
    }


def _accumulate_hand_state(
    accumulator: dict[str, Any],
    *,
    weight: float,
    natural: bool,
    player_third_value: int | None,
    banker_third_value: int | None,
    player_point: int,
    banker_point: int,
    card_count: int,
) -> None:
    """Accumulate a rule-consistent hand state without changing the 48D schema."""
    contribution = max(0.0, float(weight))
    if contribution <= 0.0 or card_count not in {4, 5, 6}:
        return
    player_point = int(player_point) % 10
    banker_point = int(banker_point) % 10
    player_index = THIRD_CARD_NONE if player_third_value is None else int(player_third_value) % 10
    banker_index = THIRD_CARD_NONE if banker_third_value is None else int(banker_third_value) % 10
    outcome_index = 0 if banker_point > player_point else 1 if player_point > banker_point else 2
    accumulator["weight"] += contribution
    accumulator["natural"] += contribution * float(bool(natural))
    accumulator["player_draw"] += contribution * float(player_third_value is not None)
    accumulator["banker_draw"] += contribution * float(banker_third_value is not None)
    accumulator["cards"][card_count - 4] += contribution
    accumulator["player_third"][player_index] += contribution
    accumulator["banker_third"][banker_index] += contribution
    accumulator["player_points"][player_point] += contribution
    accumulator["banker_points"][banker_point] += contribution
    accumulator["winner"][outcome_index] += contribution
    accumulator["winning_points"][max(player_point, banker_point)] += contribution
    accumulator["cards_consumed"] += contribution * card_count
    point_diff = (banker_point - player_point) / 9.0
    accumulator["point_diff_norm"] += contribution * point_diff
    accumulator["abs_point_diff_norm"] += contribution * abs(point_diff)


def _accumulate_simulated_hand(accumulator: dict[str, Any], hand: SimulatedHand, weight: float) -> None:
    _accumulate_hand_state(
        accumulator,
        weight=weight,
        natural=hand.natural_8_9,
        player_third_value=hand.player_third_value,
        banker_third_value=hand.banker_third_value,
        player_point=hand.player_point,
        banker_point=hand.banker_point,
        card_count=hand.card_count,
    )


def _finalise_hand_state_posterior(accumulator: dict[str, Any]) -> dict[str, Any]:
    total = float(accumulator["weight"])
    if total <= 1e-12:
        return _empty_hand_state_posterior()
    cards = _normalise(accumulator["cards"])
    player_third = _normalise(accumulator["player_third"])
    banker_third = _normalise(accumulator["banker_third"])
    player_points = _normalise(accumulator["player_points"])
    banker_points = _normalise(accumulator["banker_points"])
    winner = _normalise(accumulator["winner"])
    winning_points = _normalise(accumulator["winning_points"])
    draw_uncertainty = _clip(
        .25 * _normalised_entropy(cards)
        + .15 * _normalised_entropy((accumulator["player_draw"] / total, 1.0 - accumulator["player_draw"] / total))
        + .20 * _normalised_entropy((accumulator["banker_draw"] / total, 1.0 - accumulator["banker_draw"] / total))
        + .20 * .5 * (_normalised_entropy(player_third) + _normalised_entropy(banker_third))
        + .20 * .5 * (_normalised_entropy(player_points) + _normalised_entropy(banker_points))
    )
    return {
        "available": True,
        "total_weight": total,
        "natural_probability": _clip(accumulator["natural"] / total),
        "player_draw_probability": _clip(accumulator["player_draw"] / total),
        "banker_draw_probability": _clip(accumulator["banker_draw"] / total),
        "cards_distribution": cards.tolist(),
        "player_third_value_distribution": player_third.tolist(),
        "banker_third_value_distribution": banker_third.tolist(),
        "player_final_point_distribution": player_points.tolist(),
        "banker_final_point_distribution": banker_points.tolist(),
        "winner_distribution": winner.tolist(),
        "winning_point_distribution": winning_points.tolist(),
        "expected_cards_consumed": float(accumulator["cards_consumed"] / total),
        "expected_point_diff_norm": float(accumulator["point_diff_norm"] / total),
        "expected_abs_point_diff_norm": float(accumulator["abs_point_diff_norm"] / total),
        "draw_state_uncertainty": draw_uncertainty,
    }


def _distribution_shift(prior: Sequence[float], posterior: Sequence[float]) -> float:
    """Bounded total-variation distance for latent-state attribution."""
    before = _normalise(np.asarray(prior, dtype=np.float64))
    after = _normalise(np.asarray(posterior, dtype=np.float64))
    if before.shape != after.shape:
        return 1.0
    return _clip(.5 * float(np.sum(np.abs(before - after))))


def _outcome_probability(snapshot: dict[str, Any], actual: str) -> float:
    index = {"B": 0, "P": 1, "T": 2}.get(actual, 2)
    values = snapshot.get("winner_distribution", ())
    if len(values) != 3:
        return 1.0 / 3.0
    return _clip(float(_normalise(np.asarray(values, dtype=np.float64))[index]), .01, .99)


def _physical_ev_from_winner_distribution(snapshot: dict[str, Any]) -> tuple[float, float, float]:
    winner = _normalise(np.asarray(snapshot.get("winner_distribution", ()), dtype=np.float64))
    if winner.size != 3:
        return 0.0, 0.0, 0.0
    banker, player = float(winner[0]), float(winner[1])
    ev_banker = banker * .95 - player
    ev_player = player - banker
    return ev_banker, ev_player, ev_banker - ev_player


def _pre_hand_snapshot(
    hand_state: dict[str, Any],
    *,
    particle_uncertainty: float,
    recent_ess_ratio: float,
    physical_ev: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Small immutable-in-spirit forecast record; never a card reconstruction."""
    ev_banker, ev_player, ev_gap = physical_ev or _physical_ev_from_winner_distribution(hand_state)
    return {
        "available": bool(hand_state.get("available", False)),
        "winner_distribution": list(hand_state.get("winner_distribution", (0.0, 0.0, 0.0))),
        "cards_distribution": list(hand_state.get("cards_distribution", (0.0, 0.0, 0.0))),
        "player_final_point_distribution": list(hand_state.get("player_final_point_distribution", (0.0,) * 10)),
        "banker_final_point_distribution": list(hand_state.get("banker_final_point_distribution", (0.0,) * 10)),
        "expected_consumed_cards": float(hand_state.get("expected_cards_consumed", 0.0)),
        "physical_ev_banker": float(ev_banker),
        "physical_ev_player": float(ev_player),
        "physical_ev_gap": float(ev_gap),
        "particle_uncertainty": _clip(particle_uncertainty),
        "recent_ess_ratio": _clip(recent_ess_ratio),
    }


def _empty_shoe_error_correction() -> dict[str, Any]:
    return {
        "shoe_error_correction_version": float(SHOE_ERROR_CORRECTION_VERSION),
        "shoe_error_correction_policy": SHOE_ERROR_CORRECTION_POLICY,
        "available": False,
        "shoe_posterior_health": 1.0,
        "draw_state_health": 1.0,
        "physical_ev_health": 1.0,
        "posterior_reliability_multiplier": 1.0,
        "draw_reliability_multiplier": 1.0,
        "physical_ev_reliability_multiplier": 1.0,
        "recent_miscalibration_score": 0.0,
        "particle_observation_reused": True,
    }


def _asymmetric_health_update(health: float, bad_evidence: float) -> float:
    """Conservative EMA: errors lower health faster than healthy observations restore it."""
    current = _clip(health)
    bad = _clip(bad_evidence)
    if bad > .04:
        # Even maximal evidence only moves a pristine health state to .85.
        return _clip(.80 * current + .20 * (1.0 - .75 * bad))
    return _clip(current + .06 * (1.0 - current))


def _apply_shoe_error_correction(
    state: dict[str, Any],
    memory: Sequence[dict[str, float]],
    *,
    prior_snapshot: dict[str, Any],
    posterior_hidden_given_actual: dict[str, Any],
    actual: str,
    recent_ess_ratio: float,
    particle_uncertainty: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Diagnose a B/P/T observation without touching its existing likelihood update."""
    prior = prior_snapshot
    posterior = posterior_hidden_given_actual
    actual_probability = _outcome_probability(prior, actual)
    prediction_surprise = float(-np.log(np.clip(actual_probability, .01, .99)))
    expected = _normalise(np.asarray(prior.get("winner_distribution", ()), dtype=np.float64))
    if expected.size != 3:
        expected = np.full(3, 1.0 / 3.0, dtype=np.float64)
    one_hot = np.zeros(3, dtype=np.float64)
    one_hot[{"B": 0, "P": 1, "T": 2}.get(actual, 2)] = 1.0
    brier_residual = float(np.mean((expected - one_hot) ** 2))
    draw_count_shift = _distribution_shift(prior.get("cards_distribution", ()), posterior.get("cards_distribution", ()))
    point_distribution_shift = .5 * (
        _distribution_shift(prior.get("player_final_point_distribution", ()), posterior.get("player_final_point_distribution", ()))
        + _distribution_shift(prior.get("banker_final_point_distribution", ()), posterior.get("banker_final_point_distribution", ()))
    )
    consumption_shift = _clip(abs(
        float(prior.get("expected_consumed_cards", 0.0))
        - float(posterior.get("expected_cards_consumed", 0.0))
    ) / 2.0)
    ess = _clip(recent_ess_ratio)
    uncertainty = _clip(particle_uncertainty)
    surprise_severity = _clip((prediction_surprise - .65) / 1.65)
    prior_memory = list(memory)[-SHOE_ERROR_CORRECTION_MEMORY:]
    recent_miscalibration_score = float(np.mean([item.get("surprise_severity", 0.0) for item in prior_memory])) if prior_memory else 0.0
    latent_shift = max(draw_count_shift, point_distribution_shift, consumption_shift)
    posterior_problem = _clip(.55 * surprise_severity + .25 * (1.0 - ess) + .20 * uncertainty)
    draw_problem = _clip(.50 * surprise_severity + .35 * latent_shift + .15 * uncertainty)
    ev_strength = _clip(abs(float(prior.get("physical_ev_gap", 0.0))) / .10)
    ev_overconfidence = _clip(ev_strength * (
        .60 * surprise_severity + .25 * recent_miscalibration_score + .15 * (1.0 - ess)
    ))
    updated = dict(state or _empty_shoe_error_correction())
    updated["shoe_posterior_health"] = _asymmetric_health_update(updated.get("shoe_posterior_health", 1.0), posterior_problem)
    updated["draw_state_health"] = _asymmetric_health_update(updated.get("draw_state_health", 1.0), draw_problem)
    updated["physical_ev_health"] = _asymmetric_health_update(updated.get("physical_ev_health", 1.0), ev_overconfidence)
    updated.update({
        "shoe_error_correction_version": float(SHOE_ERROR_CORRECTION_VERSION),
        "shoe_error_correction_policy": SHOE_ERROR_CORRECTION_POLICY,
        "available": bool(prior.get("available", False) and posterior.get("available", False)),
        "actual_outcome": actual,
        "actual_probability": actual_probability,
        "prediction_surprise": prediction_surprise,
        "brier_residual": brier_residual,
        "draw_count_shift": draw_count_shift,
        "point_distribution_shift": point_distribution_shift,
        "consumption_shift": consumption_shift,
        "recent_ess_ratio": ess,
        "particle_uncertainty": uncertainty,
        "recent_miscalibration_score": recent_miscalibration_score,
        "posterior_problem": posterior_problem,
        "draw_point_problem": draw_problem,
        "physical_ev_overconfidence": ev_overconfidence,
        "posterior_reliability_multiplier": _clip(.90 + .10 * updated["shoe_posterior_health"], .90, 1.0),
        "draw_reliability_multiplier": _clip(.90 + .10 * updated["draw_state_health"], .90, 1.0),
        "physical_ev_reliability_multiplier": _clip(.80 + .20 * updated["physical_ev_health"], .80, 1.0),
        # The Particle B/P/T likelihood above remains the sole observation update.
        "particle_observation_reused": True,
        "particle_posterior_reweighted": False,
    })
    next_memory = prior_memory + [{
        "surprise": prediction_surprise,
        "surprise_severity": surprise_severity,
        "brier_residual": brier_residual,
        "draw_count_shift": draw_count_shift,
        "point_distribution_shift": point_distribution_shift,
        "consumption_shift": consumption_shift,
        "recent_ess_ratio": ess,
        "particle_uncertainty": uncertainty,
    }]
    return updated, next_memory[-SHOE_ERROR_CORRECTION_MEMORY:]


def _draw_rank(counts: np.ndarray, rng: np.random.Generator) -> int:
    total = int(np.sum(counts))
    if total <= 0:
        raise IndexError("empty particle shoe")
    pick = int(rng.integers(total))
    cumulative = 0
    for index, count in enumerate(counts):
        cumulative += int(count)
        if pick < cumulative:
            counts[index] -= 1
            return index
    raise RuntimeError("rank draw overflow")


def _deal_from_counts(base_counts: np.ndarray, rng: np.random.Generator) -> tuple[SimulatedHand, np.ndarray]:
    counts = np.asarray(base_counts, dtype=np.int16).copy()
    if int(np.sum(counts)) < 6:
        raise IndexError("not enough particle cards")

    p1 = _draw_rank(counts, rng)
    b1 = _draw_rank(counts, rng)
    p2 = _draw_rank(counts, rng)
    b2 = _draw_rank(counts, rng)
    player = [p1, p2]
    banker = [b1, b2]

    player_total = (_rank_value(p1) + _rank_value(p2)) % 10
    banker_total = (_rank_value(b1) + _rank_value(b2)) % 10
    player_initial_total = player_total
    banker_initial_total = banker_total
    natural_8_9 = player_total in {8, 9} or banker_total in {8, 9}
    player_third_value: int | None = None
    banker_third_value: int | None = None

    if not natural_8_9:
        if player_total <= 5:
            p3 = _draw_rank(counts, rng)
            player.append(p3)
            player_third_value = _rank_value(p3)
            player_total = (player_total + player_third_value) % 10
        if _banker_draws(banker_total, player_third_value):
            b3 = _draw_rank(counts, rng)
            banker.append(b3)
            banker_third_value = _rank_value(b3)
            banker_total = (banker_total + banker_third_value) % 10

    outcome = "B" if banker_total > player_total else "P" if player_total > banker_total else "T"
    return SimulatedHand(
        outcome,
        player_total,
        banker_total,
        tuple(player + banker),
        player_initial_total,
        banker_initial_total,
        natural_8_9,
        player_third_value,
        banker_third_value,
    ), counts


def _point_value_counts(rank_counts: Sequence[int]) -> np.ndarray:
    """Aggregate the 13 rank counts into baccarat points while retaining deck depletion."""
    ranks = np.asarray(rank_counts, dtype=np.int16).reshape(-1)
    if ranks.size != RANKS:
        raise ValueError(f"expected {RANKS} rank counts, got {ranks.size}")
    points = np.zeros(10, dtype=np.int16)
    points[0] = int(np.sum(ranks[9:]))
    points[1:10] = ranks[:9]
    return points


def _draw_point_value(counts: np.ndarray, rng: np.random.Generator) -> int:
    total = int(np.sum(counts))
    if total <= 0:
        raise IndexError("empty point-value particle shoe")
    pick = int(rng.integers(total))
    cumulative = 0
    for value, count in enumerate(counts):
        cumulative += int(count)
        if pick < cumulative:
            counts[value] -= 1
            return value
    raise RuntimeError("point-value draw overflow")


def _rb_add_terminal_state(
    accumulator: dict[str, Any],
    *,
    weight: float,
    natural: bool,
    player_third_value: int | None,
    banker_third_value: int | None,
    player_point: int,
    banker_point: int,
) -> None:
    _accumulate_hand_state(
        accumulator,
        weight=weight,
        natural=natural,
        player_third_value=player_third_value,
        banker_third_value=banker_third_value,
        player_point=player_point,
        banker_point=banker_point,
        card_count=4 + int(player_third_value is not None) + int(banker_third_value is not None),
    )


def _rb_integrate_draw_branches(
    accumulator: dict[str, Any],
    remaining: np.ndarray,
    *,
    player_initial_total: int,
    banker_initial_total: int,
    weight: float,
) -> None:
    """Integrate all legal third-card branches exactly for one four-card state."""
    natural = player_initial_total in {8, 9} or banker_initial_total in {8, 9}
    if natural:
        _rb_add_terminal_state(
            accumulator,
            weight=weight,
            natural=True,
            player_third_value=None,
            banker_third_value=None,
            player_point=player_initial_total,
            banker_point=banker_initial_total,
        )
        return

    total_after_initial = int(np.sum(remaining))
    if player_initial_total > 5:
        if not _banker_draws(banker_initial_total, None):
            _rb_add_terminal_state(
                accumulator,
                weight=weight,
                natural=False,
                player_third_value=None,
                banker_third_value=None,
                player_point=player_initial_total,
                banker_point=banker_initial_total,
            )
            return
        for banker_third, count in enumerate(remaining):
            if count <= 0:
                continue
            _rb_add_terminal_state(
                accumulator,
                weight=weight * float(count) / total_after_initial,
                natural=False,
                player_third_value=None,
                banker_third_value=banker_third,
                player_point=player_initial_total,
                banker_point=(banker_initial_total + banker_third) % 10,
            )
        return

    for player_third, count in enumerate(remaining):
        if count <= 0:
            continue
        player_weight = weight * float(count) / total_after_initial
        after_player = remaining.copy()
        after_player[player_third] -= 1
        player_point = (player_initial_total + player_third) % 10
        if not _banker_draws(banker_initial_total, player_third):
            _rb_add_terminal_state(
                accumulator,
                weight=player_weight,
                natural=False,
                player_third_value=player_third,
                banker_third_value=None,
                player_point=player_point,
                banker_point=banker_initial_total,
            )
            continue
        total_after_player = int(np.sum(after_player))
        for banker_third, banker_count in enumerate(after_player):
            if banker_count <= 0:
                continue
            _rb_add_terminal_state(
                accumulator,
                weight=player_weight * float(banker_count) / total_after_player,
                natural=False,
                player_third_value=player_third,
                banker_third_value=banker_third,
                player_point=player_point,
                banker_point=(banker_initial_total + banker_third) % 10,
            )


def _rb_next_hand_forecast(
    particles: Sequence[np.ndarray],
    *,
    weights: Sequence[float] | None = None,
    random_state: int,
    initial_state_draws: int = RB_INITIAL_STATE_DRAWS,
) -> dict[str, Any]:
    """Finite-deck Rao--Blackwell forecast for draw states, points and B/P/T.

    The initial four cards are sampled from each particle's point-count shoe;
    every legal third-card branch is then integrated exactly without
    replacement.  Rank-level consumption remains in the original Monte Carlo
    forecast, where the 13-rank resolution is needed.
    """
    raw_weights=_normalise(np.asarray(weights,dtype=np.float64)) if weights is not None else np.full(len(particles),1.0/max(1,len(particles)),dtype=np.float64)
    valid_particles=[(counts,float(raw_weights[index])) for index,counts in enumerate(particles) if int(np.sum(counts))>=6]
    if not valid_particles:
        return _empty_hand_state_posterior()
    rng = np.random.default_rng(random_state + 1618033)
    accumulator = _new_hand_state_accumulator()
    state_draws = max(1, int(initial_state_draws))
    valid_weight=sum(weight for _,weight in valid_particles)
    for rank_counts, particle_weight in valid_particles:
        base_points = _point_value_counts(rank_counts)
        for _ in range(state_draws):
            points = base_points.copy()
            p1 = _draw_point_value(points, rng)
            b1 = _draw_point_value(points, rng)
            p2 = _draw_point_value(points, rng)
            b2 = _draw_point_value(points, rng)
            player_initial_total = (p1 + p2) % 10
            banker_initial_total = (b1 + b2) % 10
            _rb_integrate_draw_branches(
                accumulator,
                points,
                player_initial_total=player_initial_total,
                banker_initial_total=banker_initial_total,
                weight=(particle_weight/max(valid_weight,1e-12)) / state_draws,
            )
    forecast = _finalise_hand_state_posterior(accumulator)
    forecast["forecast_method"] = "rao_blackwellized_conditional"
    forecast["initial_state_draws"] = float(state_draws)
    return forecast


def _systematic_resample(
    counts: list[np.ndarray],
    consumed: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    target_count: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    source_count=len(counts)
    n=max(1,int(target_count or source_count))
    cdf = np.cumsum(weights)
    start = float(rng.random()) / n
    positions = start + np.arange(n, dtype=np.float64) / n
    indices = np.searchsorted(cdf, positions, side="left")
    indices = np.clip(indices, 0, source_count - 1)
    return [counts[int(i)].copy() for i in indices], consumed[indices].copy()


def _unique_particle_ratio(particles: Sequence[np.ndarray]) -> float:
    if not particles:
        return 0.0
    return float(len({tuple(np.asarray(counts,dtype=np.int16).tolist()) for counts in particles})/len(particles))


def _rejuvenate_particles(
    particles: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    rate: float = REJUVENATION_RATE,
) -> tuple[list[np.ndarray], int]:
    """Make a tiny legal rank-composition move after a collapsed resample."""
    output=[np.asarray(counts,dtype=np.int16).copy() for counts in particles]
    if not output:
        return output,0
    selected=max(1,int(round(len(output)*_clip(rate,0.02,0.05))))
    indices=rng.choice(len(output),size=min(selected,len(output)),replace=False)
    changed=0
    for index in np.asarray(indices,dtype=int):
        counts=output[int(index)]
        sources=np.flatnonzero(counts>0)
        targets=np.flatnonzero(counts<INITIAL_PER_RANK)
        if not len(sources) or not len(targets):
            continue
        source=int(sources[int(rng.integers(len(sources)))])
        target=int(targets[int(rng.integers(len(targets)))])
        if source==target:
            alternatives=targets[targets!=source]
            if not len(alternatives):
                continue
            target=int(alternatives[int(rng.integers(len(alternatives)))])
        if counts[source]<=0 or counts[target]>=INITIAL_PER_RANK:
            continue
        counts[source]-=1
        counts[target]+=1
        changed+=1
    return output,changed


def _filter_particles(
    history: Sequence[str],
    *,
    particle_count: int,
    random_state: int,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    rng = np.random.default_rng(random_state)
    particles = [np.full(RANKS, INITIAL_PER_RANK, dtype=np.int16) for _ in range(particle_count)]
    consumed = np.zeros(particle_count, dtype=np.float64)
    ess_history: list[float] = []
    log_weights=np.full(particle_count,-math.log(max(1,particle_count)),dtype=np.float64)

    for actual in history:
        next_particles: list[np.ndarray] = []
        next_consumed = np.zeros(particle_count, dtype=np.float64)
        log_likelihoods = np.zeros(particle_count, dtype=np.float64)

        for index, base in enumerate(particles):
            proposals: list[tuple[SimulatedHand, np.ndarray]] = []
            matches: list[tuple[SimulatedHand, np.ndarray]] = []
            for _ in range(LIKELIHOOD_DRAWS):
                hand, after = _deal_from_counts(base, rng)
                proposals.append((hand, after))
                if hand.outcome == actual:
                    matches.append((hand, after))

            match_count = len(matches)
            likelihood = (match_count + 0.15) / (LIKELIHOOD_DRAWS + 0.45)

            if not matches:
                for _ in range(FALLBACK_DRAWS):
                    hand, after = _deal_from_counts(base, rng)
                    if hand.outcome == actual:
                        matches.append((hand, after))
                        break

            if matches:
                chosen_hand, chosen_after = matches[int(rng.integers(len(matches)))]
                if match_count == 0:
                    likelihood = max(likelihood, 0.02)
            else:
                chosen_hand, chosen_after = proposals[int(rng.integers(len(proposals)))]
                likelihood = 1e-4

            next_particles.append(chosen_after)
            next_consumed[index] = consumed[index] + chosen_hand.card_count
            log_likelihoods[index] = math.log(max(likelihood,1e-12))

        weights=_normalise_log_weights(log_weights+log_likelihoods)
        ess = 1.0 / max(1e-12, float(np.sum(weights * weights)))
        ess_ratio=_clip(ess/particle_count)
        ess_history.append(ess_ratio)
        if should_resample(ess_ratio):
            particles,consumed=_systematic_resample(next_particles,next_consumed,weights,rng)
            if _unique_particle_ratio(particles)<REJUVENATION_UNIQUE_RATIO:
                particles,_=_rejuvenate_particles(particles,rng)
            log_weights=np.full(particle_count,-math.log(max(1,particle_count)),dtype=np.float64)
        else:
            particles,consumed=next_particles,next_consumed
            log_weights=_log_weights_from_normalised(weights)

    return particles, consumed, ess_history


def _forecast_particles(
    particles: Sequence[np.ndarray],
    consumed: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    random_state: int,
    forecast_draws: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(random_state + 104729)
    out = np.zeros(PHYSICS_DIM, dtype=np.float64)
    sample_mass = 0.0
    particle_weights=_normalise(np.asarray(weights,dtype=np.float64)) if weights is not None else np.full(len(particles),1.0/max(1,len(particles)),dtype=np.float64)

    for index,counts in enumerate(particles):
        for _ in range(forecast_draws):
            if int(np.sum(counts)) < 6:
                continue
            hand, _ = _deal_from_counts(counts, rng)
            mass=float(particle_weights[index])/max(1,int(forecast_draws))
            sample_mass += mass
            out[{4: 0, 5: 1, 6: 2}[hand.card_count]] += mass
            out[3 + hand.player_point] += mass
            out[13 + hand.banker_point] += mass
            out[23 + {"B": 0, "P": 1, "T": 2}[hand.outcome]] += mass
            for rank in hand.ranks:
                out[26 + rank] += mass
            diff = hand.banker_point - hand.player_point
            out[46] += mass*diff / 9.0
            out[47] += mass*abs(diff) / 9.0

    if sample_mass <= 1e-12:
        raise RuntimeError("particle forecast produced no samples")

    out[0:3] /= sample_mass
    out[3:13] /= sample_mass
    out[13:23] /= sample_mass
    out[23:26] /= sample_mass
    out[26:39] /= sample_mass
    rb_forecast = _rb_next_hand_forecast(particles, weights=particle_weights, random_state=random_state)
    if rb_forecast["available"]:
        out[0:3] = np.asarray(rb_forecast["cards_distribution"], dtype=np.float64)
        out[3:13] = np.asarray(rb_forecast["player_final_point_distribution"], dtype=np.float64)
        out[13:23] = np.asarray(rb_forecast["banker_final_point_distribution"], dtype=np.float64)
        out[23:26] = np.asarray(rb_forecast["winner_distribution"], dtype=np.float64)
        out[46] = float(rb_forecast["expected_point_diff_norm"])
        out[47] = float(rb_forecast["expected_abs_point_diff_norm"])
        rank_total = float(np.sum(out[26:39]))
        if rank_total > 1e-12:
            out[26:39] *= float(rb_forecast["expected_cards_consumed"]) / rank_total

    p_b,p_p=float(out[23]),float(out[24])
    physical_ev_b=p_b*.95-p_p
    physical_ev_p=p_p-p_b
    out[39]=physical_ev_b
    out[40]=physical_ev_p
    out[41]=physical_ev_b-physical_ev_p
    out[42]=0.0  # filled from posterior uncertainty after ESS/composition diagnostics
    out[43] = float(np.dot(particle_weights,consumed)) if len(consumed) else 0.0

    mean_counts = np.average(np.vstack(particles).astype(np.float64),axis=0,weights=particle_weights)
    remaining_total = max(1.0, float(np.sum(mean_counts)))
    out[44] = float(np.sum(mean_counts[:5]) / remaining_total)
    out[45] = float(np.sum(mean_counts[8:]) / remaining_total)
    if not rb_forecast["available"]:
        out[46] /= sample_mass
        out[47] /= sample_mass
    return out.astype(np.float32), rb_forecast


class ParticleShoeTracker:
    """Incremental particle state for chronological rows from the same shoe."""

    def __init__(
        self,
        *,
        particle_count: int | None = None,
        mode: str = "runtime",
        particle_filter_version: int = PARTICLE_FILTER_VERSION,
        random_state: int = RANDOM_STATE,
    ):
        self.mode="teacher" if str(mode).lower()=="teacher" else "runtime"
        self.particle_filter_version=int(particle_filter_version)
        self.particle_count_override=max(16,int(particle_count)) if particle_count is not None else None
        self.random_state = int(random_state)
        self.reset()

    @property
    def adaptive_enabled(self) -> bool:
        return self.particle_filter_version>=PARTICLE_FILTER_VERSION

    def _initial_budget(self) -> int:
        if self.particle_count_override is not None:
            return self.particle_count_override
        if self.mode=="teacher":
            return PARTICLE_TEACHER
        return PARTICLE_MID if self.adaptive_enabled else LEGACY_PARTICLE_COUNT

    def _normalised_weights(self) -> np.ndarray:
        return _normalise_log_weights(self.log_weights)

    def _resize_particle_budget(self, target: int) -> None:
        target=max(16,int(target))
        if target==len(self.particles):
            return
        self.last_rejuvenated=0
        weights=self._normalised_weights()
        self.particles,self.consumed=_systematic_resample(
            self.particles,self.consumed,weights,self.rng,target_count=target,
        )
        if target>len(weights) and _unique_particle_ratio(self.particles)<REJUVENATION_UNIQUE_RATIO:
            self.particles,self.last_rejuvenated=_rejuvenate_particles(self.particles,self.rng)
        self.log_weights=np.full(target,-math.log(target),dtype=np.float64)
        self.last_budget_transition=(len(weights),target)

    def _prepare_next_hand_budget(self) -> None:
        if self.particle_count_override is not None:
            return
        if self.mode=="teacher":
            target=PARTICLE_TEACHER
        elif self.adaptive_enabled:
            target=select_particle_budget(
                self.last_diagnostics,
                mode=self.mode,
                previous_budget=len(self.particles),
            )
        else:
            target=LEGACY_PARTICLE_COUNT
        self._resize_particle_budget(target)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        initial_budget=self._initial_budget()
        self.particles = [np.full(RANKS, INITIAL_PER_RANK, dtype=np.int16) for _ in range(initial_budget)]
        self.consumed = np.zeros(initial_budget, dtype=np.float64)
        self.log_weights=np.full(initial_budget,-math.log(initial_budget),dtype=np.float64)
        self.ess_history: list[float] = []
        self.history: list[str] = []
        self.current_hand_posterior = _empty_hand_state_posterior()
        self.pre_hand_snapshot: dict[str, Any] = _pre_hand_snapshot(
            _empty_hand_state_posterior(), particle_uncertainty=0.0, recent_ess_ratio=1.0,
        )
        self.pre_hand_snapshot_history_round = 0
        self.shoe_error_correction = _empty_shoe_error_correction()
        self.error_memory: list[dict[str, float]] = []
        self.last_diagnostics: dict[str, Any]={}
        self.last_resampled=False
        self.last_rejuvenated=0
        self.last_unique_particle_ratio=1.0
        self.last_budget_transition=(initial_budget,initial_budget)

    def _advance(self, actual: str) -> None:
        self._prepare_next_hand_budget()
        particle_count=len(self.particles)
        next_particles: list[np.ndarray] = []
        next_consumed = np.zeros(particle_count, dtype=np.float64)
        log_likelihoods = np.zeros(particle_count, dtype=np.float64)
        hand_state = _new_hand_state_accumulator()
        prior_hand_state = _new_hand_state_accumulator()
        prior_particle_weights=self._normalised_weights()

        for index, base in enumerate(self.particles):
            proposals: list[tuple[SimulatedHand, np.ndarray]] = []
            matches: list[tuple[SimulatedHand, np.ndarray]] = []
            for _ in range(LIKELIHOOD_DRAWS):
                hand, after = _deal_from_counts(base, self.rng)
                proposals.append((hand, after))
                # Reuse the same likelihood proposals as the prior hidden-state
                # snapshot; this adds no extra Particle simulation.
                _accumulate_simulated_hand(
                    prior_hand_state,
                    hand,
                    float(prior_particle_weights[index]) / LIKELIHOOD_DRAWS,
                )
                if hand.outcome == actual:
                    matches.append((hand, after))

            match_count = len(matches)
            likelihood = (match_count + 0.15) / (LIKELIHOOD_DRAWS + 0.45)
            if not matches:
                for _ in range(FALLBACK_DRAWS):
                    hand, after = _deal_from_counts(base, self.rng)
                    if hand.outcome == actual:
                        matches.append((hand, after))
                        break

            if matches:
                chosen_hand, chosen_after = matches[int(self.rng.integers(len(matches)))]
                if match_count == 0:
                    likelihood = max(likelihood, 0.02)
                # B/P/T remains the observation.  This only records the
                # matching latent hand states with particle x likelihood mass.
                posterior_weight = float(prior_particle_weights[index]) * likelihood / len(matches)
                for matching_hand, _ in matches:
                    _accumulate_simulated_hand(hand_state, matching_hand, posterior_weight)
            else:
                chosen_hand, chosen_after = proposals[int(self.rng.integers(len(proposals)))]
                likelihood = 1e-4

            next_particles.append(chosen_after)
            next_consumed[index] = self.consumed[index] + chosen_hand.card_count
            log_likelihoods[index] = math.log(max(likelihood,1e-12))

        weights=_normalise_log_weights(self.log_weights+log_likelihoods)
        ess = 1.0 / max(1e-12, float(np.sum(weights * weights)))
        ess_ratio = _clip(ess / particle_count)
        self.ess_history.append(ess_ratio)
        self.current_hand_posterior = _finalise_hand_state_posterior(hand_state)
        proposal_prior = _finalise_hand_state_posterior(prior_hand_state)
        if self.pre_hand_snapshot_history_round == len(self.history) and self.pre_hand_snapshot.get("available"):
            prior_snapshot = self.pre_hand_snapshot
        else:
            prior_snapshot = _pre_hand_snapshot(
                proposal_prior,
                particle_uncertainty=_clip(1.0 - ess_ratio),
                recent_ess_ratio=ess_ratio,
            )
        self.shoe_error_correction, self.error_memory = _apply_shoe_error_correction(
            self.shoe_error_correction,
            self.error_memory,
            prior_snapshot=prior_snapshot,
            posterior_hidden_given_actual=self.current_hand_posterior,
            actual=actual,
            recent_ess_ratio=ess_ratio,
            particle_uncertainty=float(prior_snapshot.get("particle_uncertainty", 1.0)),
        )
        if self.current_hand_posterior["available"]:
            # The observed B/P/T likelihood remains the only posterior weight.
            # Use its matching hidden-state expectation only to de-noise card
            # consumption, preserving the particle composition and B/P/T path.
            selected_increment = float(np.dot(weights, next_consumed - self.consumed))
            expected_increment = float(self.current_hand_posterior["expected_cards_consumed"])
            next_consumed += expected_increment - selected_increment
        self.last_resampled=should_resample(ess_ratio)
        self.last_rejuvenated=0
        if self.last_resampled:
            self.particles,self.consumed=_systematic_resample(next_particles,next_consumed,weights,self.rng)
            self.last_unique_particle_ratio=_unique_particle_ratio(self.particles)
            if self.last_unique_particle_ratio<REJUVENATION_UNIQUE_RATIO:
                self.particles,self.last_rejuvenated=_rejuvenate_particles(self.particles,self.rng)
                self.last_unique_particle_ratio=_unique_particle_ratio(self.particles)
            self.log_weights=np.full(particle_count,-math.log(particle_count),dtype=np.float64)
        else:
            self.particles,self.consumed=next_particles,next_consumed
            self.log_weights=_log_weights_from_normalised(weights)
            self.last_unique_particle_ratio=_unique_particle_ratio(self.particles)
        self.history.append(actual)

    def sync(self, history: str | Sequence[str]) -> None:
        seq = _tokens(history)
        prefix = self.history
        if len(seq) < len(prefix) or seq[:len(prefix)] != prefix:
            self.reset()
        for actual in seq[len(self.history):]:
            self._advance(actual)

    def estimate(self, history: str | Sequence[str] | None = None, *, forecast_draws: int = FORECAST_DRAWS) -> ParticlePhysicsEstimate:
        if history is not None:
            self.sync(history)
        effective_forecast_draws=max(1,int(forecast_draws))
        if self.mode=="runtime" and self.adaptive_enabled and float(self.last_diagnostics.get("posterior_uncertainty",0.0))>.55:
            effective_forecast_draws=min(4,max(effective_forecast_draws,3))
        physics, next_hand_forecast = _forecast_particles(
            self.particles,
            self.consumed,
            weights=self._normalised_weights(),
            random_state=self.random_state + len(self.history) * 1009,
            forecast_draws=effective_forecast_draws,
        )
        matrix = np.vstack(self.particles).astype(np.float64)
        weights=self._normalised_weights()
        mean_counts=np.average(matrix,axis=0,weights=weights)
        variance=np.average((matrix-mean_counts)**2,axis=0,weights=weights)
        spread = float(np.mean(np.sqrt(np.maximum(variance,0.0)) / INITIAL_PER_RANK))
        recent_ess = float(np.mean(self.ess_history[-8:])) if self.ess_history else 1.0
        consumed_mean=float(np.dot(weights,self.consumed)) if len(self.consumed) else 0.0
        consumed_std = float(np.sqrt(np.dot(weights,(self.consumed-consumed_mean)**2))) if len(self.consumed) else 0.0
        base_uncertainty = _clip(0.60 * min(1.0, spread * 4.0) + 0.40 * (1.0 - recent_ess))
        if self.history:
            draw_state_uncertainty = _clip(
                .65 * float(self.current_hand_posterior["draw_state_uncertainty"])
                + .35 * float(next_hand_forecast["draw_state_uncertainty"])
            )
            uncertainty = _clip(.75 * base_uncertainty + .25 * draw_state_uncertainty)
        else:
            draw_state_uncertainty = 0.0
            uncertainty = base_uncertainty
        physics[42]=np.float32(uncertainty)
        self.pre_hand_snapshot = _pre_hand_snapshot(
            next_hand_forecast,
            particle_uncertainty=uncertainty,
            recent_ess_ratio=recent_ess,
            physical_ev=(float(physics[39]), float(physics[40]), float(physics[41])),
        )
        self.pre_hand_snapshot_history_round = len(self.history)
        diagnostics = {
            "particle_count": float(len(self.particles)),
            "particle_budget": float(len(self.particles)),
            "particle_mode": self.mode,
            "particle_filter_version": float(self.particle_filter_version if self.adaptive_enabled else 0),
            "particle_policy": PARTICLE_POLICY if self.adaptive_enabled else "legacy_fixed_64",
            "particle_teacher_count": float(PARTICLE_TEACHER),
            "particle_runtime_min": float(PARTICLE_MIN),
            "particle_runtime_mid": float(PARTICLE_MID),
            "particle_runtime_max": float(PARTICLE_MAX),
            "forecast_draws": float(effective_forecast_draws),
            "ess_resample_threshold": ESS_RESAMPLE_THRESHOLD,
            "resampling_policy": "ess_triggered_systematic",
            "weight_policy": "log_weight_normalization",
            "resampled": self.last_resampled,
            "unique_particle_ratio": self.last_unique_particle_ratio,
            "rejuvenated_particles": float(self.last_rejuvenated),
            "budget_transition": tuple(float(value) for value in self.last_budget_transition),
            "history_rounds": float(len(self.history)),
            "expected_consumed_cards": float(physics[43]),
            "current_hand_expected_cards_consumed": float(self.current_hand_posterior["expected_cards_consumed"]),
            "consumed_cards_std": consumed_std,
            "recent_ess_ratio": recent_ess,
            "composition_spread": spread,
            "posterior_uncertainty": uncertainty,
            "draw_state_uncertainty": draw_state_uncertainty,
            "physical_ev_banker": float(physics[39]),
            "physical_ev_player": float(physics[40]),
            "physical_ev_gap": float(physics[41]),
            "physics_hand_state_version": float(PHYSICS_HAND_STATE_VERSION),
            "physics_hand_state_policy": PHYSICS_HAND_STATE_POLICY,
            "current_hand_posterior": self.current_hand_posterior,
            "next_hand_forecast": next_hand_forecast,
            "pre_hand_snapshot": self.pre_hand_snapshot,
            "shoe_error_correction_version": float(SHOE_ERROR_CORRECTION_VERSION),
            "shoe_error_correction_policy": SHOE_ERROR_CORRECTION_POLICY,
            "shoe_posterior_health": float(self.shoe_error_correction["shoe_posterior_health"]),
            "draw_state_health": float(self.shoe_error_correction["draw_state_health"]),
            "physical_ev_health": float(self.shoe_error_correction["physical_ev_health"]),
            "posterior_reliability_multiplier": float(self.shoe_error_correction["posterior_reliability_multiplier"]),
            "draw_reliability_multiplier": float(self.shoe_error_correction["draw_reliability_multiplier"]),
            "physical_ev_reliability_multiplier": float(self.shoe_error_correction["physical_ev_reliability_multiplier"]),
            "recent_miscalibration_score": float(self.shoe_error_correction["recent_miscalibration_score"]),
            "error_memory_size": float(len(self.error_memory)),
            "shoe_error_correction": dict(self.shoe_error_correction),
        }
        diagnostics["particle_quality"]=particle_quality(diagnostics)
        self.last_diagnostics=dict(diagnostics)
        return ParticlePhysicsEstimate(physics, diagnostics)


def estimate_particle_physics(
    history: str | Sequence[str],
    *,
    particle_count: int | None = None,
    mode: str = "runtime",
    particle_filter_version: int = PARTICLE_FILTER_VERSION,
    forecast_draws: int = FORECAST_DRAWS,
    random_state: int = RANDOM_STATE,
) -> ParticlePhysicsEstimate:
    tracker = ParticleShoeTracker(
        particle_count=particle_count,
        mode=mode,
        particle_filter_version=particle_filter_version,
        random_state=random_state,
    )
    return tracker.estimate(history, forecast_draws=forecast_draws)


def _effective_progress(rounds: int, diagnostics: dict[str, float]) -> float:
    """Continuous shoe progress using both hand count and inferred card consumption."""
    round_progress=_clip(float(rounds)/70.0)
    card_progress=_clip(float(diagnostics.get("expected_consumed_cards",0.0))/PLAYABLE_CARDS_ESTIMATE)
    uncertainty=_clip(diagnostics.get("posterior_uncertainty",1.0))
    round_weight=.55+.25*uncertainty
    return _clip(round_weight*round_progress+(1.0-round_weight)*card_progress)


def _fusion_weight(rounds: int, diagnostics: dict[str, float]) -> float:
    progress_round=70.0*_effective_progress(rounds,diagnostics)
    base=_interp(progress_round,(
        (0,.18),(10,.20),(20,.27),(30,.33),(40,.39),
        (50,.45),(55,.48),(60,.52),(65,.55),(70,.57),
    ))
    ess=_clip(diagnostics.get("recent_ess_ratio",1.0))
    uncertainty=_clip(diagnostics.get("posterior_uncertainty",1.0))
    reliability=(.70+.30*ess)*(1.0-.20*uncertainty)
    return _clip(base*reliability,.10,.58)


def _physical_ev_reliability(rounds: int, diagnostics: dict[str, float]) -> float:
    progress_round=70.0*_effective_progress(rounds,diagnostics)
    stage=_interp(progress_round,(
        (0,.32),(10,.36),(20,.45),(30,.53),(40,.61),
        (50,.70),(55,.75),(60,.80),(65,.84),(70,.87),
    ))
    ess=_clip(diagnostics.get("recent_ess_ratio",1.0))
    uncertainty=_clip(diagnostics.get("posterior_uncertainty",1.0))
    return _clip(stage*(.72+.28*ess)*(1.0-.30*uncertainty),.20,.90)


def particle_reliability(diagnostics: dict[str, float]) -> float:
    """Reliability of the B/P/T-only particle posterior, not a card reconstruction claim."""
    ess=_clip(diagnostics.get("recent_ess_ratio",1.0))
    uncertainty=_clip(diagnostics.get("posterior_uncertainty",1.0))
    return _clip(ess*(1.0-uncertainty))


def early35_evidence_weight(rounds: float, diagnostics: dict[str, float]) -> float:
    """Bayesian Particle evidence weight with a continuous 35→40 hand hand-off."""
    stage=_interp(rounds,EARLY35_EVIDENCE_ANCHORS)
    adjustment=.65+.35*particle_reliability(diagnostics)
    early=_clip(stage*adjustment,.10,.80)
    if rounds <= EARLY35_BRIDGE_START:
        return early
    baseline=_fusion_weight(rounds,diagnostics)
    if rounds >= EARLY35_BRIDGE_END:
        return baseline
    return _blend(early,baseline,(rounds-EARLY35_BRIDGE_START)/(EARLY35_BRIDGE_END-EARLY35_BRIDGE_START))


def early35_physical_ev_reliability(rounds: float, diagnostics: dict[str, float]) -> float:
    """Shrink raw Particle Physical EV until the posterior has enough evidence."""
    stage=_interp(rounds,EARLY35_EV_RELIABILITY_ANCHORS)
    ess=_clip(diagnostics.get("recent_ess_ratio",1.0))
    uncertainty=_clip(diagnostics.get("posterior_uncertainty",1.0))
    early=_clip(stage*(.70+.30*ess)*(1.0-.35*uncertainty),0.0,.85)
    if rounds <= EARLY35_BRIDGE_START:
        return early
    baseline=_physical_ev_reliability(rounds,diagnostics)
    if rounds >= EARLY35_BRIDGE_END:
        return baseline
    return _blend(early,baseline,(rounds-EARLY35_BRIDGE_START)/(EARLY35_BRIDGE_END-EARLY35_BRIDGE_START))


def fuse_particle_physics(
    mlp_48d: Sequence[float],
    history: str | Sequence[str],
    *,
    particle_count: int | None = None,
    particle_mode: str = "runtime",
    particle_filter_version: int | None = None,
    forecast_draws: int = FORECAST_DRAWS,
    random_state: int = RANDOM_STATE,
    tracker: ParticleShoeTracker | None = None,
    early35_version: int = EARLY35_VERSION,
    shoe_error_correction_version: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fuse rule-consistent particle estimates into the existing 48D semantics."""
    mlp = np.asarray(mlp_48d, dtype=np.float64).reshape(-1)
    if mlp.size != PHYSICS_DIM:
        raise ValueError(f"expected {PHYSICS_DIM} MLP features, got {mlp.size}")

    seq = _tokens(history)
    enabled_version=(tracker.particle_filter_version if particle_filter_version is None and tracker is not None
                     else int(particle_filter_version or 0))
    estimate = (
        tracker.estimate(seq, forecast_draws=forecast_draws)
        if tracker is not None
        else estimate_particle_physics(
            seq,
            particle_count=particle_count,
            mode=particle_mode,
            particle_filter_version=enabled_version,
            forecast_draws=forecast_draws,
            random_state=random_state,
        )
    )
    particle = estimate.physics_48d.astype(np.float64)
    early35_enabled=int(early35_version)>=EARLY35_VERSION
    weight=_fusion_weight(len(seq),estimate.diagnostics)
    evidence_weight=(early35_evidence_weight(len(seq),estimate.diagnostics)
                     if early35_enabled else weight)
    ev_reliability=(early35_physical_ev_reliability(len(seq),estimate.diagnostics)
                    if early35_enabled else _physical_ev_reliability(len(seq),estimate.diagnostics))
    quality=particle_quality(estimate.diagnostics)
    if enabled_version>=PARTICLE_FILTER_VERSION:
        # Reliability only shrinks raw Physical EV.  Its formula and direction
        # remain untouched, and a larger N alone grants no confidence bonus.
        ev_reliability*=.70+.30*quality
    correction_enabled = int(shoe_error_correction_version) >= SHOE_ERROR_CORRECTION_VERSION
    correction = estimate.diagnostics.get("shoe_error_correction", {}) if correction_enabled else {}
    posterior_multiplier = _clip(correction.get("posterior_reliability_multiplier", 1.0), .90, 1.0)
    draw_multiplier = _clip(correction.get("draw_reliability_multiplier", 1.0), .90, 1.0)
    physical_ev_multiplier = _clip(correction.get("physical_ev_reliability_multiplier", 1.0), .80, 1.0)
    raw_ev=particle[39:42].copy()
    particle[39:42]*=ev_reliability * physical_ev_multiplier
    fused = mlp.copy()

    for start, end in ((0, 3), (3, 13), (13, 23), (23, 26)):
        base_block_weight=evidence_weight if early35_enabled else weight
        block_multiplier = posterior_multiplier if start == 23 else draw_multiplier
        block_weight=base_block_weight * block_multiplier
        fused[start:end] = _normalise((1.0 - block_weight) * _normalise(mlp[start:end]) + block_weight * _normalise(particle[start:end]))

    rank_weight = weight * draw_multiplier
    fused[26:39] = (1.0 - rank_weight) * np.clip(mlp[26:39], 0.0, None) + rank_weight * particle[26:39]
    # Physical EV is deliberately particle-first and is computed before Core.
    # Do not dilute these four slots with pattern-derived MLP output.
    fused[39:43] = particle[39:43]
    fused[43:48] = (1.0 - rank_weight) * mlp[43:48] + rank_weight * particle[43:48]

    expected_cards = float(4.0 * fused[0] + 5.0 * fused[1] + 6.0 * fused[2])
    rank_total = float(np.sum(fused[26:39]))
    if rank_total > 1e-12:
        fused[26:39] *= expected_cards / rank_total

    fused[39] = np.clip(fused[39], -1.0, .95)
    fused[40] = np.clip(fused[40], -1.0, 1.0)
    fused[41] = np.clip(fused[41], -2.0, 2.0)
    fused[42] = np.clip(fused[42], 0.0, 1.0)
    fused[43] = np.clip(fused[43], 0.0, TOTAL_CARDS)
    fused[44:46] = np.clip(fused[44:46], 0.0, 1.0)
    fused[46] = np.clip(fused[46], -1.0, 1.0)
    fused[47] = np.clip(fused[47], 0.0, 1.0)

    diagnostics = dict(estimate.diagnostics)
    diagnostics["effective_progress"] = _effective_progress(len(seq),estimate.diagnostics)
    diagnostics["effective_progress_round"] = 70.0*diagnostics["effective_progress"]
    diagnostics["fusion_weight"] = float(weight)
    diagnostics["particle_evidence_weight"] = float(evidence_weight)
    diagnostics["physical_ev_reliability"] = float(ev_reliability)
    diagnostics["particle_quality"] = float(quality)
    diagnostics["particle_filter_version"] = float(PARTICLE_FILTER_VERSION if enabled_version>=PARTICLE_FILTER_VERSION else 0)
    diagnostics["particle_policy"] = PARTICLE_POLICY if enabled_version>=PARTICLE_FILTER_VERSION else "legacy_fixed_64"
    diagnostics["shoe_error_correction_version"] = float(SHOE_ERROR_CORRECTION_VERSION if correction_enabled else 0)
    diagnostics["posterior_reliability_multiplier"] = float(posterior_multiplier)
    diagnostics["draw_reliability_multiplier"] = float(draw_multiplier)
    diagnostics["physical_ev_reliability_multiplier"] = float(physical_ev_multiplier)
    diagnostics["effective_particle_evidence_weight"] = float(evidence_weight * posterior_multiplier)
    diagnostics["effective_draw_evidence_weight"] = float(evidence_weight * draw_multiplier)
    diagnostics["early35_version"] = float(EARLY35_VERSION if early35_enabled else 0)
    diagnostics["raw_physical_ev_banker"] = float(raw_ev[0])
    diagnostics["raw_physical_ev_player"] = float(raw_ev[1])
    diagnostics["physical_ev_banker"] = float(fused[39])
    diagnostics["physical_ev_player"] = float(fused[40])
    diagnostics["physical_ev_gap"] = float(fused[41])
    diagnostics["expected_next_card_count"] = expected_cards
    return fused.astype(np.float32), diagnostics
