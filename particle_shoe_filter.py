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
from typing import Any, Iterable, Sequence

import numpy as np

DECKS = 8
RANKS = 13
INITIAL_PER_RANK = DECKS * 4
TOTAL_CARDS = 52 * DECKS
PHYSICS_DIM = 48
PARTICLE_COUNT = 64
LIKELIHOOD_DRAWS = 4
FALLBACK_DRAWS = 12
FORECAST_DRAWS = 2
RANDOM_STATE = 20261005

_OUTCOMES = ("B", "P", "T")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalise(block: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(block, dtype=np.float64), 0.0, None)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full(len(x), 1.0 / max(1, len(x)), dtype=np.float64)
    return x / total


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

    @property
    def card_count(self) -> int:
        return len(self.ranks)


@dataclass(frozen=True)
class ParticlePhysicsEstimate:
    physics_48d: np.ndarray
    diagnostics: dict[str, float]


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

    if player_total not in {8, 9} and banker_total not in {8, 9}:
        player_third_value: int | None = None
        if player_total <= 5:
            p3 = _draw_rank(counts, rng)
            player.append(p3)
            player_third_value = _rank_value(p3)
            player_total = (player_total + player_third_value) % 10
        if _banker_draws(banker_total, player_third_value):
            b3 = _draw_rank(counts, rng)
            banker.append(b3)
            banker_total = (banker_total + _rank_value(b3)) % 10

    outcome = "B" if banker_total > player_total else "P" if player_total > banker_total else "T"
    return SimulatedHand(outcome, player_total, banker_total, tuple(player + banker)), counts


def _systematic_resample(
    counts: list[np.ndarray],
    consumed: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    n = len(counts)
    cdf = np.cumsum(weights)
    start = float(rng.random()) / n
    positions = start + np.arange(n, dtype=np.float64) / n
    indices = np.searchsorted(cdf, positions, side="left")
    indices = np.clip(indices, 0, n - 1)
    return [counts[int(i)].copy() for i in indices], consumed[indices].copy()


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

    for actual in history:
        next_particles: list[np.ndarray] = []
        next_consumed = np.zeros(particle_count, dtype=np.float64)
        raw_weights = np.zeros(particle_count, dtype=np.float64)

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
            raw_weights[index] = likelihood

        weight_sum = float(raw_weights.sum())
        weights = raw_weights / weight_sum if weight_sum > 1e-12 else np.full(particle_count, 1.0 / particle_count)
        ess = 1.0 / max(1e-12, float(np.sum(weights * weights)))
        ess_history.append(_clip(ess / particle_count))
        particles, consumed = _systematic_resample(next_particles, next_consumed, weights, rng)

    return particles, consumed, ess_history


def _forecast_particles(
    particles: Sequence[np.ndarray],
    consumed: np.ndarray,
    *,
    random_state: int,
    forecast_draws: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state + 104729)
    out = np.zeros(PHYSICS_DIM, dtype=np.float64)
    samples = 0

    for counts in particles:
        for _ in range(forecast_draws):
            if int(np.sum(counts)) < 6:
                continue
            hand, _ = _deal_from_counts(counts, rng)
            samples += 1
            out[{4: 0, 5: 1, 6: 2}[hand.card_count]] += 1.0
            out[3 + hand.player_point] += 1.0
            out[13 + hand.banker_point] += 1.0
            out[23 + {"B": 0, "P": 1, "T": 2}[hand.outcome]] += 1.0
            for rank in hand.ranks:
                out[26 + rank] += 1.0
            diff = hand.banker_point - hand.player_point
            out[46] += diff / 9.0
            out[47] += abs(diff) / 9.0

    if samples <= 0:
        raise RuntimeError("particle forecast produced no samples")

    out[0:3] /= samples
    out[3:13] /= samples
    out[13:23] /= samples
    out[23:26] /= samples
    out[26:39] /= samples
    p_b,p_p=float(out[23]),float(out[24])
    physical_ev_b=p_b*.95-p_p
    physical_ev_p=p_p-p_b
    out[39]=physical_ev_b
    out[40]=physical_ev_p
    out[41]=physical_ev_b-physical_ev_p
    out[42]=0.0  # filled from posterior uncertainty after ESS/composition diagnostics
    out[43] = float(np.mean(consumed)) if len(consumed) else 0.0

    mean_counts = np.mean(np.vstack(particles).astype(np.float64), axis=0)
    remaining_total = max(1.0, float(np.sum(mean_counts)))
    out[44] = float(np.sum(mean_counts[:5]) / remaining_total)
    out[45] = float(np.sum(mean_counts[8:]) / remaining_total)
    out[46] /= samples
    out[47] /= samples
    return out.astype(np.float32)


class ParticleShoeTracker:
    """Incremental particle state for chronological rows from the same shoe."""

    def __init__(self, *, particle_count: int = PARTICLE_COUNT, random_state: int = RANDOM_STATE):
        self.particle_count = max(16, int(particle_count))
        self.random_state = int(random_state)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.particles = [np.full(RANKS, INITIAL_PER_RANK, dtype=np.int16) for _ in range(self.particle_count)]
        self.consumed = np.zeros(self.particle_count, dtype=np.float64)
        self.ess_history: list[float] = []
        self.history: list[str] = []

    def _advance(self, actual: str) -> None:
        next_particles: list[np.ndarray] = []
        next_consumed = np.zeros(self.particle_count, dtype=np.float64)
        raw_weights = np.zeros(self.particle_count, dtype=np.float64)

        for index, base in enumerate(self.particles):
            proposals: list[tuple[SimulatedHand, np.ndarray]] = []
            matches: list[tuple[SimulatedHand, np.ndarray]] = []
            for _ in range(LIKELIHOOD_DRAWS):
                hand, after = _deal_from_counts(base, self.rng)
                proposals.append((hand, after))
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
            else:
                chosen_hand, chosen_after = proposals[int(self.rng.integers(len(proposals)))]
                likelihood = 1e-4

            next_particles.append(chosen_after)
            next_consumed[index] = self.consumed[index] + chosen_hand.card_count
            raw_weights[index] = likelihood

        weight_sum = float(raw_weights.sum())
        weights = raw_weights / weight_sum if weight_sum > 1e-12 else np.full(self.particle_count, 1.0 / self.particle_count)
        ess = 1.0 / max(1e-12, float(np.sum(weights * weights)))
        self.ess_history.append(_clip(ess / self.particle_count))
        self.particles, self.consumed = _systematic_resample(next_particles, next_consumed, weights, self.rng)
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
        physics = _forecast_particles(
            self.particles,
            self.consumed,
            random_state=self.random_state + len(self.history) * 1009,
            forecast_draws=max(1, int(forecast_draws)),
        )
        matrix = np.vstack(self.particles).astype(np.float64)
        spread = float(np.mean(np.std(matrix, axis=0) / INITIAL_PER_RANK))
        recent_ess = float(np.mean(self.ess_history[-8:])) if self.ess_history else 1.0
        consumed_std = float(np.std(self.consumed)) if len(self.consumed) else 0.0
        uncertainty = _clip(0.60 * min(1.0, spread * 4.0) + 0.40 * (1.0 - recent_ess))
        physics[42]=np.float32(uncertainty)
        diagnostics = {
            "particle_count": float(self.particle_count),
            "history_rounds": float(len(self.history)),
            "expected_consumed_cards": float(physics[43]),
            "consumed_cards_std": consumed_std,
            "recent_ess_ratio": recent_ess,
            "composition_spread": spread,
            "posterior_uncertainty": uncertainty,
            "physical_ev_banker": float(physics[39]),
            "physical_ev_player": float(physics[40]),
            "physical_ev_gap": float(physics[41]),
        }
        return ParticlePhysicsEstimate(physics, diagnostics)


def estimate_particle_physics(
    history: str | Sequence[str],
    *,
    particle_count: int = PARTICLE_COUNT,
    forecast_draws: int = FORECAST_DRAWS,
    random_state: int = RANDOM_STATE,
) -> ParticlePhysicsEstimate:
    tracker = ParticleShoeTracker(particle_count=particle_count, random_state=random_state)
    return tracker.estimate(history, forecast_draws=forecast_draws)


def _fusion_weight(rounds: int, diagnostics: dict[str, float]) -> float:
    if rounds < 12:
        base = 0.18
    elif rounds < 20:
        base = 0.25
    elif rounds <= 40:
        base = 0.34
    elif rounds <= 50:
        base = 0.40
    else:
        base = 0.46
    reliability = 0.75 + 0.25 * _clip(diagnostics.get("recent_ess_ratio", 1.0))
    return _clip(base * reliability, 0.10, 0.46)


def fuse_particle_physics(
    mlp_48d: Sequence[float],
    history: str | Sequence[str],
    *,
    particle_count: int = PARTICLE_COUNT,
    forecast_draws: int = FORECAST_DRAWS,
    random_state: int = RANDOM_STATE,
    tracker: ParticleShoeTracker | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fuse rule-consistent particle estimates into the existing 48D semantics."""
    mlp = np.asarray(mlp_48d, dtype=np.float64).reshape(-1)
    if mlp.size != PHYSICS_DIM:
        raise ValueError(f"expected {PHYSICS_DIM} MLP features, got {mlp.size}")

    seq = _tokens(history)
    estimate = (
        tracker.estimate(seq, forecast_draws=forecast_draws)
        if tracker is not None
        else estimate_particle_physics(
            seq,
            particle_count=particle_count,
            forecast_draws=forecast_draws,
            random_state=random_state,
        )
    )
    particle = estimate.physics_48d.astype(np.float64)
    weight = _fusion_weight(len(seq), estimate.diagnostics)
    fused = mlp.copy()

    for start, end in ((0, 3), (3, 13), (13, 23), (23, 26)):
        fused[start:end] = _normalise((1.0 - weight) * _normalise(mlp[start:end]) + weight * _normalise(particle[start:end]))

    fused[26:39] = (1.0 - weight) * np.clip(mlp[26:39], 0.0, None) + weight * particle[26:39]
    # Physical EV is deliberately particle-first and is computed before Core.
    # Do not dilute these four slots with pattern-derived MLP output.
    fused[39:43] = particle[39:43]
    fused[43:48] = (1.0 - weight) * mlp[43:48] + weight * particle[43:48]

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
    diagnostics["fusion_weight"] = float(weight)
    diagnostics["expected_next_card_count"] = expected_cards
    return fused.astype(np.float32), diagnostics
