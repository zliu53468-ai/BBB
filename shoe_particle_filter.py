#!/usr/bin/env python3
"""Enhanced blind-box Shoe Particle Filter for BBB.

The filter never observes hidden card identities. Each particle is one plausible
remaining eight-deck shoe represented by baccarat point values 0..9.

When physical observations are available, the Bayesian likelihood uses:
- settled B/P outcome;
- total cards dealt in the round (4/5/6);
- final Player and Banker points;
- frozen 256D Core residual (actual_B - core_pb).

High-information 5/6-card rounds receive 1.5x likelihood precision. A hard
remaining-card constraint suppresses particles whose total consumption is
physically implausible relative to 4.8 cards per settled round.

The public inference output is a four-dimensional next-round tensor:
[p_4cards, p_6cards, win_point, lose_point].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

INITIAL_POINT_COUNTS = np.asarray([128] + [32] * 9, dtype=np.int16)
INITIAL_TOTAL_CARDS = int(np.sum(INITIAL_POINT_COUNTS))
PSEUDO_CARD_FEATURE_NAMES: tuple[str, ...] = (
    "p_4cards",
    "p_6cards",
    "win_point",
    "lose_point",
)

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "decks": 8,
    "point_bins": 10,
    "initial_point_counts": INITIAL_POINT_COUNTS.tolist(),
    "Q_early": 0.005,
    "Q_late": 0.025,
    "early_round_end": 15,
    "late_round_start": 45,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
    "random_state": 42,
    "info_gain_multiplier": 1.5,
    "expected_cards_per_round": 4.8,
    "constraint_sigma_per_sqrt_round": 0.85,
    "constraint_hard_z": 3.5,
    "likelihood_weights": {
        "outcome": 0.40,
        "total_cards": 0.22,
        "points": 0.23,
        "core_residual": 0.15,
    },
}

_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = float(2**32)


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class SimulatedRound:
    sign: int
    player_total: int
    banker_total: int
    player_cards: int
    banker_cards: int

    @property
    def total_cards(self) -> int:
        return self.player_cards + self.banker_cards

    @property
    def winner_point(self) -> int | None:
        if self.sign > 0:
            return self.banker_total
        if self.sign < 0:
            return self.player_total
        return None

    @property
    def loser_point(self) -> int | None:
        if self.sign > 0:
            return self.player_total
        if self.sign < 0:
            return self.banker_total
        return None


class DeterministicRNG:
    """Cross-runtime LCG used by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE


class EnhancedShoeParticleFilter:
    """1000-particle latent eight-deck filter with physical constraints."""

    def __init__(
        self,
        *,
        n_particles: int = 1000,
        q_early: float = 0.005,
        q_late: float = 0.025,
        early_round_end: int = 15,
        late_round_start: int = 45,
        r: float = 0.25,
        resample_threshold: float = 500.0,
        random_state: int = 42,
    ) -> None:
        self.n_particles = int(n_particles)
        self.q_early = float(q_early)
        self.q_late = float(q_late)
        self.early_round_end = int(early_round_end)
        self.late_round_start = int(late_round_start)
        self.r = float(r)
        self.resample_threshold = float(resample_threshold)
        self.random_state = int(random_state)

        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.tile(INITIAL_POINT_COUNTS, (self.n_particles, 1))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_information_multiplier = 1.0
        self.last_constraint_survival = 1.0
        self._tensor = np.zeros(4, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.tile(INITIAL_POINT_COUNTS, (self.n_particles, 1))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_information_multiplier = 1.0
        self.last_constraint_survival = 1.0
        self._tensor = np.zeros(4, dtype=np.float64)

    def effective_sample_size(self) -> float:
        denom = float(np.sum(self.weights * self.weights))
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def effective_q(self, current_round: float) -> float:
        current = float(current_round)
        if current < self.early_round_end:
            return self.q_early
        if current > self.late_round_start:
            return self.q_late
        span = max(1.0, float(self.late_round_start - self.early_round_end))
        ratio = float(np.clip((current - self.early_round_end) / span, 0.0, 1.0))
        return self.q_early + (self.q_late - self.q_early) * ratio

    def _draw_point(self, counts: np.ndarray) -> int:
        total = int(np.sum(counts))
        if total <= 0:
            raise RuntimeError("virtual shoe is empty")
        threshold = self.rng.uniform() * total
        running = 0
        for point in range(10):
            running += int(counts[point])
            if threshold < running:
                counts[point] -= 1
                return point
        counts[9] -= 1
        return 9

    @staticmethod
    def _banker_draws(banker_total: int, player_third: int | None) -> bool:
        if player_third is None:
            return banker_total <= 5
        if banker_total <= 2:
            return True
        if banker_total == 3:
            return player_third != 8
        if banker_total == 4:
            return 2 <= player_third <= 7
        if banker_total == 5:
            return 4 <= player_third <= 7
        if banker_total == 6:
            return 6 <= player_third <= 7
        return False

    def _simulate_round(self, counts: np.ndarray) -> SimulatedRound:
        if int(np.sum(counts)) < 6:
            return SimulatedRound(0, 0, 0, 0, 0)

        player = [self._draw_point(counts)]
        banker = [self._draw_point(counts)]
        player.append(self._draw_point(counts))
        banker.append(self._draw_point(counts))

        player_total = sum(player) % 10
        banker_total = sum(banker) % 10
        natural = player_total in (8, 9) or banker_total in (8, 9)

        player_third: int | None = None
        if not natural:
            if player_total <= 5:
                player_third = self._draw_point(counts)
                player.append(player_third)
                player_total = sum(player) % 10

            if self._banker_draws(banker_total, player_third):
                banker.append(self._draw_point(counts))
                banker_total = sum(banker) % 10

        sign = 1 if banker_total > player_total else -1 if player_total > banker_total else 0
        return SimulatedRound(
            sign=sign,
            player_total=player_total,
            banker_total=banker_total,
            player_cards=len(player),
            banker_cards=len(banker),
        )

    def _information_multiplier(self, observed_total_cards: int | None) -> float:
        return (
            float(PF_CONFIG["info_gain_multiplier"])
            if observed_total_cards in (5, 6)
            else 1.0
        )

    def _log_likelihood(
        self,
        simulated: SimulatedRound,
        *,
        real_outcome: int | float,
        core_pb: float,
        observed_total_cards: int | None,
        observed_player_point: int | None,
        observed_banker_point: int | None,
    ) -> float:
        actual_b = 1.0 if float(real_outcome) >= 0.5 else 0.0
        observed_sign = 1.0 if actual_b >= 0.5 else -1.0
        simulated_sign = float(simulated.sign)

        weights = PF_CONFIG["likelihood_weights"]
        weighted_error = float(weights["outcome"]) * ((observed_sign - simulated_sign) / 2.0) ** 2
        active_weight = float(weights["outcome"])

        if observed_total_cards in (4, 5, 6):
            card_error = (float(observed_total_cards) - simulated.total_cards) / 2.0
            weighted_error += float(weights["total_cards"]) * card_error * card_error
            active_weight += float(weights["total_cards"])

        if (
            observed_player_point is not None
            and observed_banker_point is not None
            and 0 <= observed_player_point <= 9
            and 0 <= observed_banker_point <= 9
        ):
            p_error = (float(observed_player_point) - simulated.player_total) / 9.0
            b_error = (float(observed_banker_point) - simulated.banker_total) / 9.0
            point_error = 0.5 * (p_error * p_error + b_error * b_error)
            weighted_error += float(weights["points"]) * point_error
            active_weight += float(weights["points"])

        core_residual = actual_b - clip(float(core_pb), 0.0, 1.0)
        core_target = float(np.clip(2.0 * core_residual, -1.0, 1.0))
        if simulated.sign == 0:
            simulated_support = 0.0
        else:
            point_margin = abs(simulated.banker_total - simulated.player_total) / 9.0
            simulated_support = float(simulated.sign) * (0.5 + 0.5 * point_margin)
        core_error = core_target - simulated_support
        weighted_error += float(weights["core_residual"]) * core_error * core_error
        active_weight += float(weights["core_residual"])

        normalized_error = weighted_error / max(active_weight, 1e-12)
        info_multiplier = self._information_multiplier(observed_total_cards)
        return -0.5 * info_multiplier * normalized_error / max(self.r, 1e-12)

    def _apply_physical_constraint(self, current_round: float) -> None:
        rounds = max(1.0, float(current_round))
        expected_remaining = (
            INITIAL_TOTAL_CARDS
            - float(PF_CONFIG["expected_cards_per_round"]) * rounds
        )
        sigma = max(
            3.0,
            float(PF_CONFIG["constraint_sigma_per_sqrt_round"]) * math.sqrt(rounds),
        )
        hard_z = float(PF_CONFIG["constraint_hard_z"])

        remaining = np.sum(self.particles, axis=1).astype(np.float64)
        z = np.abs(remaining - expected_remaining) / sigma
        survivors = z <= hard_z
        self.last_constraint_survival = float(np.mean(survivors))

        if np.any(survivors):
            self.weights[~survivors] = 0.0
            total = float(np.sum(self.weights))
            if total > 0.0 and np.isfinite(total):
                self.weights /= total
                return

        # Degenerate fallback: retain a soft physical prior rather than collapsing.
        soft = np.exp(-0.5 * np.minimum(z, hard_z) ** 2)
        self.weights *= soft
        total = float(np.sum(self.weights))
        if total <= 0.0 or not np.isfinite(total):
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

    def systematic_resample(self) -> None:
        positions = (self.rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.n_particles)

    def _rejuvenate(self, q_eff: float) -> None:
        n_mutations = int(math.ceil(self.n_particles * max(0.0, min(1.0, q_eff))))
        for _ in range(n_mutations):
            idx = min(
                self.n_particles - 1,
                int(self.rng.uniform() * self.n_particles),
            )
            counts = self.particles[idx]
            sources = np.flatnonzero(counts > 0)
            destinations = np.flatnonzero(counts < INITIAL_POINT_COUNTS)
            if len(sources) == 0 or len(destinations) == 0:
                continue
            src = int(
                sources[
                    min(
                        len(sources) - 1,
                        int(self.rng.uniform() * len(sources)),
                    )
                ]
            )
            valid_destinations = destinations[destinations != src]
            if len(valid_destinations) == 0:
                continue
            dst = int(
                valid_destinations[
                    min(
                        len(valid_destinations) - 1,
                        int(self.rng.uniform() * len(valid_destinations)),
                    )
                ]
            )
            counts[src] -= 1
            counts[dst] += 1

    def _forecast_tensor(self) -> np.ndarray:
        p4_mass = 0.0
        p6_mass = 0.0
        decisive_mass = 0.0
        weighted_win_point = 0.0
        weighted_lose_point = 0.0
        total_weight = float(np.sum(self.weights))

        if total_weight <= 0.0:
            return np.zeros(4, dtype=np.float64)

        for i in range(self.n_particles):
            counts = self.particles[i].copy()
            simulated = self._simulate_round(counts)
            weight = float(self.weights[i])

            if simulated.total_cards == 4:
                p4_mass += weight
            if simulated.total_cards == 6:
                p6_mass += weight

            if simulated.sign != 0:
                decisive_mass += weight
                weighted_win_point += weight * float(simulated.winner_point)
                weighted_lose_point += weight * float(simulated.loser_point)

        p_4cards = p4_mass / total_weight
        p_6cards = p6_mass / total_weight
        win_point = weighted_win_point / decisive_mass if decisive_mass > 1e-12 else 0.0
        lose_point = weighted_lose_point / decisive_mass if decisive_mass > 1e-12 else 0.0

        return np.asarray(
            [
                np.clip(p_4cards, 0.0, 1.0),
                np.clip(p_6cards, 0.0, 1.0),
                np.clip(win_point, 0.0, 9.0),
                np.clip(lose_point, 0.0, 9.0),
            ],
            dtype=np.float64,
        )

    def current_pseudo_card_feature(self) -> np.ndarray:
        return self._tensor.copy()

    def observe_and_project(
        self,
        *,
        real_outcome: int | float,
        core_pb: float,
        current_round: float,
        observed_total_cards: int | None = None,
        observed_player_point: int | None = None,
        observed_banker_point: int | None = None,
    ) -> np.ndarray:
        log_weights = np.empty(self.n_particles, dtype=np.float64)

        for i in range(self.n_particles):
            counts = self.particles[i].copy()
            simulated = self._simulate_round(counts)
            self.particles[i] = counts
            log_weights[i] = self._log_likelihood(
                simulated,
                real_outcome=real_outcome,
                core_pb=core_pb,
                observed_total_cards=observed_total_cards,
                observed_player_point=observed_player_point,
                observed_banker_point=observed_banker_point,
            )

        log_weights -= float(np.max(log_weights))
        self.weights *= np.exp(log_weights)
        total = float(np.sum(self.weights))
        if total <= 0.0 or not np.isfinite(total):
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        self.last_information_multiplier = self._information_multiplier(
            observed_total_cards
        )
        self._apply_physical_constraint(current_round)

        ess = self.effective_sample_size()
        self.last_ess = ess
        self.last_resampled = False
        if ess < self.resample_threshold:
            self.systematic_resample()
            self.last_resampled = True

        q_eff = self.effective_q(current_round)
        self._rejuvenate(q_eff)
        self._tensor = self._forecast_tensor()
        self.last_effective_q = q_eff
        self.updates += 1
        return self.current_pseudo_card_feature()


# Backward-safe alias for callers that import the old class name.
ShoeParticleFilter = EnhancedShoeParticleFilter


def new_shoe_particle_filter() -> EnhancedShoeParticleFilter:
    return EnhancedShoeParticleFilter(
        n_particles=PF_CONFIG["n_particles"],
        q_early=PF_CONFIG["Q_early"],
        q_late=PF_CONFIG["Q_late"],
        early_round_end=PF_CONFIG["early_round_end"],
        late_round_start=PF_CONFIG["late_round_start"],
        r=PF_CONFIG["R"],
        resample_threshold=PF_CONFIG["resample_threshold"],
        random_state=PF_CONFIG["random_state"],
    )
