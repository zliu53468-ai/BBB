#!/usr/bin/env python3
"""Blind-box Shoe Particle Filter for BBB downstream pseudo-count estimation.

The filter never observes actual card identities. Each particle is one plausible
remaining eight-deck shoe represented by baccarat point values 0..9. Particles
are propagated by standard baccarat drawing rules with sampling without
replacement, then reweighted from the observed B/P result and Core residual.

This is a probabilistic latent-shoe simulation, not knowledge of the true
remaining cards.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

INITIAL_POINT_COUNTS = np.asarray([128] + [32] * 9, dtype=np.int16)

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "decks": 8,
    "point_bins": 10,
    "initial_point_counts": INITIAL_POINT_COUNTS.tolist(),
    "Q_early": 0.005,
    "Q_late": 0.02,
    "early_round_end": 15,
    "late_round_start": 45,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
    "random_state": 42,
    "pseudo_count_clip": 1.0,
}

_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = float(2**32)


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


class DeterministicRNG:
    """Cross-runtime LCG used by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE


class ShoeParticleFilter:
    """1000-particle blind remaining-shoe Monte Carlo filter."""

    def __init__(
        self,
        *,
        n_particles: int = 1000,
        q_early: float = 0.005,
        q_late: float = 0.02,
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
        self.pseudo_count = 0.0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_residual = 0.0
        self.last_observed_outcome = 0
        self.reset()

    def reset(self) -> None:
        """Reset all particles to a fresh eight-deck shoe and pseudo_count=0."""
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.tile(INITIAL_POINT_COUNTS, (self.n_particles, 1))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.pseudo_count = 0.0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_residual = 0.0
        self.last_observed_outcome = 0

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

    def _simulate_round(self, counts: np.ndarray) -> tuple[int, int, int]:
        """Consume one virtual baccarat round.

        Returns:
            outcome_sign: +1 B, -1 P, 0 T
            player_total: final point total
            banker_total: final point total
        """
        if int(np.sum(counts)) < 6:
            return 0, 0, 0

        player_cards = [self._draw_point(counts)]
        banker_cards = [self._draw_point(counts)]
        player_cards.append(self._draw_point(counts))
        banker_cards.append(self._draw_point(counts))

        player_total = sum(player_cards) % 10
        banker_total = sum(banker_cards) % 10
        natural = player_total in (8, 9) or banker_total in (8, 9)

        player_third: int | None = None
        if not natural:
            if player_total <= 5:
                player_third = self._draw_point(counts)
                player_cards.append(player_third)
                player_total = sum(player_cards) % 10

            if self._banker_draws(banker_total, player_third):
                banker_cards.append(self._draw_point(counts))
                banker_total = sum(banker_cards) % 10

        if banker_total > player_total:
            return 1, player_total, banker_total
        if player_total > banker_total:
            return -1, player_total, banker_total
        return 0, player_total, banker_total

    def _proposal_likelihood(
        self,
        *,
        simulated_sign: int,
        player_total: int,
        banker_total: int,
        actual_b: float,
        core_pb: float,
    ) -> float:
        actual = 1.0 if float(actual_b) >= 0.5 else 0.0
        observed_sign = 1.0 if actual >= 0.5 else -1.0
        residual = actual - clip(float(core_pb), 0.0, 1.0)

        # More surprising Core residuals provide stronger evidence when selecting
        # among latent virtual shoes.
        observed_strength = 0.5 + 0.5 * min(1.0, abs(residual))
        target_score = observed_sign * observed_strength

        if simulated_sign == 0:
            simulated_score = 0.0
        else:
            point_margin = abs(float(banker_total) - float(player_total)) / 9.0
            simulated_score = float(simulated_sign) * (0.5 + 0.5 * point_margin)

        error = target_score - simulated_score
        variance = max(self.r, 1e-12)
        return math.exp(-0.5 * error * error / variance)

    def systematic_resample(self) -> None:
        positions = (self.rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.n_particles)

    def _rejuvenate(self, q_eff: float) -> None:
        """Inject hidden-card uncertainty while preserving cards remaining.

        Q is interpreted as the fraction of particles receiving one feasible
        point-bin swap after each observed round.
        """
        n_mutations = int(math.ceil(self.n_particles * max(0.0, min(1.0, q_eff))))
        for _ in range(n_mutations):
            idx = min(self.n_particles - 1, int(self.rng.uniform() * self.n_particles))
            counts = self.particles[idx]
            sources = np.flatnonzero(counts > 0)
            destinations = np.flatnonzero(counts < INITIAL_POINT_COUNTS)
            if len(sources) == 0 or len(destinations) == 0:
                continue
            src = int(sources[min(len(sources) - 1, int(self.rng.uniform() * len(sources)))])
            valid_destinations = destinations[destinations != src]
            if len(valid_destinations) == 0:
                continue
            dst = int(valid_destinations[min(len(valid_destinations) - 1, int(self.rng.uniform() * len(valid_destinations)))])
            counts[src] -= 1
            counts[dst] += 1

    def _forecast_pseudo_count(self) -> float:
        banker_mass = 0.0
        player_mass = 0.0
        for i in range(self.n_particles):
            counts = self.particles[i].copy()
            sign, _, _ = self._simulate_round(counts)
            weight = float(self.weights[i])
            if sign > 0:
                banker_mass += weight
            elif sign < 0:
                player_mass += weight

        decisive_mass = banker_mass + player_mass
        if decisive_mass <= 1e-12:
            return 0.0
        value = (banker_mass - player_mass) / decisive_mass
        return float(np.clip(value, -1.0, 1.0))

    def current_pseudo_count(self) -> float:
        return float(np.clip(self.pseudo_count, -1.0, 1.0))

    def observe_and_project(
        self,
        *,
        real_outcome: int | float,
        core_pb: float,
        current_round: float,
    ) -> float:
        """Propagate latent shoes, weight them, then forecast the next-round bias."""
        actual_b = 1.0 if float(real_outcome) >= 0.5 else 0.0
        log_weights = np.empty(self.n_particles, dtype=np.float64)

        for i in range(self.n_particles):
            counts = self.particles[i].copy()
            sign, player_total, banker_total = self._simulate_round(counts)
            self.particles[i] = counts
            likelihood = self._proposal_likelihood(
                simulated_sign=sign,
                player_total=player_total,
                banker_total=banker_total,
                actual_b=actual_b,
                core_pb=core_pb,
            )
            log_weights[i] = math.log(max(likelihood, 1e-300))

        log_weights -= float(np.max(log_weights))
        self.weights *= np.exp(log_weights)
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        ess = self.effective_sample_size()
        self.last_ess = ess
        self.last_resampled = False
        if ess < self.resample_threshold:
            self.systematic_resample()
            self.last_resampled = True

        q_eff = self.effective_q(current_round)
        self._rejuvenate(q_eff)
        self.pseudo_count = self._forecast_pseudo_count()
        self.last_effective_q = q_eff
        self.last_residual = actual_b - clip(float(core_pb), 0.0, 1.0)
        self.last_observed_outcome = int(actual_b)
        self.updates += 1
        return self.current_pseudo_count()


def new_shoe_particle_filter() -> ShoeParticleFilter:
    return ShoeParticleFilter(
        n_particles=PF_CONFIG["n_particles"],
        q_early=PF_CONFIG["Q_early"],
        q_late=PF_CONFIG["Q_late"],
        early_round_end=PF_CONFIG["early_round_end"],
        late_round_start=PF_CONFIG["late_round_start"],
        r=PF_CONFIG["R"],
        resample_threshold=PF_CONFIG["resample_threshold"],
        random_state=PF_CONFIG["random_state"],
    )
