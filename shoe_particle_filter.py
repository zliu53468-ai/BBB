#!/usr/bin/env python3
"""Blind physical Shoe Particle Filter for BBB downstream 10D inference.

The application does not need to observe hidden card identities. Each particle
is one plausible remaining eight-deck baccarat shoe, represented by point-value
counts 0..9.

The particle posterior is updated causally from settled B/P outcomes and the
frozen 256D Core residual. If real total-card count / final points happen to be
available in historical data, they can be supplied as optional extra evidence;
blind runtime operation does not require them and never invents them.

Before each next-round prediction, the particle population performs a
side-effect-free one-step Monte Carlo rollout conditioned softly on the current
Core P(B), producing three physical expectations:

    pred_card_count
    pred_banker_point
    pred_player_point
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

INITIAL_POINT_COUNTS = np.asarray([128] + [32] * 9, dtype=np.int16)
PHYSICAL_FEATURE_NAMES: tuple[str, ...] = (
    "pred_card_count",
    "pred_banker_point",
    "pred_player_point",
)

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
    "likelihood_weights": {
        "outcome": 0.55,
        "total_cards": 0.12,
        "points": 0.18,
        "core_residual": 0.15,
    },
    "core_forecast_strength": 0.35,
    "persistence_boost": 1.15,
    "turbulence_uniform_mix": 0.35,
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


class DeterministicRNG:
    """Cross-runtime LCG shared by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE


class ShoeParticleFilter:
    """1000-particle blind remaining-shoe Monte Carlo estimator."""

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
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_core_alignment: float | None = None
        self.last_turbulence_break = False
        self.reset()

    def reset(self) -> None:
        """Reset a new shoe to the standard 416-card eight-deck distribution."""
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.tile(INITIAL_POINT_COUNTS, (self.n_particles, 1))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_core_alignment = None
        self.last_turbulence_break = False

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

    def _core_alignment(self, *, actual_b: float, core_pb: float) -> float:
        predicted_b = clip(float(core_pb), 0.0, 1.0) > 0.5
        actual_is_b = float(actual_b) >= 0.5
        return 1.0 if predicted_b == actual_is_b else -1.0

    def _log_likelihood(
        self,
        simulated: SimulatedRound,
        *,
        actual_b: float,
        core_pb: float,
        observed_total_cards: int | None,
        observed_player_point: int | None,
        observed_banker_point: int | None,
        persistence_multiplier: float,
    ) -> float:
        actual = 1.0 if float(actual_b) >= 0.5 else 0.0
        observed_sign = 1.0 if actual >= 0.5 else -1.0
        weights = PF_CONFIG["likelihood_weights"]

        weighted_error = float(weights["outcome"]) * ((observed_sign - simulated.sign) / 2.0) ** 2
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
            player_error = (float(observed_player_point) - simulated.player_total) / 9.0
            banker_error = (float(observed_banker_point) - simulated.banker_total) / 9.0
            point_error = 0.5 * (player_error * player_error + banker_error * banker_error)
            weighted_error += float(weights["points"]) * point_error
            active_weight += float(weights["points"])

        core_residual = actual - clip(float(core_pb), 0.0, 1.0)
        core_target = float(np.clip(2.0 * core_residual, -1.0, 1.0))
        if simulated.sign == 0:
            simulated_support = 0.0
        else:
            margin = abs(simulated.banker_total - simulated.player_total) / 9.0
            simulated_support = float(simulated.sign) * (0.5 + 0.5 * margin)

        core_error = core_target - simulated_support
        weighted_error += float(weights["core_residual"]) * core_error * core_error
        active_weight += float(weights["core_residual"])

        normalized_error = weighted_error / max(active_weight, 1e-12)
        return (
            -0.5
            * float(persistence_multiplier)
            * normalized_error
            / max(self.r, 1e-12)
        )

    def systematic_resample(self) -> None:
        positions = (self.rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.n_particles)

    def _rejuvenate(self, q_eff: float) -> None:
        """Inject hidden-card uncertainty while preserving total cards remaining."""
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
                sources[min(len(sources) - 1, int(self.rng.uniform() * len(sources)))]
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

    def predict_physical_features(
        self,
        *,
        core_pb: float,
        current_round: float,
    ) -> np.ndarray:
        """One-step blind physical forecast without mutating the PF state.

        The current Core P(B) softly reweights forward rollouts. When Core is
        near 0.5 the conditioning is almost neutral; stronger Core confidence
        gives more weight to physically plausible rollouts that agree with it.
        """
        del current_round  # round dependence is already encoded in posterior/Q history
        saved_rng_state = self.rng.state

        weighted_card_count = 0.0
        weighted_banker_point = 0.0
        weighted_player_point = 0.0
        total_weight = 0.0

        p_b = clip(float(core_pb), 0.0, 1.0)
        expected_sign = 2.0 * p_b - 1.0
        core_confidence = abs(expected_sign)
        strength = float(PF_CONFIG["core_forecast_strength"])

        try:
            for i in range(self.n_particles):
                counts = self.particles[i].copy()
                simulated = self._simulate_round(counts)
                particle_weight = float(self.weights[i])

                sign_value = float(simulated.sign)
                alignment_error = sign_value - expected_sign
                core_factor = math.exp(
                    -0.5
                    * strength
                    * core_confidence
                    * alignment_error
                    * alignment_error
                    / max(self.r, 1e-12)
                )
                weight = particle_weight * core_factor
                total_weight += weight
                weighted_card_count += weight * float(simulated.total_cards)
                weighted_banker_point += weight * float(simulated.banker_total)
                weighted_player_point += weight * float(simulated.player_total)
        finally:
            self.rng.state = saved_rng_state

        if total_weight <= 1e-12:
            return np.asarray([4.8, 4.5, 4.5], dtype=np.float64)

        return np.asarray(
            [
                np.clip(weighted_card_count / total_weight, 4.0, 6.0),
                np.clip(weighted_banker_point / total_weight, 0.0, 9.0),
                np.clip(weighted_player_point / total_weight, 0.0, 9.0),
            ],
            dtype=np.float64,
        )

    def observe_and_project(
        self,
        *,
        real_outcome: int | float,
        core_pb: float,
        current_round: float,
        observed_total_cards: int | None = None,
        observed_player_point: int | None = None,
        observed_banker_point: int | None = None,
    ) -> None:
        """Update posterior after a settled round.

        Physical observations are optional. In fully blind operation only
        outcome + Core residual are used.
        """
        actual_b = 1.0 if float(real_outcome) >= 0.5 else 0.0
        alignment = self._core_alignment(actual_b=actual_b, core_pb=core_pb)
        turbulence_break = (
            self.last_core_alignment is not None
            and alignment != self.last_core_alignment
        )

        if turbulence_break:
            uniform = 1.0 / self.n_particles
            mix = float(PF_CONFIG["turbulence_uniform_mix"])
            self.weights = (1.0 - mix) * self.weights + mix * uniform
            self.weights /= float(np.sum(self.weights))

        persistence_multiplier = (
            float(PF_CONFIG["persistence_boost"])
            if self.last_core_alignment == alignment
            else 1.0
        )

        log_weights = np.empty(self.n_particles, dtype=np.float64)
        for i in range(self.n_particles):
            counts = self.particles[i].copy()
            simulated = self._simulate_round(counts)
            self.particles[i] = counts
            log_weights[i] = self._log_likelihood(
                simulated,
                actual_b=actual_b,
                core_pb=core_pb,
                observed_total_cards=observed_total_cards,
                observed_player_point=observed_player_point,
                observed_banker_point=observed_banker_point,
                persistence_multiplier=persistence_multiplier,
            )

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
        self.last_effective_q = q_eff
        self.last_core_alignment = alignment
        self.last_turbulence_break = turbulence_break
        self.updates += 1


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
