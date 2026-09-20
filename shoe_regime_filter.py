#!/usr/bin/env python3
"""Shoe Regime Particle Filter for BBB downstream state estimation.

Frozen upstream inputs are not modified. This module receives only the settled
outcome, Core P(B), and current shoe round so it can estimate the current shoe
environment. It never performs physical card counting or remaining-card
inference.

State semantics:
    +1.0  strong, sustained Core-aligned regularity
     0.0  turbulence / non-regular environment
    -1.0  sustained Core-opposed regularity
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "state_dim": 1,
    "Q_early": 0.005,
    "Q_late": 0.02,
    "early_round_end": 15,
    "late_round_start": 45,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
    "random_state": 42,
    "state_clip": 1.0,
    "likelihood_weights": {
        "directionality": 0.55,
        "residual_alignment": 0.30,
        "persistence": 0.15,
    },
}

_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = float(2**32)


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


class DeterministicRNG:
    """Cross-runtime LCG shared by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE

    def normal(self) -> float:
        u1 = max(self.uniform(), 1e-15)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class ShoeRegimeParticleFilter:
    """Bootstrap PF tracking regularity vs turbulence inside one shoe."""

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
        state_clip: float = 1.0,
    ) -> None:
        self.n_particles = int(n_particles)
        self.q_early = float(q_early)
        self.q_late = float(q_late)
        self.early_round_end = int(early_round_end)
        self.late_round_start = int(late_round_start)
        self.r = float(r)
        self.resample_threshold = float(resample_threshold)
        self.random_state = int(random_state)
        self.state_clip = float(state_clip)

        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.zeros(self.n_particles, dtype=np.float64)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_alignment: float | None = None
        self.last_observation = 0.0
        self.last_measurements = {
            "directionality": 0.0,
            "residual_alignment": 0.0,
            "persistence": 0.0,
        }
        self.last_effective_q = self.q_early
        self.last_resampled = False
        self.last_ess = float(self.n_particles)
        self.reset()

    def reset(self) -> None:
        """Start a new shoe with every particle and regime_state exactly zero."""
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.zeros(self.n_particles, dtype=np.float64)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_alignment = None
        self.last_observation = 0.0
        self.last_measurements = {
            "directionality": 0.0,
            "residual_alignment": 0.0,
            "persistence": 0.0,
        }
        self.last_effective_q = self.q_early
        self.last_resampled = False
        self.last_ess = float(self.n_particles)

    def estimate(self) -> float:
        value = float(np.sum(self.particles * self.weights))
        return float(np.clip(value, -self.state_clip, self.state_clip))

    def effective_sample_size(self) -> float:
        denom = float(np.sum(self.weights * self.weights))
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def systematic_resample(self) -> None:
        positions = (self.rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    def effective_q(self, current_round: float) -> float:
        """Round-adaptive process noise: 0.005 early, 0.02 near shoe tail."""
        current = float(current_round)
        if current < self.early_round_end:
            return self.q_early
        if current > self.late_round_start:
            return self.q_late

        span = max(1.0, float(self.late_round_start - self.early_round_end))
        ratio = float(np.clip((current - self.early_round_end) / span, 0.0, 1.0))
        return self.q_early + (self.q_late - self.q_early) * ratio

    def measurement_components(
        self,
        *,
        actual_b: float,
        core_pb: float,
    ) -> tuple[dict[str, float], float, bool]:
        """Create the three likelihood measurements from the settled round.

        directionality:
            +1 if Core direction matched the result, else -1.
        residual_alignment:
            1 - 2*abs(actual_B-core_pb), clipped to [-1,1].
            High-confidence correct outcomes approach +1; confident misses
            approach -1.
        persistence:
            +1/-1 when the current correct/miss state repeats, otherwise 0.

        A sudden change from correct->miss or miss->correct is treated as
        turbulence. In that case all likelihood measurements become zero so
        the posterior is pulled back toward a neutral regime instead of
        immediately declaring a new trend.
        """
        actual = 1.0 if float(actual_b) >= 0.5 else 0.0
        p_b = clip(float(core_pb), 0.0, 1.0)
        predicted_b = p_b > 0.5
        actual_is_b = actual >= 0.5

        directionality = 1.0 if predicted_b == actual_is_b else -1.0
        residual = actual - p_b
        residual_alignment = float(np.clip(1.0 - 2.0 * abs(residual), -1.0, 1.0))

        turbulence_break = (
            self.last_alignment is not None
            and directionality != self.last_alignment
        )

        if turbulence_break:
            components = {
                "directionality": 0.0,
                "residual_alignment": 0.0,
                "persistence": 0.0,
            }
        else:
            persistence = directionality if self.last_alignment == directionality else 0.0
            components = {
                "directionality": directionality,
                "residual_alignment": residual_alignment,
                "persistence": persistence,
            }

        return components, directionality, turbulence_break

    def update_likelihood(
        self,
        measurements: dict[str, float],
        *,
        turbulence_break: bool = False,
    ) -> float:
        """Apply the three-dimensional Gaussian likelihood to particle weights."""
        weights_cfg = PF_CONFIG["likelihood_weights"]
        variance = max(self.r, 1e-12)

        # A structural break discards accumulated weight concentration before
        # applying the neutral likelihood, allowing rapid return toward zero.
        if turbulence_break:
            self.weights.fill(1.0 / self.n_particles)

        weighted_error = np.zeros(self.n_particles, dtype=np.float64)
        for name in ("directionality", "residual_alignment", "persistence"):
            measurement = float(np.clip(measurements[name], -1.0, 1.0))
            diff = measurement - self.particles
            weighted_error += float(weights_cfg[name]) * (diff * diff)

        log_likelihood = -0.5 * weighted_error / variance
        log_likelihood -= float(np.max(log_likelihood))
        self.weights *= np.exp(log_likelihood)

        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        ess_before_resample = self.effective_sample_size()
        self.last_ess = ess_before_resample
        self.last_resampled = False
        if ess_before_resample < self.resample_threshold:
            self.systematic_resample()
            self.last_resampled = True

        w = PF_CONFIG["likelihood_weights"]
        self.last_observation = float(np.clip(
            w["directionality"] * measurements["directionality"]
            + w["residual_alignment"] * measurements["residual_alignment"]
            + w["persistence"] * measurements["persistence"],
            -1.0,
            1.0,
        ))
        self.last_measurements = dict(measurements)
        return self.estimate()

    def predict(self, *, current_round: float) -> float:
        q_eff = self.effective_q(current_round)
        std = math.sqrt(max(q_eff, 1e-12))
        noise = np.asarray(
            [self.rng.normal() * std for _ in range(self.n_particles)],
            dtype=np.float64,
        )
        self.particles = np.clip(
            self.particles + noise,
            -self.state_clip,
            self.state_clip,
        )
        self.last_effective_q = q_eff
        return self.estimate()

    def observe_and_project(
        self,
        *,
        actual_b: float,
        core_pb: float,
        current_round: float,
        estimated_total_hands: float | None = None,
    ) -> float:
        """Update from the settled round, then project the state for next round."""
        del estimated_total_hands  # kept only for API compatibility with the fixed 7D pipeline
        measurements, alignment, turbulence_break = self.measurement_components(
            actual_b=actual_b,
            core_pb=core_pb,
        )
        self.update_likelihood(
            measurements,
            turbulence_break=turbulence_break,
        )
        self.last_alignment = alignment
        self.updates += 1
        return self.predict(current_round=float(current_round) + 1.0)


def new_shoe_regime_filter() -> ShoeRegimeParticleFilter:
    return ShoeRegimeParticleFilter(
        n_particles=PF_CONFIG["n_particles"],
        q_early=PF_CONFIG["Q_early"],
        q_late=PF_CONFIG["Q_late"],
        early_round_end=PF_CONFIG["early_round_end"],
        late_round_start=PF_CONFIG["late_round_start"],
        r=PF_CONFIG["R"],
        resample_threshold=PF_CONFIG["resample_threshold"],
        random_state=PF_CONFIG["random_state"],
        state_clip=PF_CONFIG["state_clip"],
    )
