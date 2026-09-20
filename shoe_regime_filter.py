#!/usr/bin/env python3
"""Shoe Regime Particle Filter for BBB downstream state estimation.

This module does not inspect card identities or remaining-card composition.
It estimates only the current shoe environment/regime used as the eighth
downstream XGBoost feature.

State semantics:
    +1.0  sustained Core-aligned regularity
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
    "Q_start": 0.005,
    "Q_end": 0.02,
    "R": 0.25,
    "resample_threshold": 500.0,
    "resampling": "systematic",
    "random_state": 42,
    "state_clip": 1.0,
    "observation_weights": {
        "direction_alignment": 0.55,
        "confidence_alignment": 0.30,
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
        q_start: float = 0.005,
        q_end: float = 0.02,
        r: float = 0.25,
        resample_threshold: float = 500.0,
        random_state: int = 42,
        state_clip: float = 1.0,
    ) -> None:
        self.n_particles = int(n_particles)
        self.q_start = float(q_start)
        self.q_end = float(q_end)
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
        self.last_effective_q = self.q_start
        self.reset()

    def reset(self) -> None:
        """Reset a new shoe to a zero-centered latent regime."""
        self.rng = DeterministicRNG(self.random_state)
        std = math.sqrt(max(self.q_start, 1e-12))
        draws = np.asarray([self.rng.normal() * std for _ in range(self.n_particles)], dtype=np.float64)
        draws -= float(np.mean(draws))
        self.particles = np.clip(draws, -self.state_clip, self.state_clip)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.updates = 0
        self.last_alignment = None
        self.last_observation = 0.0
        self.last_effective_q = self.q_start

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

    @staticmethod
    def shoe_progress(current_round: float, estimated_total_hands: float) -> float:
        total = max(2.0, float(estimated_total_hands))
        return float(np.clip((float(current_round) - 1.0) / (total - 1.0), 0.0, 1.0))

    def effective_q(self, current_round: float, estimated_total_hands: float) -> float:
        progress = self.shoe_progress(current_round, estimated_total_hands)
        return max(1e-12, self.q_start + (self.q_end - self.q_start) * progress)

    def make_observation(self, *, actual_b: float, core_pb: float) -> tuple[float, float]:
        actual_is_b = float(actual_b) >= 0.5
        p_b = clip(float(core_pb), 0.0, 1.0)
        core_is_b = p_b > 0.5
        alignment = 1.0 if core_is_b == actual_is_b else -1.0
        actual_probability = p_b if actual_is_b else (1.0 - p_b)
        confidence_alignment = float(np.clip(2.0 * (actual_probability - 0.5), -1.0, 1.0))
        weights = PF_CONFIG["observation_weights"]

        if alignment > 0.0:
            persistence = 1.0 if self.last_alignment == 1.0 else 0.0
            observation = (
                float(weights["direction_alignment"])
                + float(weights["confidence_alignment"]) * max(0.0, confidence_alignment)
                + float(weights["persistence"]) * persistence
            )
        elif self.last_alignment == -1.0:
            persistence = -1.0
            observation = -(
                float(weights["direction_alignment"])
                + float(weights["confidence_alignment"]) * abs(min(0.0, confidence_alignment))
                + float(weights["persistence"])
            )
        else:
            # One isolated break is turbulence, not an immediate reversal regime.
            observation = 0.0

        return float(np.clip(observation, -1.0, 1.0)), alignment

    def update_observation(self, observation: float) -> float:
        measurement = float(np.clip(observation, -1.0, 1.0))
        variance = max(self.r, 1e-12)
        error = measurement - self.particles
        log_likelihood = -0.5 * (error * error) / variance
        log_likelihood -= float(np.max(log_likelihood))
        self.weights *= np.exp(log_likelihood)
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        if self.effective_sample_size() < self.resample_threshold:
            self.systematic_resample()

        self.last_observation = measurement
        return self.estimate()

    def predict(self, *, current_round: float, estimated_total_hands: float) -> float:
        q_eff = self.effective_q(current_round, estimated_total_hands)
        std = math.sqrt(q_eff)
        noise = np.asarray([self.rng.normal() * std for _ in range(self.n_particles)], dtype=np.float64)
        self.particles = np.clip(self.particles + noise, -self.state_clip, self.state_clip)
        self.last_effective_q = q_eff
        return self.estimate()

    def observe_and_project(
        self,
        *,
        actual_b: float,
        core_pb: float,
        current_round: float,
        estimated_total_hands: float,
    ) -> float:
        observation, alignment = self.make_observation(actual_b=actual_b, core_pb=core_pb)
        self.update_observation(observation)
        self.last_alignment = alignment
        self.updates += 1
        return self.predict(
            current_round=float(current_round) + 1.0,
            estimated_total_hands=estimated_total_hands,
        )


def new_shoe_regime_filter() -> ShoeRegimeParticleFilter:
    return ShoeRegimeParticleFilter(
        n_particles=PF_CONFIG["n_particles"],
        q_start=PF_CONFIG["Q_start"],
        q_end=PF_CONFIG["Q_end"],
        r=PF_CONFIG["R"],
        resample_threshold=PF_CONFIG["resample_threshold"],
        random_state=PF_CONFIG["random_state"],
        state_clip=PF_CONFIG["state_clip"],
    )
