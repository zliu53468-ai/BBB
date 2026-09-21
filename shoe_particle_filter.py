#!/usr/bin/env python3
"""Blind physical Shoe Particle Filter with change-point anomaly brake.

The frozen upstream 256D/V23 Core and seven upstream features are untouched.
Each particle remains one plausible eight-deck shoe represented by baccarat
point-value counts 0..9.

The PF now exposes four downstream values:
    pred_card_count
    pred_banker_point
    pred_player_point
    anomaly_score

anomaly_score is a hidden change-point signal in [0, 1]. A long directional run
that suddenly breaks raises it above 0.9. Alternating / unstable outcomes keep
it elevated, while stable continuation decays it quickly toward zero.

During an anomaly, the PF posterior is deliberately de-concentrated toward a
uniform particle distribution and the likelihood is tempered. This prevents
the latent shoe state from mechanically carrying the previous run too far into
a structural break.
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
    "anomaly_score",
)

PF_CONFIG: dict[str, Any] = {
    "n_particles": 1000,
    "decks": 8,
    "point_bins": 10,
    "initial_point_counts": INITIAL_POINT_COUNTS.tolist(),
    "Q_early": 0.005,
    "Q_late": 0.03,
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
    "anomaly": {
        "long_run_min": 3,
        "long_run_break_floor": 0.95,
        "short_run_break": 0.72,
        "alternation_floor": 0.72,
        "core_miss_base": 0.30,
        "core_miss_confidence_gain": 0.35,
        "stable_decay": 0.55,
        "posterior_collapse_max_mix": 0.92,
        "likelihood_precision_floor": 0.15,
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


class DeterministicRNG:
    """Cross-runtime LCG shared by Python replay and browser inference."""

    def __init__(self, seed: int = 42) -> None:
        self.state = int(seed) & _UINT32_MASK

    def uniform(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & _UINT32_MASK
        return (self.state + 0.5) / _UINT32_SCALE


class ShoeParticleFilter:
    """1000-particle blind shoe estimator with internal change-point brake."""

    def __init__(
        self,
        *,
        n_particles: int = 1000,
        q_early: float = 0.005,
        q_late: float = 0.03,
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
        self.weights = np.full(
            self.n_particles,
            1.0 / self.n_particles,
            dtype=np.float64,
        )
        self.updates = 0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_core_alignment: float | None = None
        self.last_turbulence_break = False
        self.recent_outcomes: list[int] = []
        self.anomaly_score = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset a new shoe to the standard 416-card eight-deck distribution."""
        self.rng = DeterministicRNG(self.random_state)
        self.particles = np.tile(INITIAL_POINT_COUNTS, (self.n_particles, 1))
        self.weights = np.full(
            self.n_particles,
            1.0 / self.n_particles,
            dtype=np.float64,
        )
        self.updates = 0
        self.last_effective_q = self.q_early
        self.last_ess = float(self.n_particles)
        self.last_resampled = False
        self.last_core_alignment = None
        self.last_turbulence_break = False
        self.recent_outcomes = []
        self.anomaly_score = 0.0

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
        ratio = float(
            np.clip(
                (current - self.early_round_end) / span,
                0.0,
                1.0,
            )
        )
        return self.q_early + (self.q_late - self.q_early) * ratio

    def current_anomaly_score(self) -> float:
        return float(np.clip(self.anomaly_score, 0.0, 1.0))

    @staticmethod
    def _run_length(values: list[int]) -> int:
        if not values:
            return 0
        side = values[-1]
        run = 1
        for value in reversed(values[:-1]):
            if value != side:
                break
            run += 1
        return run

    @staticmethod
    def _alternation_strength(values: list[int]) -> float:
        tail = values[-4:]
        if len(tail) < 3:
            return 0.0
        switches = sum(
            1
            for left, right in zip(tail, tail[1:])
            if left != right
        )
        return switches / max(1, len(tail) - 1)

    def _detect_anomaly(
        self,
        *,
        actual_b: float,
        core_pb: float,
    ) -> float:
        cfg = PF_CONFIG["anomaly"]
        actual_sign = 1 if float(actual_b) >= 0.5 else -1
        history = self.recent_outcomes
        prior_run = self._run_length(history)
        structural_break = 0.0

        if history and actual_sign != history[-1]:
            if prior_run >= int(cfg["long_run_min"]):
                run_excess = min(
                    1.0,
                    max(
                        0.0,
                        (prior_run - int(cfg["long_run_min"])) / 4.0,
                    ),
                )
                structural_break = min(
                    1.0,
                    float(cfg["long_run_break_floor"])
                    + 0.05 * run_excess,
                )
            elif prior_run >= 2:
                structural_break = float(cfg["short_run_break"])

        candidate_tail = [*history[-3:], actual_sign]
        alternation = self._alternation_strength(candidate_tail)
        alternation_score = (
            float(cfg["alternation_floor"]) * alternation
            if alternation >= (2.0 / 3.0)
            else 0.0
        )

        core_direction = 1 if clip(core_pb, 0.0, 1.0) > 0.5 else -1
        core_confidence = abs(2.0 * clip(core_pb, 0.0, 1.0) - 1.0)
        core_miss = (
            float(cfg["core_miss_base"])
            + float(cfg["core_miss_confidence_gain"]) * core_confidence
            if core_direction != actual_sign
            else 0.0
        )

        decayed_previous = (
            self.current_anomaly_score()
            * float(cfg["stable_decay"])
        )
        return float(
            np.clip(
                max(
                    structural_break,
                    alternation_score,
                    core_miss,
                    decayed_previous,
                ),
                0.0,
                1.0,
            )
        )

    def _apply_anomaly_collapse(self, anomaly_score: float) -> None:
        cfg = PF_CONFIG["anomaly"]
        mix = min(
            float(cfg["posterior_collapse_max_mix"]),
            float(cfg["posterior_collapse_max_mix"])
            * float(anomaly_score) ** 2,
        )
        if mix <= 0.0:
            return
        uniform = 1.0 / self.n_particles
        self.weights = (
            (1.0 - mix) * self.weights
            + mix * uniform
        )
        total = float(np.sum(self.weights))
        if total > 0.0 and np.isfinite(total):
            self.weights /= total
        else:
            self.weights.fill(uniform)

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
        natural = (
            player_total in (8, 9)
            or banker_total in (8, 9)
        )

        player_third: int | None = None
        if not natural:
            if player_total <= 5:
                player_third = self._draw_point(counts)
                player.append(player_third)
                player_total = sum(player) % 10

            if self._banker_draws(banker_total, player_third):
                banker.append(self._draw_point(counts))
                banker_total = sum(banker) % 10

        sign = (
            1
            if banker_total > player_total
            else -1
            if player_total > banker_total
            else 0
        )
        return SimulatedRound(
            sign=sign,
            player_total=player_total,
            banker_total=banker_total,
            player_cards=len(player),
            banker_cards=len(banker),
        )

    def _core_alignment(
        self,
        *,
        actual_b: float,
        core_pb: float,
    ) -> float:
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
        anomaly_score: float,
    ) -> float:
        actual = 1.0 if float(actual_b) >= 0.5 else 0.0
        observed_sign = 1.0 if actual >= 0.5 else -1.0
        weights = PF_CONFIG["likelihood_weights"]

        weighted_error = (
            float(weights["outcome"])
            * ((observed_sign - simulated.sign) / 2.0) ** 2
        )
        active_weight = float(weights["outcome"])

        if observed_total_cards in (4, 5, 6):
            card_error = (
                float(observed_total_cards)
                - simulated.total_cards
            ) / 2.0
            weighted_error += (
                float(weights["total_cards"])
                * card_error
                * card_error
            )
            active_weight += float(weights["total_cards"])

        if (
            observed_player_point is not None
            and observed_banker_point is not None
            and 0 <= observed_player_point <= 9
            and 0 <= observed_banker_point <= 9
        ):
            player_error = (
                float(observed_player_point)
                - simulated.player_total
            ) / 9.0
            banker_error = (
                float(observed_banker_point)
                - simulated.banker_total
            ) / 9.0
            point_error = 0.5 * (
                player_error * player_error
                + banker_error * banker_error
            )
            weighted_error += (
                float(weights["points"])
                * point_error
            )
            active_weight += float(weights["points"])

        core_residual = actual - clip(
            float(core_pb),
            0.0,
            1.0,
        )
        core_target = float(
            np.clip(
                2.0 * core_residual,
                -1.0,
                1.0,
            )
        )
        if simulated.sign == 0:
            simulated_support = 0.0
        else:
            margin = abs(
                simulated.banker_total
                - simulated.player_total
            ) / 9.0
            simulated_support = (
                float(simulated.sign)
                * (0.5 + 0.5 * margin)
            )

        core_error = core_target - simulated_support
        weighted_error += (
            float(weights["core_residual"])
            * core_error
            * core_error
        )
        active_weight += float(weights["core_residual"])

        normalized_error = (
            weighted_error
            / max(active_weight, 1e-12)
        )
        anomaly_cfg = PF_CONFIG["anomaly"]
        precision = max(
            float(anomaly_cfg["likelihood_precision_floor"]),
            1.0 - 0.85 * float(anomaly_score) ** 2,
        )
        return (
            -0.5
            * float(persistence_multiplier)
            * precision
            * normalized_error
            / max(self.r, 1e-12)
        )

    def systematic_resample(self) -> None:
        positions = (
            self.rng.uniform()
            + np.arange(self.n_particles)
        ) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indexes = np.searchsorted(
            cumulative,
            positions,
            side="left",
        )
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.n_particles)

    def _rejuvenate(self, q_eff: float) -> None:
        """Inject hidden-card uncertainty while preserving total cards."""
        n_mutations = int(
            math.ceil(
                self.n_particles
                * max(0.0, min(1.0, q_eff))
            )
        )
        for _ in range(n_mutations):
            idx = min(
                self.n_particles - 1,
                int(
                    self.rng.uniform()
                    * self.n_particles
                ),
            )
            counts = self.particles[idx]
            sources = np.flatnonzero(counts > 0)
            destinations = np.flatnonzero(
                counts < INITIAL_POINT_COUNTS
            )
            if len(sources) == 0 or len(destinations) == 0:
                continue
            src = int(
                sources[
                    min(
                        len(sources) - 1,
                        int(
                            self.rng.uniform()
                            * len(sources)
                        ),
                    )
                ]
            )
            valid_destinations = destinations[
                destinations != src
            ]
            if len(valid_destinations) == 0:
                continue
            dst = int(
                valid_destinations[
                    min(
                        len(valid_destinations) - 1,
                        int(
                            self.rng.uniform()
                            * len(valid_destinations)
                        ),
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
        """Return side-effect-free [cards, banker, player, anomaly]."""
        del current_round
        saved_rng_state = self.rng.state

        weighted_card_count = 0.0
        weighted_banker_point = 0.0
        weighted_player_point = 0.0
        total_weight = 0.0

        p_b = clip(float(core_pb), 0.0, 1.0)
        expected_sign = 2.0 * p_b - 1.0
        core_confidence = abs(expected_sign)
        strength = float(
            PF_CONFIG["core_forecast_strength"]
        )

        try:
            for i in range(self.n_particles):
                counts = self.particles[i].copy()
                simulated = self._simulate_round(counts)
                particle_weight = float(self.weights[i])

                alignment_error = (
                    float(simulated.sign)
                    - expected_sign
                )
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
                weighted_card_count += (
                    weight
                    * float(simulated.total_cards)
                )
                weighted_banker_point += (
                    weight
                    * float(simulated.banker_total)
                )
                weighted_player_point += (
                    weight
                    * float(simulated.player_total)
                )
        finally:
            self.rng.state = saved_rng_state

        if total_weight <= 1e-12:
            physical = [4.8, 4.5, 4.5]
        else:
            physical = [
                float(
                    np.clip(
                        weighted_card_count / total_weight,
                        4.0,
                        6.0,
                    )
                ),
                float(
                    np.clip(
                        weighted_banker_point / total_weight,
                        0.0,
                        9.0,
                    )
                ),
                float(
                    np.clip(
                        weighted_player_point / total_weight,
                        0.0,
                        9.0,
                    )
                ),
            ]

        return np.asarray(
            [
                *physical,
                self.current_anomaly_score(),
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
        """Update posterior and anomaly state after a settled B/P round."""
        actual_b = (
            1.0
            if float(real_outcome) >= 0.5
            else 0.0
        )
        alignment = self._core_alignment(
            actual_b=actual_b,
            core_pb=core_pb,
        )
        anomaly_score = self._detect_anomaly(
            actual_b=actual_b,
            core_pb=core_pb,
        )
        self.anomaly_score = anomaly_score
        self._apply_anomaly_collapse(anomaly_score)

        turbulence_break = (
            self.last_core_alignment is not None
            and alignment != self.last_core_alignment
        )
        anomaly_cfg = PF_CONFIG["anomaly"]
        persistence_multiplier = (
            1.0
            + (
                float(PF_CONFIG["persistence_boost"]) - 1.0
            )
            * (1.0 - anomaly_score)
            if self.last_core_alignment == alignment
            else 1.0
        )

        log_weights = np.empty(
            self.n_particles,
            dtype=np.float64,
        )
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
                anomaly_score=anomaly_score,
            )

        log_weights -= float(np.max(log_weights))
        self.weights *= np.exp(log_weights)
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(
                1.0 / self.n_particles
            )
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
        actual_sign = 1 if actual_b >= 0.5 else -1
        self.recent_outcomes.append(actual_sign)
        self.recent_outcomes = self.recent_outcomes[-8:]
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
