#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B/P/T -> latent shoe posterior particle filter + MCMC rejuvenation.

重要限制：
僅有 B/P/T 無法唯一還原真實殘牌。本模組估計的是：
    P(remaining_shoe | observed B/P/T history)
而不是宣稱「知道真實未開牌」。

用途：
1. 離線產生物理後驗 supervision。
2. 驗證 Physics surrogate/MLP 的物理一致性。
3. 提供下一局 Physics P(B)、4/5/6 張牌概率、點數與牌堆失衡摘要。

線上 browser 不建議直接跑大量粒子；production 仍可使用離線訓練好的輕量 surrogate。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from physics_feature_extractor import (
    Card,
    DECKS,
    PHYSICS_DIM,
    deal_baccarat_hand,
    new_eight_deck_shoe,
    normalize_history,
)

OUTCOME_EPSILON = 0.025


@dataclass
class Particle:
    shoe: list[Card]
    cursor: int = 0
    weight: float = 1.0


@dataclass(frozen=True)
class PosteriorSummary:
    physics_48d: np.ndarray
    physics_p_b: float
    ess: float
    n_particles: int


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """系統重採樣；比 multinomial resampling 變異更低。"""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    out = np.empty(n, dtype=np.int32)
    i = 0
    j = 0
    while i < n:
        while j < n - 1 and positions[i] > cdf[j]:
            j += 1
        out[i] = j
        i += 1
    return out


def _rejuvenate_unseen_tail(p: Particle, rng: np.random.Generator, swaps: int = 2) -> None:
    """
    MCMC rejuvenation：
    只交換「尚未發出的 unseen tail」，不碰已消耗牌，因此不破壞已觀測歷史。
    這是一個保持 unseen-card multiset 不變的對稱 proposal。
    """
    start = p.cursor
    n = len(p.shoe) - start
    if n < 2:
        return
    for _ in range(max(0, swaps)):
        a = int(rng.integers(start, len(p.shoe)))
        b = int(rng.integers(start, len(p.shoe)))
        p.shoe[a], p.shoe[b] = p.shoe[b], p.shoe[a]


def _remaining_rank_ratios(particles: Sequence[Particle]) -> np.ndarray:
    counts = np.zeros(13, dtype=np.float64)
    total = 0.0
    for p in particles:
        for c in p.shoe[p.cursor:]:
            counts[c.rank - 1] += 1.0
            total += 1.0
    if total <= 0:
        return np.full(13, 1 / 13, dtype=np.float32)
    return (counts / total).astype(np.float32)


class PhysicsPosteriorExtractor:
    """
    Sequential Monte Carlo (particle filter) + MCMC rejuvenation.

    每個觀測 B/P/T 都會：
      1. 對每個 latent shoe 發一局；
      2. 依是否符合 observed outcome 給 likelihood；
      3. 正規化；
      4. ESS 太低時重採樣；
      5. 對 unseen tail 做 MCMC swap rejuvenation。
    """

    def __init__(
        self,
        *,
        n_particles: int = 256,
        outcome_epsilon: float = OUTCOME_EPSILON,
        resample_ess_ratio: float = 0.55,
        rejuvenation_swaps: int = 2,
        random_state: int = 20260923,
    ):
        self.n_particles = int(max(32, n_particles))
        self.outcome_epsilon = float(min(0.25, max(1e-6, outcome_epsilon)))
        self.resample_ess_ratio = float(min(0.95, max(0.1, resample_ess_ratio)))
        self.rejuvenation_swaps = int(max(0, rejuvenation_swaps))
        self.random_state = int(random_state)

    def _initial_particles(self, rng: np.random.Generator) -> list[Particle]:
        w = 1.0 / self.n_particles
        return [Particle(new_eight_deck_shoe(rng), 0, w) for _ in range(self.n_particles)]

    def infer_particles(self, history: str | Sequence[str]) -> tuple[list[Particle], float]:
        seq = normalize_history(history)
        rng = np.random.default_rng(self.random_state + len(seq) * 7919)
        particles = self._initial_particles(rng)

        for observed in seq:
            likelihoods = np.empty(len(particles), dtype=np.float64)

            for i, p in enumerate(particles):
                # 若剩餘牌不足，給極低 likelihood，避免 crash。
                if len(p.shoe) - p.cursor < 7:
                    likelihoods[i] = self.outcome_epsilon
                    continue
                hand, new_cursor = deal_baccarat_hand(p.shoe, p.cursor)
                p.cursor = new_cursor
                likelihoods[i] = 1.0 if hand.outcome == observed else self.outcome_epsilon

            prior = np.asarray([p.weight for p in particles], dtype=np.float64)
            weights = prior * likelihoods
            s = float(weights.sum())
            if not math.isfinite(s) or s <= 1e-18:
                weights[:] = 1.0 / len(weights)
            else:
                weights /= s

            for p, w in zip(particles, weights):
                p.weight = float(w)

            ess = 1.0 / float(np.sum(weights ** 2))
            if ess < self.resample_ess_ratio * len(particles):
                idx = _systematic_resample(weights, rng)
                particles = [
                    Particle(list(particles[int(j)].shoe), particles[int(j)].cursor, 1.0 / len(idx))
                    for j in idx
                ]
                for p in particles:
                    _rejuvenate_unseen_tail(p, rng, self.rejuvenation_swaps)

        weights = np.asarray([p.weight for p in particles], dtype=np.float64)
        weights /= max(1e-18, float(weights.sum()))
        ess = 1.0 / float(np.sum(weights ** 2))
        return particles, ess

    def summarize_next_hand(
        self,
        history: str | Sequence[str],
        *,
        rollouts_per_particle: int = 2,
    ) -> PosteriorSummary:
        particles, ess = self.infer_particles(history)
        rng = np.random.default_rng(self.random_state + 104729 + len(normalize_history(history)))

        # 48D schema 與既有 PhysicsFeatureExtractor 完全一致。
        acc = np.zeros(PHYSICS_DIM, dtype=np.float64)
        total_weight = 0.0

        for p in particles:
            for _ in range(max(1, int(rollouts_per_particle))):
                # unseen tail copy：不污染 posterior particle 本身
                clone = Particle(list(p.shoe), p.cursor, p.weight)
                _rejuvenate_unseen_tail(clone, rng, 1)
                if len(clone.shoe) - clone.cursor < 7:
                    continue

                before = clone.cursor
                hand, _ = deal_baccarat_hand(clone.shoe, clone.cursor)
                row = np.zeros(PHYSICS_DIM, dtype=np.float64)
                k = 0

                row[k + {4: 0, 5: 1, 6: 2}[hand.card_count]] = 1.0
                k += 3
                row[k + hand.player_point] = 1.0
                k += 10
                row[k + hand.banker_point] = 1.0
                k += 10
                row[k + {"B": 0, "P": 1, "T": 2}[hand.outcome]] = 1.0
                k += 3

                rank_counts = np.zeros(13, dtype=np.float64)
                suit_counts = np.zeros(4, dtype=np.float64)
                for c in hand.cards:
                    rank_counts[c.rank - 1] += 1.0
                    suit_counts[c.suit] += 1.0
                row[k:k+13] = rank_counts
                k += 13
                row[k:k+4] = suit_counts / max(1.0, float(hand.card_count))
                k += 4

                rem = clone.shoe[before:]
                z = max(1.0, float(len(rem)))
                row[k] = float(before)
                row[k+1] = sum(c.rank <= 5 for c in rem) / z
                row[k+2] = sum(c.rank >= 9 for c in rem) / z
                k += 3

                diff = hand.banker_point - hand.player_point
                row[k] = diff / 9.0
                row[k+1] = abs(diff) / 9.0

                w = p.weight / max(1, int(rollouts_per_particle))
                acc += row * w
                total_weight += w

        if total_weight <= 0:
            raise RuntimeError("posterior rollout produced no valid samples")
        acc /= total_weight

        # 將 posterior 的 rank composition 訊息輕量注入 13D expected rank block：
        # 保留「下一手預期消耗張數」語意，不直接把剩餘張數塞進既有 schema。
        rank_ratio = _remaining_rank_ratios(particles)
        expected_cards = float(np.dot(acc[:3], np.array([4.0, 5.0, 6.0])))
        acc[26:39] = rank_ratio * expected_cards

        physics_pb = float(acc[23])
        return PosteriorSummary(
            physics_48d=acc.astype(np.float32),
            physics_p_b=physics_pb,
            ess=float(ess),
            n_particles=len(particles),
        )


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="BPPBTBBP")
    ap.add_argument("--particles", type=int, default=256)
    a = ap.parse_args()

    m = PhysicsPosteriorExtractor(n_particles=a.particles)
    s = m.summarize_next_hand(a.history)
    print(json.dumps({
        "physics_p_b": s.physics_p_b,
        "ess": s.ess,
        "n_particles": s.n_particles,
        "physics_dim": int(s.physics_48d.size),
    }, ensure_ascii=False, indent=2))
