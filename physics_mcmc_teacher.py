#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速 MCMC / SMC Physics Teacher
================================

目的：
    將 B/P/T 歷史轉成「當前 8 副牌隱藏牌鞋的後驗分佈」，再從該後驗
    做下一局 posterior predictive，輸出與 production Physics Student
    完全一致的 48D teacher label。

為什麼要做成「串流 Session」：
    舊作法若對每個 history 都從第 0 局重新跑 MCMC，成本近似 O(T^2)。
    本模組每觀察一局只呼叫 observe() 一次，posterior 直接往前推，
    因此整靴成本近似 O(T)，可把 CI / GPT 修改與驗證控制在正常時間內。

重要物理限制：
    B/P/T 本身不能唯一還原真實未開牌。
    我們估計的是 P(S_t | B/P/T_1:t)，不是宣稱知道真正剩餘牌。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from physics_feature_extractor import (
    Card,
    PHYSICS_DIM,
    deal_baccarat_hand,
    new_eight_deck_shoe,
    normalize_history,
)


@dataclass
class Particle:
    """一個可能的完整 8 副牌牌鞋狀態。"""

    shoe: list[Card]
    cursor: int
    weight: float


@dataclass(frozen=True)
class TeacherSummary:
    """Physics Teacher 對『下一局』的 48D posterior predictive。"""

    physics_48d: np.ndarray
    physics_p_b: float
    physics_p_p: float
    physics_p_t: float
    ess: float
    n_particles: int


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """低變異 systematic resampling。"""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    out = np.empty(n, dtype=np.int32)
    i = j = 0
    while i < n:
        while j < n - 1 and positions[i] > cdf[j]:
            j += 1
        out[i] = j
        i += 1
    return out


def _swap_unseen_tail(p: Particle, rng: np.random.Generator, swaps: int) -> None:
    """
    MCMC rejuvenation kernel。

    只交換 cursor 之後尚未發出的牌：
      - 不改變已經用來解釋歷史 B/P/T 的 consumed prefix。
      - 不改變 unseen multiset，只改變 unseen ordering。
      - proposal 對稱，因此對「已觀測歷史」的接受率可視為 1。
    """
    start = int(p.cursor)
    if len(p.shoe) - start < 2:
        return
    for _ in range(max(0, int(swaps))):
        a = int(rng.integers(start, len(p.shoe)))
        b = int(rng.integers(start, len(p.shoe)))
        p.shoe[a], p.shoe[b] = p.shoe[b], p.shoe[a]


def _normalized_weights(particles: Sequence[Particle]) -> np.ndarray:
    w = np.asarray([max(0.0, float(p.weight)) for p in particles], dtype=np.float64)
    s = float(w.sum())
    if not math.isfinite(s) or s <= 1e-18:
        return np.full(len(particles), 1.0 / max(1, len(particles)), dtype=np.float64)
    return w / s


class MCMCPhysicsTeacherSession:
    """
    可逐局推進的 posterior session。

    使用方式：
        session = teacher.start_session()
        summary = session.summarize_next_hand()   # 目前 history 的下一局物理先驗
        session.observe("B")                      # 真實下一局開 B 後更新 posterior
        summary = session.summarize_next_hand()
    """

    def __init__(
        self,
        *,
        n_particles: int = 64,
        outcome_epsilon: float = 0.025,
        resample_ess_ratio: float = 0.55,
        rejuvenation_swaps: int = 2,
        random_state: int = 20260923,
    ):
        self.n_particles = int(max(16, n_particles))
        self.outcome_epsilon = float(min(0.25, max(1e-6, outcome_epsilon)))
        self.resample_ess_ratio = float(min(0.95, max(0.10, resample_ess_ratio)))
        self.rejuvenation_swaps = int(max(0, rejuvenation_swaps))
        self.rng = np.random.default_rng(int(random_state))
        w = 1.0 / self.n_particles
        self.particles = [
            Particle(new_eight_deck_shoe(self.rng), 0, w)
            for _ in range(self.n_particles)
        ]
        self.history: list[str] = []

    def observe(self, outcome: str) -> float:
        """
        用一個新的 B/P/T outcome 更新 latent-shoe posterior。

        每個 particle 先依自己隱藏牌鞋真正發一局，再比較模擬 outcome
        與 observed outcome 是否一致，作為 importance likelihood。
        """
        observed = str(outcome or "").strip().upper()
        if observed not in {"B", "P", "T"}:
            raise ValueError("outcome must be B/P/T")

        likelihood = np.empty(self.n_particles, dtype=np.float64)

        for i, p in enumerate(self.particles):
            if len(p.shoe) - p.cursor < 7:
                likelihood[i] = self.outcome_epsilon
                continue

            hand, next_cursor = deal_baccarat_hand(p.shoe, p.cursor)
            p.cursor = next_cursor
            likelihood[i] = 1.0 if hand.outcome == observed else self.outcome_epsilon

        prior = _normalized_weights(self.particles)
        post = prior * likelihood
        s = float(post.sum())
        if not math.isfinite(s) or s <= 1e-18:
            post[:] = 1.0 / self.n_particles
        else:
            post /= s

        for p, w in zip(self.particles, post):
            p.weight = float(w)

        ess = 1.0 / float(np.sum(post ** 2))

        # ESS 太低代表粒子退化；重採樣後再對 unseen tail 做 MCMC rejuvenation。
        if ess < self.resample_ess_ratio * self.n_particles:
            idx = _systematic_resample(post, self.rng)
            old = self.particles
            self.particles = [
                Particle(list(old[int(j)].shoe), int(old[int(j)].cursor), 1.0 / self.n_particles)
                for j in idx
            ]
            for p in self.particles:
                _swap_unseen_tail(p, self.rng, self.rejuvenation_swaps)

            ess = float(self.n_particles)

        self.history.append(observed)
        return float(ess)

    def observe_many(self, history: str | Iterable[str]) -> None:
        """相容工具：一次把一段 B/P/T 歷史串流進 posterior。"""
        for token in normalize_history(history):
            self.observe(token)

    def _weighted_remaining_rank_ratio(self) -> np.ndarray:
        """
        後驗加權的 A-K 剩餘比例。

        這裡不是『真實剩餘張數』，而是多個可能牌鞋狀態的 posterior mean。
        """
        weights = _normalized_weights(self.particles)
        counts = np.zeros(13, dtype=np.float64)
        total = 0.0

        for p, w in zip(self.particles, weights):
            rem = p.shoe[p.cursor:]
            if not rem:
                continue
            for card in rem:
                counts[card.rank - 1] += w
            total += w * len(rem)

        if total <= 1e-12:
            return np.full(13, 1.0 / 13.0, dtype=np.float64)
        return counts / total

    def summarize_next_hand(self) -> TeacherSummary:
        """
        由目前 posterior 直接做下一局 posterior predictive。

        為速度考量：
          - 每 particle 只需 deal 1 次。
          - deal_baccarat_hand 不會修改 shoe，因此不需要複製 416 張牌。
          - 64 particles 已可提供穩定 teacher target；大量精度需求可離線調大。
        """
        weights = _normalized_weights(self.particles)
        acc = np.zeros(PHYSICS_DIM, dtype=np.float64)

        for p, w in zip(self.particles, weights):
            if len(p.shoe) - p.cursor < 7:
                continue

            before = int(p.cursor)
            hand, _ = deal_baccarat_hand(p.shoe, before)

            row = np.zeros(PHYSICS_DIM, dtype=np.float64)
            k = 0

            # 3D：下一局 4/5/6 張牌概率
            row[k + {4: 0, 5: 1, 6: 2}[hand.card_count]] = 1.0
            k += 3

            # 10D + 10D：Player / Banker 最終點數分佈
            row[k + hand.player_point] = 1.0
            k += 10
            row[k + hand.banker_point] = 1.0
            k += 10

            # 3D：純物理 B/P/T posterior predictive probability
            row[k + {"B": 0, "P": 1, "T": 2}[hand.outcome]] = 1.0
            k += 3

            # 13D 先保留位置；稍後用 posterior remaining rank ratio × expected cards
            # 轉成「下一手預期消耗 A-K 張數」。
            k += 13

            # 4D：下一手實際抽牌花色比例，最後對所有 particles 加權平均。
            suits = np.zeros(4, dtype=np.float64)
            for c in hand.cards:
                suits[c.suit] += 1.0
            row[k:k+4] = suits / max(1.0, float(hand.card_count))
            k += 4

            # 3D：已消耗張數 + 低牌/高牌剩餘密度
            rem = p.shoe[before:]
            z = max(1.0, float(len(rem)))
            row[k] = float(before)
            row[k+1] = sum(c.rank <= 5 for c in rem) / z
            row[k+2] = sum(c.rank >= 9 for c in rem) / z
            k += 3

            # 2D：點差期望 / 絕對點差期望
            diff = hand.banker_point - hand.player_point
            row[k] = diff / 9.0
            row[k+1] = abs(diff) / 9.0

            acc += row * float(w)

        # 把 A-K posterior remaining composition 映射成下一局預期抽牌張數。
        rank_ratio = self._weighted_remaining_rank_ratio()
        expected_cards = float(np.dot(acc[:3], np.asarray([4.0, 5.0, 6.0])))
        acc[26:39] = rank_ratio * expected_cards

        weights = _normalized_weights(self.particles)
        ess = 1.0 / float(np.sum(weights ** 2))

        return TeacherSummary(
            physics_48d=acc.astype(np.float32),
            physics_p_b=float(acc[23]),
            physics_p_p=float(acc[24]),
            physics_p_t=float(acc[25]),
            ess=float(ess),
            n_particles=self.n_particles,
        )


class MCMCPhysicsTeacher:
    """Factory + 相容 API。"""

    def __init__(
        self,
        *,
        n_particles: int = 64,
        outcome_epsilon: float = 0.025,
        resample_ess_ratio: float = 0.55,
        rejuvenation_swaps: int = 2,
        random_state: int = 20260923,
    ):
        self.n_particles = int(n_particles)
        self.outcome_epsilon = float(outcome_epsilon)
        self.resample_ess_ratio = float(resample_ess_ratio)
        self.rejuvenation_swaps = int(rejuvenation_swaps)
        self.random_state = int(random_state)

    def start_session(self, *, seed_offset: int = 0) -> MCMCPhysicsTeacherSession:
        return MCMCPhysicsTeacherSession(
            n_particles=self.n_particles,
            outcome_epsilon=self.outcome_epsilon,
            resample_ess_ratio=self.resample_ess_ratio,
            rejuvenation_swaps=self.rejuvenation_swaps,
            random_state=self.random_state + int(seed_offset) * 1000003,
        )

    def summarize_history(self, history: str | Sequence[str]) -> TeacherSummary:
        s = self.start_session(seed_offset=len(normalize_history(history)))
        s.observe_many(history)
        return s.summarize_next_hand()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="BPPBTBBP")
    ap.add_argument("--particles", type=int, default=64)
    a = ap.parse_args()

    teacher = MCMCPhysicsTeacher(n_particles=a.particles)
    summary = teacher.summarize_history(a.history)
    print(json.dumps({
        "physics_p_b": summary.physics_p_b,
        "physics_p_p": summary.physics_p_p,
        "physics_p_t": summary.physics_p_t,
        "ess": summary.ess,
        "n_particles": summary.n_particles,
        "physics_dim": int(summary.physics_48d.size),
    }, ensure_ascii=False, indent=2))
