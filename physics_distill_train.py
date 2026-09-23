#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC Physics Teacher -> Physics MLP Student 蒸餾訓練
===================================================

這支檔案是全新 inverted architecture 的離線第一階段：

    8-deck 真實模擬產生 B/P/T 歷史
        ↓
    Streaming MCMC / SMC Teacher
        ↓
    Teacher Physics 48D
        ↓
    PhysicsFeatureExtractor (MLP Student)
        ↓
    physics_multitask_model.joblib / .json

設計重點：
1. Teacher 在預測某一局時，只能看到該局之前的 B/P/T，避免 label leakage。
2. 每一靴只建立一個 posterior session，逐局 observe，避免 O(T^2) 重算。
3. train/validation 依 shoe_id 切分，避免同一靴前後局同時出現在 train/valid。
4. production browser 仍只跑輕量 MLP Student，不直接跑 MCMC。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics_feature_extractor import (
    PHYSICS_DIM,
    PhysicsFeatureExtractor,
    deal_baccarat_hand,
    history_to_vector,
    new_eight_deck_shoe,
    sanitize_physics_prediction,
)
from physics_mcmc_teacher import MCMCPhysicsTeacher


@dataclass
class TeacherDataset:
    x: np.ndarray
    y: np.ndarray
    shoe_ids: np.ndarray
    histories: list[str]


def build_teacher_dataset(
    *,
    n_shoes: int,
    cut_cards: int = 60,
    n_particles: int = 64,
    sample_stride: int = 2,
    warmup_hands: int = 2,
    max_hands_per_shoe: int = 90,
    random_state: int = 20260923,
) -> TeacherDataset:
    """
    產生無洩漏的 MCMC Teacher dataset。

    真實 simulator 只負責產生『觀測到的 B/P/T』；
    teacher target 完全由 posterior particles 自己推演，不偷看真實未開牌。
    """
    rng = np.random.default_rng(int(random_state))
    teacher = MCMCPhysicsTeacher(
        n_particles=n_particles,
        random_state=random_state + 17,
        rejuvenation_swaps=2,
    )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[int] = []
    histories: list[str] = []

    for shoe_id in range(int(n_shoes)):
        real_shoe = new_eight_deck_shoe(rng)
        real_cursor = 0
        history: list[str] = []
        session = teacher.start_session(seed_offset=shoe_id + 1)

        for hand_index in range(int(max_hands_per_shoe)):
            if len(real_shoe) - real_cursor <= int(cut_cards) + 6:
                break

            # 先用目前『只包含過去局』的 posterior 產生下一局 teacher target。
            if hand_index >= int(warmup_hands) and hand_index % max(1, int(sample_stride)) == 0:
                summary = session.summarize_next_hand()
                xs.append(history_to_vector(history))
                ys.append(summary.physics_48d)
                ids.append(shoe_id)
                histories.append("".join(history))

            # 再揭露真實下一局 outcome，更新 teacher posterior。
            real_hand, real_cursor = deal_baccarat_hand(real_shoe, real_cursor)
            session.observe(real_hand.outcome)
            history.append(real_hand.outcome)

    if not xs:
        raise ValueError("teacher simulation produced no samples")

    return TeacherDataset(
        x=np.vstack(xs).astype(np.float32),
        y=np.vstack(ys).astype(np.float32),
        shoe_ids=np.asarray(ids, dtype=np.int32),
        histories=histories,
    )


def evaluate_student(
    model: PhysicsFeatureExtractor,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    raw_scaled = model.model.predict(model.scaler.transform(x))
    raw = model.target_scaler.inverse_transform(raw_scaled)
    pred = np.vstack([sanitize_physics_prediction(v) for v in raw])
    truth = np.asarray(y, dtype=np.float32)

    winner_pred = np.argmax(pred[:, 23:26], axis=1)
    winner_true = np.argmax(truth[:, 23:26], axis=1)

    return {
        "teacher_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "teacher_winner_argmax_accuracy": float(np.mean(winner_pred == winner_true)),
        "physics_pb_mae": float(np.mean(np.abs(pred[:, 23] - truth[:, 23]))),
        "physics_pp_mae": float(np.mean(np.abs(pred[:, 24] - truth[:, 24]))),
        "physics_pt_mae": float(np.mean(np.abs(pred[:, 25] - truth[:, 25]))),
    }


def train_distilled_student(
    *,
    n_shoes: int,
    cut_cards: int,
    n_particles: int,
    sample_stride: int,
    validation_fraction: float,
    random_state: int,
) -> tuple[PhysicsFeatureExtractor, dict[str, float]]:
    data = build_teacher_dataset(
        n_shoes=n_shoes,
        cut_cards=cut_cards,
        n_particles=n_particles,
        sample_stride=sample_stride,
        random_state=random_state,
    )

    unique = np.unique(data.shoe_ids)
    split = max(1, min(len(unique) - 1, int(round(len(unique) * (1.0 - validation_fraction)))))
    train_ids = set(int(v) for v in unique[:split])
    train_mask = np.asarray([int(s) in train_ids for s in data.shoe_ids], dtype=bool)
    valid_mask = ~train_mask

    student = PhysicsFeatureExtractor(random_state=random_state)
    student.fit(data.x[train_mask], data.y[train_mask])
    metrics = evaluate_student(student, data.x[valid_mask], data.y[valid_mask])

    metrics.update({
        "teacher_rows": float(len(data.x)),
        "training_rows": float(train_mask.sum()),
        "validation_rows": float(valid_mask.sum()),
        "teacher_shoes": float(n_shoes),
        "teacher_particles": float(n_particles),
        "sample_stride": float(sample_stride),
    })

    student.metadata = {
        **metrics,
        "teacher_type": "streaming_mcmc_smc_posterior",
        "teacher_target": "posterior_predictive_physics_48d",
        "history_only_online_input": True,
        "no_future_card_leakage": True,
        "physics_dim": PHYSICS_DIM,
        "semantic_note": (
            "MLP distills P(hidden 8-deck shoe state | B/P/T history); "
            "it does not reconstruct the true unseen cards."
        ),
    }
    return student, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shoes", type=int, default=48)
    ap.add_argument("--cut-cards", type=int, default=60)
    ap.add_argument("--particles", type=int, default=64)
    ap.add_argument("--sample-stride", type=int, default=2)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=20260923)
    ap.add_argument("--output", default="physics_multitask_model.joblib")
    ap.add_argument("--browser-output", default="physics_multitask_model.json")
    ap.add_argument("--metrics-output", default="physics_distill_metrics.json")
    a = ap.parse_args()

    student, metrics = train_distilled_student(
        n_shoes=a.shoes,
        cut_cards=a.cut_cards,
        n_particles=a.particles,
        sample_stride=a.sample_stride,
        validation_fraction=a.validation_fraction,
        random_state=a.random_state,
    )

    student.save(a.output)
    student.export_browser_bundle(a.browser_output)
    Path(a.metrics_output).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "ok": True,
        "metrics": metrics,
        "model": a.output,
        "browser": a.browser_output,
        "metrics_file": a.metrics_output,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
