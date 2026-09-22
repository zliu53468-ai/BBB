#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Probability Predictor

新 production target：
    y = actual_B in {0, 1}

不再學 residual，也不再做：
    final = core + delta

56D input：
    [1D Core P(B)] + [原固定 7D] + [Physics 48D] = 56D

Physics 48D 內已含 winner_p_b / winner_p_p / winner_p_t，
因此不額外重複加入 Physics_P(B)，避免變成 57D 與 duplicated feature。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from xgboost import XGBClassifier

from physics_feature_extractor import (
    PHYSICS_DIM,
    PHYSICS_FEATURE_NAMES,
    PhysicsFeatureExtractor,
    prepare_xgboost_input,
)
from xgb_residual_bias import (
    FEATURE_NAMES as ORIGINAL_7D_FEATURE_NAMES,
    build_features as build_original_7d,
    deterministic_validation_mask,
    load_training_records,
)

MODEL_TYPE = "xgb_global_probability_classifier"
RANDOM_STATE = 20260923

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    ("core_p_b_external",)
    + tuple(f"original7_{x}" for x in ORIGINAL_7D_FEATURE_NAMES)
    + PHYSICS_FEATURE_NAMES
)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
assert GLOBAL_DIM == 56


def clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    v = float(v)
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def logit(p: float) -> float:
    p = clip(p, 1e-7, 1 - 1e-7)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def prepare_inverted_xgboost_input(
    core_pb: float,
    original_7d: Sequence[float],
    history_path: str | Sequence[str],
    *,
    extractor: PhysicsFeatureExtractor | None = None,
) -> np.ndarray:
    """
    使用者指定的正式入口。

    注意：
    - Core P(B) 保留。
    - original_7d 值與順序完全保留。
    - Physics 48D 由獨立 PhysicsFeatureExtractor 產生。
    - 總維度固定 56D。
    """
    x = prepare_xgboost_input(core_pb, original_7d, history_path, extractor=extractor)
    if x.size != GLOBAL_DIM:
        raise RuntimeError(f"expected {GLOBAL_DIM} features, got {x.size}")
    return x.astype(np.float32)


def _actual_b(r: Mapping[str, Any]) -> int:
    if r.get("actual_b") is not None:
        return 1 if float(r["actual_b"]) >= 0.5 else 0
    a = str(r.get("actual_outcome") or r.get("actual") or "").upper()
    if a == "B":
        return 1
    if a == "P":
        return 0
    raise ValueError("directional B/P row required")


def _history(r: Mapping[str, Any]) -> str | Sequence[str]:
    return r.get("history") or r.get("history_fingerprint") or ""


def _core_pb(r: Mapping[str, Any]) -> float:
    value = r.get("core_p_b", r.get("core_pb"))
    if value is None:
        raise ValueError("missing core_p_b")
    return clip(float(value))


def _original_7d(r: Mapping[str, Any], core_pb: float) -> np.ndarray:
    if all(r.get(n) is not None for n in ORIGINAL_7D_FEATURE_NAMES):
        return np.asarray([float(r[n]) for n in ORIGINAL_7D_FEATURE_NAMES], dtype=np.float32)
    return build_original_7d(
        core_p_b=core_pb,
        history=_history(r),
        estimated_total_hands=float(r.get("estimated_total_hands", 60) or 60),
        stage=float(r["stage"]) if r.get("stage") is not None else None,
        depth=float(r["depth"]) if r.get("depth") is not None else None,
    ).as_vector().astype(np.float32)


def make_training_arrays(
    records: Sequence[Mapping[str, Any]],
    physics: PhysicsFeatureExtractor,
):
    xs: list[np.ndarray] = []
    ys: list[int] = []
    shoes: list[str] = []

    for i, r in enumerate(records):
        try:
            pb = _core_pb(r)
            y = _actual_b(r)
            x = prepare_inverted_xgboost_input(
                pb,
                _original_7d(r, pb),
                _history(r),
                extractor=physics,
            )
        except (TypeError, ValueError, KeyError):
            continue

        if x.size != GLOBAL_DIM or not np.all(np.isfinite(x)):
            continue

        xs.append(x)
        ys.append(y)
        shoes.append(str(r.get("shoe_id") or f"row_{i}"))

    if not xs:
        raise ValueError("no valid training rows")

    return (
        np.vstack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int8),
        shoes,
    )


def build_classifier(*, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """
    直接學 P(B)，不是 residual。
    參數刻意保守，降低 synthetic/offline data 上過擬合。
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=320,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.35,
        reg_lambda=12.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p.astype(float) - y.astype(float)) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    pp = np.clip(p.astype(float), 1e-7, 1 - 1e-7)
    yy = y.astype(float)
    return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))


def _acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p > 0.5) == (y > 0)))


def evaluate(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return {
        "samples": float(len(x)),
        "accuracy": _acc(p, y),
        "brier": _brier(p, y),
        "logloss": _logloss(p, y),
        "mean_p_b": float(np.mean(p)),
    }


def dynamic_clip_bounds(vector: Sequence[float]) -> tuple[float, float]:
    """
    動態 envelope：
    - 前期：牌堆資訊少，限制較窄。
    - 中後期：允許略寬。
    - Physics winner imbalance 明顯時，再小幅放寬。

    此處不是下注策略，只是概率輸出穩定器。
    """
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size != GLOBAL_DIM:
        raise ValueError("feature dim mismatch")

    # original7 的 remaining_ratio 位於 merged index 4：
    # [core external=0] + original7[remaining_ratio index=3] => 4
    remaining_ratio = clip(float(x[4]))
    physics_b = clip(float(x[8 + 23]))
    physics_p = clip(float(x[8 + 24]))
    imbalance = min(1.0, abs(physics_b - physics_p) / 0.20)

    width = 0.055 + 0.050 * (1.0 - remaining_ratio) + 0.015 * imbalance
    width = min(0.12, max(0.05, width))
    return 0.5 - width, 0.5 + width


def apply_dynamic_clip(raw_p_b: float, vector: Sequence[float]) -> float:
    lo, hi = dynamic_clip_bounds(vector)
    return clip(raw_p_b, lo, hi)


def _tree_leaf(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node = tree
    for _ in range(256):
        if "leaf" in node:
            return float(node.get("leaf", 0.0))
        split = str(node.get("split", ""))
        idx = int(split[1:]) if split.startswith("f") and split[1:].isdigit() else GLOBAL_FEATURE_NAMES.index(split)
        value = float(np.float32(vector[idx])) if 0 <= idx < len(vector) else math.nan
        threshold = float(np.float32(node.get("split_condition", 0.0)))
        nxt = node.get("missing") if not math.isfinite(value) else (node.get("yes") if value < threshold else node.get("no"))
        child = None
        for c in node.get("children") or []:
            if int(c.get("nodeid", -999)) == int(nxt):
                child = c
                break
        if child is None:
            return 0.0
        node = child
    return 0.0


def export_browser_bundle(
    model: XGBClassifier,
    reference_x: np.ndarray,
    path: str | Path,
    *,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """
    XGBoost binary:logistic 的 tree dump 是 margin contribution。
    Browser 端：
        margin = base_margin + sum(tree_leaf)
        p = sigmoid(margin)
    """
    trees = [json.loads(t) for t in model.get_booster().get_dump(dump_format="json")]
    ref = np.asarray(reference_x[0], dtype=float)
    tree_sum = sum(_tree_leaf(t, ref) for t in trees)
    native_p = float(model.predict_proba(ref.reshape(1, -1))[0, 1])
    base_margin = logit(native_p) - tree_sum

    # portable parity 檢查
    for vector in np.asarray(reference_x[: min(128, len(reference_x))], dtype=float):
        portable = sigmoid(base_margin + sum(_tree_leaf(t, vector) for t in trees))
        expected = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
        if abs(portable - expected) > 2e-5:
            raise RuntimeError(f"portable probability mismatch {portable} vs {expected}")

    bundle = {
        "schema_version": 3,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(GLOBAL_FEATURE_NAMES),
        "link": "sigmoid",
        "base_margin": float(base_margin),
        "trees": trees,
        "postprocess": {
            "type": "dynamic_symmetric_clip",
            "min_width": 0.05,
            "max_width": 0.12,
        },
        "training": {
            "target": "actual_B_absolute_probability",
            "residual": False,
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    Path(path).write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return bundle


class GlobalProbabilityPredictor:
    def __init__(self, model: XGBClassifier, physics: PhysicsFeatureExtractor):
        self.model = model
        self.physics = physics

    def predict(
        self,
        core_pb: float,
        original_7d: Sequence[float],
        history_path: str | Sequence[str],
    ) -> dict[str, Any]:
        x = prepare_inverted_xgboost_input(
            core_pb,
            original_7d,
            history_path,
            extractor=self.physics,
        )
        raw = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        final = apply_dynamic_clip(raw, x)
        return {
            "core_p_b": clip(core_pb),
            "global_raw_p_b": raw,
            "final_p_b": final,
            "direction": "B" if final > 0.5 else "P",
            "feature_dim": int(x.size),
        }


def train_command(a: argparse.Namespace) -> int:
    physics = PhysicsFeatureExtractor.load(a.physics_model)
    records = load_training_records(Path(a.input))
    x, y, shoes = make_training_arrays(records, physics)

    if len(x) < a.min_samples:
        raise SystemExit(f"need {a.min_samples} rows; got {len(x)}")

    valid = deterministic_validation_mask(shoes, fraction=a.validation_fraction)
    train = ~valid

    probe = build_classifier(random_state=a.random_state)
    probe.fit(x[train], y[train])
    metrics = evaluate(probe, x[valid], y[valid])
    print(json.dumps({"validation": metrics}, ensure_ascii=False, indent=2))

    final = build_classifier(random_state=a.random_state)
    final.fit(x, y)

    if a.joblib_output:
        joblib.dump(final, a.joblib_output)

    export_browser_bundle(final, x, a.output, metrics=metrics)
    print(f"wrote {a.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)

    t = s.add_parser("train")
    t.add_argument("--input", required=True)
    t.add_argument("--physics-model", required=True)
    t.add_argument("--output", default="global_probability_model.json")
    t.add_argument("--joblib-output", default="")
    t.add_argument("--min-samples", type=int, default=3000)
    t.add_argument("--validation-fraction", type=float, default=0.2)
    t.add_argument("--random-state", type=int, default=RANDOM_STATE)
    t.set_defaults(func=train_command)
    return p


def main() -> int:
    a = build_parser().parse_args()
    return int(a.func(a))


if __name__ == "__main__":
    raise SystemExit(main())
