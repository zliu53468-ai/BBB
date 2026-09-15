#!/usr/bin/env python3
"""BBB 8-deck (416-card) LightGBM + XGBoost residual ensemble.

第一層是既有 256D/V23 Frozen Base，本檔完全不修改其權重與推導邏輯。
第二層的 LightGBM 與 XGBoost 只學習同一個殘差目標：

    y_residual = actual_B(1/0) - P_core

正式決策：

    delta_lgbm = LightGBM(features)
    delta_xgb  = XGBoost(features)
    delta_raw  = (delta_lgbm + delta_xgb) / 2
    delta      = clip(delta_raw, -0.10, +0.10)
    final_p_B  = clip(P_core + delta, 0.0, 1.0)
    direction  = "B" if final_p_B > 0.50 else "P"

不存在 PASS。

物理牌靴特徵以 8 副牌 / 416 張為固定基準。若呼叫端能提供實際
remaining_cards，會直接使用；若只有 B/P/T 歷史，則使用明確標記為
estimated 的 deterministic fallback，不會把估算值冒充成實際剩餘牌數。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

TOTAL_CARDS = 416.0
MAX_HANDS = 70.0
DEFAULT_MAX_DELTA = 0.10
DEFAULT_RANDOM_STATE = 20260916
DEFAULT_ESTIMATED_CARDS_PER_HAND = 4.90

FEATURE_NAMES: tuple[str, ...] = (
    "core_confidence",
    "current_hand",
    "avg_cards_per_hand",
    "remaining_cards_ratio",
    "shoe_progress_delta",
    "sx_markov_p_same",
    "stage",
    "depth",
)
MODEL_TYPE = "lgbm_xgb_residual_ensemble"
SCHEMA_VERSION = 1


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """裁切有限數值，避免 NaN/Inf 污染特徵。"""
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    """只保留合法 B/P/T 並維持原始時間順序。"""
    if history is None:
        return []
    values: Iterable[Any]
    if isinstance(history, str):
        values = [x for x in history.upper() if x in {"B", "P", "T"}]
    else:
        values = history
    out: list[str] = []
    for item in values:
        token = str(item or "").upper().strip()
        if token in {"B", "P", "T"}:
            out.append(token)
    return out


def transition_sequence(history: Sequence[str]) -> list[str]:
    """由非和局結果建立 SAME/SWITCH；T 不建立新的 S/X 狀態。"""
    bp = [x for x in history if x in {"B", "P"}]
    return ["S" if bp[i] == bp[i - 1] else "X" for i in range(1, len(bp))]


def sx_markov_p_same(
    history: Sequence[str],
    *,
    window: int = 24,
    prior: float = 1.0,
) -> float:
    """一階 S/X 局部馬可夫：估計下一個 transition token 為 SAME 的機率。"""
    tokens = transition_sequence(history)
    if not tokens:
        return 0.5
    current = tokens[-1]
    start = max(0, len(tokens) - 1 - max(2, int(window)))
    same = switch = 0.0
    for i in range(start, len(tokens) - 1):
        if tokens[i] != current:
            continue
        if tokens[i + 1] == "S":
            same += 1.0
        elif tokens[i + 1] == "X":
            switch += 1.0
    return clip((same + prior) / (same + switch + 2.0 * prior))


def current_stage(history: Sequence[str]) -> int:
    """目前 B/P 同邊連續長度。"""
    bp = [x for x in history if x in {"B", "P"}]
    if not bp:
        return 0
    side = bp[-1]
    length = 1
    for i in range(len(bp) - 2, -1, -1):
        if bp[i] != side:
            break
        length += 1
    return length


def current_depth(history: Sequence[str]) -> int:
    """目前 S 或 X 的 suffix depth。"""
    tokens = transition_sequence(history)
    if not tokens:
        return 0
    state = tokens[-1]
    depth = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != state:
            break
        depth += 1
    return depth


def estimate_remaining_cards(
    current_hand: float,
    *,
    estimated_cards_per_hand: float = DEFAULT_ESTIMATED_CARDS_PER_HAND,
) -> float:
    """無實際剩餘牌數時的 deterministic fallback。"""
    hand = clip(float(current_hand), 1.0, MAX_HANDS)
    rate = clip(float(estimated_cards_per_hand), 4.0, 6.0)
    return clip(TOTAL_CARDS - hand * rate, 0.0, TOTAL_CARDS)


@dataclass(frozen=True)
class ResidualFeatures:
    """LightGBM 與 XGBoost 共用的固定特徵列。"""

    core_confidence: float
    current_hand: float
    avg_cards_per_hand: float
    remaining_cards_ratio: float
    shoe_progress_delta: float
    sx_markov_p_same: float
    stage: float
    depth: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.as_dict()], columns=list(FEATURE_NAMES), dtype=float)


@dataclass(frozen=True)
class PhysicalShoeContext:
    """用來稽核 416 張牌特徵來源，不作為額外模型特徵。"""

    remaining_cards: float
    source: str
    estimated_cards_per_hand: float


def build_features(
    *,
    core_confidence: float,
    history: str | Sequence[str],
    remaining_cards: float | None = None,
    estimated_cards_per_hand: float = DEFAULT_ESTIMATED_CARDS_PER_HAND,
    stage: float | None = None,
    depth: float | None = None,
) -> tuple[ResidualFeatures, PhysicalShoeContext]:
    """組裝 416 張物理牌靴特徵與牌路上下文。

    current_hand = 已觀察的 B/P/T 局數，限制 1..70。

    avg_cards_per_hand  = (416 - remaining_cards) / current_hand
    remaining_ratio     = remaining_cards / 416
    shoe_progress_delta = current_hand / 70
                          - (416 - remaining_cards) / 416
    """
    seq = normalize_history(history)
    current_hand = float(max(1, min(int(MAX_HANDS), len(seq))))

    if remaining_cards is None:
        cards_left = estimate_remaining_cards(
            current_hand,
            estimated_cards_per_hand=estimated_cards_per_hand,
        )
        source = "estimated"
    else:
        cards_left = clip(float(remaining_cards), 0.0, TOTAL_CARDS)
        source = "actual"

    consumed = TOTAL_CARDS - cards_left
    avg_cards = consumed / current_hand
    remaining_ratio = cards_left / TOTAL_CARDS
    shoe_progress_delta = (current_hand / MAX_HANDS) - (consumed / TOTAL_CARDS)

    features = ResidualFeatures(
        core_confidence=clip(float(core_confidence)),
        current_hand=current_hand,
        avg_cards_per_hand=float(avg_cards),
        remaining_cards_ratio=clip(remaining_ratio),
        shoe_progress_delta=float(shoe_progress_delta),
        sx_markov_p_same=sx_markov_p_same(seq),
        stage=float(current_stage(seq) if stage is None else stage),
        depth=float(current_depth(seq) if depth is None else depth),
    )
    physical = PhysicalShoeContext(
        remaining_cards=float(cards_left),
        source=source,
        estimated_cards_per_hand=float(estimated_cards_per_hand),
    )
    return features, physical


def build_lightgbm(*, random_state: int = DEFAULT_RANDOM_STATE) -> LGBMRegressor:
    """固定參數的 LightGBM residual regressor。"""
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        min_split_gain=0.0,
        reg_alpha=0.20,
        reg_lambda=8.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=int(random_state),
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        boost_from_average=False,
        verbosity=-1,
    )


def build_xgboost(*, random_state: int = DEFAULT_RANDOM_STATE) -> XGBRegressor:
    """固定參數的 XGBoost residual regressor。"""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=240,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=8.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.20,
        reg_lambda=8.0,
        gamma=0.0,
        random_state=int(random_state),
        n_jobs=1,
        tree_method="hist",
        base_score=0.0,
        verbosity=0,
    )


class DualResidualEnsemblePredictor:
    """Frozen Base 外掛的 LightGBM + XGBoost 50/50 殘差融合器。"""

    def __init__(
        self,
        lightgbm_model: LGBMRegressor | None = None,
        xgboost_model: XGBRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.lightgbm_model = lightgbm_model or build_lightgbm()
        self.xgboost_model = xgboost_model or build_xgboost()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def assemble_features(
        self,
        *,
        core_confidence: float,
        history: str | Sequence[str],
        remaining_cards: float | None = None,
        estimated_cards_per_hand: float = DEFAULT_ESTIMATED_CARDS_PER_HAND,
        stage: float | None = None,
        depth: float | None = None,
    ) -> tuple[pd.DataFrame, PhysicalShoeContext]:
        features, physical = build_features(
            core_confidence=core_confidence,
            history=history,
            remaining_cards=remaining_cards,
            estimated_cards_per_hand=estimated_cards_per_hand,
            stage=stage,
            depth=depth,
        )
        return features.as_frame(), physical

    def fit(
        self,
        feature_frame: pd.DataFrame,
        actual_b: Sequence[int],
    ) -> "DualResidualEnsemblePredictor":
        """兩個模型用完全相同的 residual target 訓練。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float).copy()
        y = np.asarray(actual_b, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError("feature_frame 與 actual_b 長度必須一致")
        if len(x) == 0:
            raise ValueError("至少需要一筆訓練資料")

        core = x["core_confidence"].to_numpy(dtype=float)
        residual = y - core
        self.lightgbm_model.fit(x, residual, feature_name=list(FEATURE_NAMES))
        self.xgboost_model.fit(x, residual)
        return self

    def predict_components(self, feature_frame: pd.DataFrame) -> dict[str, float]:
        """同時預測兩個 residual，50/50 平均後只裁切最終融合 Delta。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float)
        delta_lgbm = float(self.lightgbm_model.predict(x)[0])
        delta_xgb = float(self.xgboost_model.predict(x)[0])
        blended_raw = 0.5 * (delta_lgbm + delta_xgb)
        delta = max(-self.max_delta, min(self.max_delta, blended_raw))
        return {
            "lightgbm_delta_raw": delta_lgbm,
            "xgboost_delta_raw": delta_xgb,
            "blended_delta_raw": blended_raw,
            "delta": delta,
        }

    def predict_and_adjust(self, feature_frame: pd.DataFrame) -> dict[str, Any]:
        """執行雙模型殘差修正，100% 強制輸出 B 或 P。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float)
        core = clip(float(x.iloc[0]["core_confidence"]))
        parts = self.predict_components(x)
        final_p_b = clip(core + parts["delta"])
        return {
            "core_confidence": core,
            **parts,
            "final_p_b": final_p_b,
            "final_p_p": 1.0 - final_p_b,
            "direction": "B" if final_p_b > 0.50 else "P",
        }

    def predict_from_context(
        self,
        *,
        core_confidence: float,
        history: str | Sequence[str],
        remaining_cards: float | None = None,
        estimated_cards_per_hand: float = DEFAULT_ESTIMATED_CARDS_PER_HAND,
        stage: float | None = None,
        depth: float | None = None,
    ) -> dict[str, Any]:
        frame, physical = self.assemble_features(
            core_confidence=core_confidence,
            history=history,
            remaining_cards=remaining_cards,
            estimated_cards_per_hand=estimated_cards_per_hand,
            stage=stage,
            depth=depth,
        )
        result = self.predict_and_adjust(frame)
        result["features"] = frame.iloc[0].to_dict()
        result["physical_shoe"] = {
            "remaining_cards": physical.remaining_cards,
            "source": physical.source,
            "estimated_cards_per_hand": physical.estimated_cards_per_hand,
        }
        return result


def _parse_actual_b(record: Mapping[str, Any]) -> int:
    if record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").upper().strip()
    if actual == "B":
        return 1
    if actual == "P":
        return 0
    raise ValueError("training row must contain actual_outcome B/P or actual_b 0/1")


def _record_to_features(record: Mapping[str, Any]) -> dict[str, float]:
    if all(name in record for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}

    history = record.get("history") or record.get("history_fingerprint") or ""
    core = float(record.get("core_confidence", record.get("core_p_b", record.get("core_pb", 0.5))))
    remaining = record.get("remaining_cards")
    features, _ = build_features(
        core_confidence=core,
        history=history,
        remaining_cards=(float(remaining) if remaining is not None else None),
        estimated_cards_per_hand=float(record.get("estimated_cards_per_hand", DEFAULT_ESTIMATED_CARDS_PER_HAND)),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    )
    return features.as_dict()


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON training file must contain a list or {'rows': [...]}")
        return [dict(row) for row in rows]
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("supported training formats: .json, .jsonl, .csv")


def records_to_training(records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, np.ndarray]:
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    for record in records:
        try:
            actual_b = _parse_actual_b(record)
            feature_rows.append(_record_to_features(record))
            labels.append(actual_b)
        except (TypeError, ValueError):
            continue
    if not feature_rows:
        raise ValueError("no valid B/P labeled training rows")
    frame = pd.DataFrame(feature_rows, columns=list(FEATURE_NAMES), dtype=float)
    return frame, np.asarray(labels, dtype=np.int8)


def export_bundle(
    predictor: DualResidualEnsemblePredictor,
    output_path: Path,
    *,
    rows: int,
) -> dict[str, Any]:
    """匯出兩組樹，讓 GitHub Pages runtime 可直接重現推論。"""
    lgbm_dump = predictor.lightgbm_model.booster_.dump_model()
    xgb_booster = predictor.xgboost_model.get_booster()
    xgb_trees = [json.loads(tree) for tree in xgb_booster.get_dump(dump_format="json")]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "max_delta": predictor.max_delta,
        "physical_shoe": {
            "total_cards": int(TOTAL_CARDS),
            "max_hands": int(MAX_HANDS),
            "default_estimated_cards_per_hand": DEFAULT_ESTIMATED_CARDS_PER_HAND,
            "remaining_cards_policy": "actual_if_supplied_else_deterministic_estimate",
        },
        "blend": {
            "lightgbm_weight": 0.5,
            "xgboost_weight": 0.5,
            "clip_min": -predictor.max_delta,
            "clip_max": predictor.max_delta,
        },
        "lightgbm": {"trained": True, "base_score": 0.0, "trees": lgbm_dump.get("tree_info", [])},
        "xgboost": {"trained": True, "base_score": 0.0, "trees": xgb_trees},
        "training": {
            "rows": int(rows),
            "target": "actual_B_minus_core_confidence",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "final_probability": "clip(core_confidence + delta, 0, 1)",
            "no_pass": True,
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return bundle


def empty_bundle() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": False,
        "feature_names": list(FEATURE_NAMES),
        "max_delta": DEFAULT_MAX_DELTA,
        "physical_shoe": {
            "total_cards": int(TOTAL_CARDS),
            "max_hands": int(MAX_HANDS),
            "default_estimated_cards_per_hand": DEFAULT_ESTIMATED_CARDS_PER_HAND,
            "remaining_cards_policy": "actual_if_supplied_else_deterministic_estimate",
        },
        "blend": {"lightgbm_weight": 0.5, "xgboost_weight": 0.5, "clip_min": -DEFAULT_MAX_DELTA, "clip_max": DEFAULT_MAX_DELTA},
        "lightgbm": {"trained": False, "base_score": 0.0, "trees": []},
        "xgboost": {"trained": False, "base_score": 0.0, "trees": []},
        "training": {
            "rows": 0,
            "status": "awaiting_labeled_B_P_data",
            "target": "actual_B_minus_core_confidence",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "final_probability": "clip(core_confidence + delta, 0, 1)",
            "no_pass": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BBB LightGBM + XGBoost residual ensemble")
    parser.add_argument("--input", type=Path, help="JSON/JSONL/CSV labeled training data")
    parser.add_argument("--output", type=Path, default=Path("ensemble_residual_model.json"))
    parser.add_argument("--write-empty", action="store_true", help="write an untrained browser bundle")
    args = parser.parse_args()

    if args.write_empty:
        args.output.write_text(json.dumps(empty_bundle(), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        return
    if args.input is None:
        parser.error("--input is required unless --write-empty is used")

    records = load_records(args.input)
    frame, labels = records_to_training(records)
    predictor = DualResidualEnsemblePredictor()
    predictor.fit(frame, labels)
    export_bundle(predictor, args.output, rows=len(frame))


if __name__ == "__main__":
    main()
