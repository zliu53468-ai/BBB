#!/usr/bin/env python3
"""BBB LightGBM 雙層殘差修正器。

第一層 256D/V23 Frozen Base 完全不在此檔修改；它只提供固定的 P_core。
第二層 LightGBM 只學習核心誤差：

    y_residual = actual_B(1/0) - P_core

正式決策：

    delta = clip(LightGBM(features), -0.10, +0.10)
    final_p_B = clip(P_core + delta, 0.0, 1.0)
    direction = "B" if final_p_B > 0.50 else "P"

沒有 PASS。相同模型、相同特徵、相同輸入會得到相同結果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


FEATURE_NAMES: tuple[str, ...] = (
    "core_p_b",
    "round_index",
    "estimated_total_hands",
    "remaining_ratio",
    "sx_markov_p_same",
    "stage",
    "depth",
)
MODEL_TYPE = "lgbm_residual_regressor"
SCHEMA_VERSION = 1
DEFAULT_MAX_DELTA = 0.10
DEFAULT_RANDOM_STATE = 20260915


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """有限數值裁切，避免 NaN/Inf 污染模型輸入。"""
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    """只保留合法 B/P/T；不改變原始時間順序。"""
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
    """建立 S/X 狀態鏈；T 不建立新的 SAME/SWITCH 狀態。"""
    bp = [x for x in history if x in {"B", "P"}]
    return ["S" if bp[i] == bp[i - 1] else "X" for i in range(1, len(bp))]


def sx_markov_p_same(history: Sequence[str], *, window: int = 24, prior: float = 1.0) -> float:
    """一階 S/X 局部馬可夫：估計目前狀態之後下一個 token 為 SAME 的機率。

    僅使用最近 window 個轉移並加 Laplace prior，避免小樣本直接落到 0/1。
    """
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
    """目前 S 或 X 的連續深度。"""
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


@dataclass(frozen=True)
class ResidualFeatures:
    """第二層 LightGBM 的固定特徵列。"""

    core_p_b: float
    round_index: float
    estimated_total_hands: float
    remaining_ratio: float
    sx_markov_p_same: float
    stage: float
    depth: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_frame(self) -> pd.DataFrame:
        """依固定欄位順序輸出一列 DataFrame。"""
        return pd.DataFrame([self.as_dict()], columns=list(FEATURE_NAMES), dtype=float)


def build_features(
    *,
    core_p_b: float,
    history: str | Sequence[str],
    estimated_total_hands: float = 60.0,
    stage: float | None = None,
    depth: float | None = None,
) -> ResidualFeatures:
    """由 Frozen Base 輸出與當前牌靴上下文組裝第二層特徵。"""
    seq = normalize_history(history)
    total_hands = clip(float(estimated_total_hands), 40.0, 90.0)
    round_index = float(max(1, min(70, len(seq) + 1)))
    remaining_ratio = clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    return ResidualFeatures(
        core_p_b=clip(float(core_p_b)),
        round_index=round_index,
        estimated_total_hands=total_hands,
        remaining_ratio=remaining_ratio,
        sx_markov_p_same=sx_markov_p_same(seq),
        stage=float(current_stage(seq) if stage is None else stage),
        depth=float(current_depth(seq) if depth is None else depth),
    )


def build_regressor(*, random_state: int = DEFAULT_RANDOM_STATE) -> LGBMRegressor:
    """固定且偏保守的 LightGBM residual regressor。

    deterministic + force_col_wise + n_jobs=1 用來減少同資料重訓時的非必要差異；
    boost_from_average=False 讓瀏覽器匯出推論更容易精確重現。
    """
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


class LightGBMResidualBiasPredictor:
    """Frozen Base 外掛的第二層 LightGBM 殘差修正器。"""

    def __init__(
        self,
        model: LGBMRegressor | None = None,
        *,
        max_delta: float = DEFAULT_MAX_DELTA,
    ) -> None:
        self.model = model or build_regressor()
        self.max_delta = clip(float(max_delta), 0.0, DEFAULT_MAX_DELTA)

    def assemble_features(
        self,
        *,
        core_p_b: float,
        history: str | Sequence[str],
        estimated_total_hands: float = 60.0,
        stage: float | None = None,
        depth: float | None = None,
    ) -> pd.DataFrame:
        """回傳 LightGBM 可直接使用的一列 DataFrame。"""
        return build_features(
            core_p_b=core_p_b,
            history=history,
            estimated_total_hands=estimated_total_hands,
            stage=stage,
            depth=depth,
        ).as_frame()

    def fit(self, feature_frame: pd.DataFrame, actual_b: Sequence[int]) -> "LightGBMResidualBiasPredictor":
        """以 actual_B - P_core 作為連續 residual target 訓練。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float).copy()
        y = np.asarray(actual_b, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError("feature_frame 與 actual_b 長度必須一致")
        core_pb = x["core_p_b"].to_numpy(dtype=float)
        residual = y - core_pb
        self.model.fit(x, residual, feature_name=list(FEATURE_NAMES))
        return self

    def predict_delta(self, feature_frame: pd.DataFrame) -> float:
        """預測並裁切第二層修正值至 +/-10%。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float)
        raw = float(self.model.predict(x)[0])
        return max(-self.max_delta, min(self.max_delta, raw))

    def correct(self, feature_frame: pd.DataFrame) -> dict[str, Any]:
        """套用第二層後強制輸出 B/P，不存在觀望狀態。"""
        x = feature_frame.loc[:, FEATURE_NAMES].astype(float)
        core_pb = clip(float(x.iloc[0]["core_p_b"]))
        delta = self.predict_delta(x)
        final_pb = clip(core_pb + delta)
        return {
            "core_p_b": core_pb,
            "delta": delta,
            "final_p_b": final_pb,
            "final_p_p": 1.0 - final_pb,
            "direction": "B" if final_pb > 0.50 else "P",
        }

    def predict_from_context(
        self,
        *,
        core_p_b: float,
        history: str | Sequence[str],
        estimated_total_hands: float = 60.0,
        stage: float | None = None,
        depth: float | None = None,
    ) -> dict[str, Any]:
        features = self.assemble_features(
            core_p_b=core_p_b,
            history=history,
            estimated_total_hands=estimated_total_hands,
            stage=stage,
            depth=depth,
        )
        result = self.correct(features)
        result["features"] = {k: float(v) for k, v in features.iloc[0].to_dict().items()}
        return result


ResidualBiasPredictor = LightGBMResidualBiasPredictor


def _parse_actual_b(record: Mapping[str, Any]) -> int:
    if record.get("actual_b") is not None:
        return 1 if float(record["actual_b"]) >= 0.5 else 0
    actual = str(record.get("actual_outcome") or record.get("actual") or "").upper().strip()
    if actual == "B":
        return 1
    if actual == "P":
        return 0
    raise ValueError("訓練列必須包含 actual_outcome=B/P 或 actual_b=0/1")


def _feature_row(record: Mapping[str, Any]) -> dict[str, float]:
    if all(record.get(name) is not None for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}
    features = build_features(
        core_p_b=float(record.get("core_p_b", 0.5)),
        history=record.get("history") or record.get("history_fingerprint") or "",
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    ).as_dict()
    for name in FEATURE_NAMES:
        if record.get(name) is not None:
            features[name] = float(record[name])
    return features


def load_training_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON 必須是 list 或 {'rows': [...]} 格式")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("訓練檔僅支援 .json / .csv")


def make_training_frame(records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows: list[dict[str, float]] = []
    actuals: list[int] = []
    shoes: list[str] = []
    for i, record in enumerate(records):
        try:
            row = _feature_row(record)
            actual = _parse_actual_b(record)
            if not all(math.isfinite(float(row[name])) for name in FEATURE_NAMES):
                continue
        except (TypeError, ValueError, KeyError):
            continue
        rows.append(row)
        actuals.append(actual)
        shoes.append(str(record.get("shoe_id") or f"row_{i}"))
    if not rows:
        raise ValueError("沒有有效的 B/P 訓練資料")
    return (
        pd.DataFrame(rows, columns=list(FEATURE_NAMES), dtype=float),
        np.asarray(actuals, dtype=np.int8),
        shoes,
    )


def deterministic_validation_mask(shoes: Sequence[str], fraction: float = 0.20) -> np.ndarray:
    threshold = int(256 * clip(fraction, 0.05, 0.50))
    mask = np.asarray(
        [hashlib.sha256(str(shoe).encode("utf-8")).digest()[0] < threshold for shoe in shoes],
        dtype=bool,
    )
    if mask.all() or (~mask).all():
        cut = max(1, int(round(len(mask) * (1.0 - fraction))))
        mask[:] = False
        mask[cut:] = True
    return mask


def direction_accuracy(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b > 0.5) == (actual_b > 0)))


def brier(prob_b: np.ndarray, actual_b: np.ndarray) -> float:
    return float(np.mean((prob_b.astype(float) - actual_b.astype(float)) ** 2))


def evaluate(model: LGBMRegressor, x: pd.DataFrame, actual_b: np.ndarray, max_delta: float) -> dict[str, float]:
    core = x["core_p_b"].to_numpy(dtype=float)
    raw_delta = np.asarray(model.predict(x.loc[:, FEATURE_NAMES]), dtype=float)
    delta = np.clip(raw_delta, -max_delta, max_delta)
    corrected = np.clip(core + delta, 0.0, 1.0)
    return {
        "samples": float(len(x)),
        "core_accuracy": direction_accuracy(core, actual_b),
        "corrected_accuracy": direction_accuracy(corrected, actual_b),
        "core_brier": brier(core, actual_b),
        "corrected_brier": brier(corrected, actual_b),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else 0.0,
    }


def _tree_leaf(node: Mapping[str, Any], vector: Sequence[float]) -> float:
    """Python 版可攜式 LightGBM 樹推論，用來驗證瀏覽器 JSON。"""
    current = node
    for _ in range(256):
        if "leaf_value" in current:
            return float(current.get("leaf_value", 0.0))
        idx = int(current.get("split_feature", -1))
        value = float(vector[idx]) if 0 <= idx < len(vector) else math.nan
        default_left = bool(current.get("default_left", True))
        if not math.isfinite(value):
            go_left = default_left
        else:
            threshold = float(current.get("threshold", 0.0))
            decision = str(current.get("decision_type", "<="))
            go_left = value <= threshold if "<=" in decision else value < threshold
        current = current.get("left_child" if go_left else "right_child") or {}
    return 0.0


def export_portable_bundle(
    model: LGBMRegressor,
    *,
    reference_x: pd.DataFrame,
    output_path: Path,
    max_delta: float,
    metrics: Mapping[str, Any],
    training_rows: int,
) -> dict[str, Any]:
    dump = model.booster_.dump_model()
    trees = dump.get("tree_info") or []

    # 用 native prediction - tree sum 推導固定 base_score，並逐列驗證匯出結果。
    first = reference_x.loc[:, FEATURE_NAMES].iloc[0].to_numpy(dtype=float)
    first_sum = sum(_tree_leaf(tree.get("tree_structure") or {}, first) for tree in trees)
    base_score = float(model.predict(reference_x.loc[:, FEATURE_NAMES].iloc[[0]])[0]) - first_sum

    for i in range(min(64, len(reference_x))):
        vector = reference_x.loc[:, FEATURE_NAMES].iloc[i].to_numpy(dtype=float)
        portable = base_score + sum(_tree_leaf(tree.get("tree_structure") or {}, vector) for tree in trees)
        native = float(model.predict(reference_x.loc[:, FEATURE_NAMES].iloc[[i]])[0])
        if abs(portable - native) > 1e-7:
            raise RuntimeError(f"LightGBM portable export mismatch: {portable} vs {native}")

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "trained": True,
        "feature_names": list(FEATURE_NAMES),
        "base_score": base_score,
        "max_delta": float(max_delta),
        "trees": trees,
        "training": {
            "rows": int(training_rows),
            "target": "actual_B_minus_core_p_B",
            "decision_rule": "B if final_p_B > 0.50 else P",
            "final_probability": "clip(core_p_B + delta, 0, 1)",
            "no_pass": True,
            "metrics": dict(metrics),
        },
    }
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return bundle


def train_command(args: argparse.Namespace) -> int:
    records = load_training_records(Path(args.input))
    x, actual_b, shoes = make_training_frame(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"至少需要 {args.min_samples} 筆有效資料，目前只有 {len(x)}")

    validation = deterministic_validation_mask(shoes, args.validation_fraction)
    train = ~validation

    model = build_regressor(random_state=args.random_state)
    train_core = x.loc[train, "core_p_b"].to_numpy(dtype=float)
    train_target = actual_b[train].astype(float) - train_core
    model.fit(x.loc[train, FEATURE_NAMES], train_target, feature_name=list(FEATURE_NAMES))

    metrics = evaluate(model, x.loc[validation, FEATURE_NAMES], actual_b[validation], args.max_delta)
    accepted = (
        metrics["corrected_brier"] <= metrics["core_brier"] + args.max_brier_regression
        and metrics["corrected_accuracy"] >= metrics["core_accuracy"] - args.max_accuracy_regression
    )
    print(json.dumps({"validation": metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    if not accepted and not args.force:
        raise SystemExit("validation gate rejected LightGBM residual model; --force 僅供診斷")

    final_model = build_regressor(random_state=args.random_state)
    full_target = actual_b.astype(float) - x["core_p_b"].to_numpy(dtype=float)
    final_model.fit(x.loc[:, FEATURE_NAMES], full_target, feature_name=list(FEATURE_NAMES))
    export_portable_bundle(
        final_model,
        reference_x=x,
        output_path=Path(args.output),
        max_delta=args.max_delta,
        metrics=metrics,
        training_rows=len(x),
    )
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB LightGBM second-layer residual trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="訓練 LGBMRegressor 並匯出瀏覽器可用 JSON")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="lightgbm_residual_model.json")
    train.add_argument("--min-samples", type=int, default=500)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--max-delta", type=float, default=DEFAULT_MAX_DELTA)
    train.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train.add_argument("--max-brier-regression", type=float, default=0.0)
    train.add_argument("--max-accuracy-regression", type=float, default=0.005)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
