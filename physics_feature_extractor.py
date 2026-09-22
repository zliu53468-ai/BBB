#!/usr/bin/env python3
"""Non-invasive baccarat physics feature plug-in.

This module intentionally does not touch the existing 256D/V23 core. It learns
conditional expectations of physical hand/shoe attributes from B/P/T history
using an offline 8-deck simulator, then concatenates those expectations after the
existing fixed 7D residual features.

Important limitation:
B/P/T history does not identify the actual remaining ranks or suits in a live
shoe. The model therefore estimates conditional distributions/expectations; it
must never be interpreted as reconstructing real unseen cards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from xgboost import XGBRegressor

DECKS = 8
CARDS_PER_DECK = 52
TOTAL_CARDS = DECKS * CARDS_PER_DECK
RANKS = tuple(range(1, 14))
RANK_LABELS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("spades", "hearts", "diamonds", "clubs")
OUTCOMES = ("B", "P", "T")
HISTORY_WINDOW = 64
HISTORY_SUMMARY_DIM = 21
HISTORY_INPUT_DIM = HISTORY_WINDOW * len(OUTCOMES) + HISTORY_SUMMARY_DIM

PHYSICS_FEATURE_NAMES: tuple[str, ...] = (
    ("cards_p4", "cards_p5", "cards_p6")
    + tuple(f"player_point_p{i}" for i in range(10))
    + tuple(f"banker_point_p{i}" for i in range(10))
    + ("winner_p_b", "winner_p_p", "winner_p_t")
    + tuple(f"next_rank_expected_{label}" for label in RANK_LABELS)
    + tuple(f"next_suit_ratio_{suit}" for suit in SUITS)
    + ("shoe_consumed_cards_norm", "remaining_low_rank_density", "remaining_high_rank_density")
    + ("expected_point_diff_norm", "expected_abs_point_diff_norm")
)
PHYSICS_DIM = len(PHYSICS_FEATURE_NAMES)
assert PHYSICS_DIM == 48

DEFAULT_MODEL_PATH = "physics_multitask_model.ubj"
DEFAULT_RANDOM_STATE = 20260922


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        values = [ch for ch in history.upper() if ch in OUTCOMES]
    else:
        values = history
    out: list[str] = []
    for item in values:
        token = str(item or "").strip().upper()
        if token in OUTCOMES:
            out.append(token)
    return out


def _entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(3.0))


def _run_length(seq: Sequence[str]) -> int:
    bp = [x for x in seq if x in {"B", "P"}]
    if not bp:
        return 0
    side = bp[-1]
    n = 1
    for value in reversed(bp[:-1]):
        if value != side:
            break
        n += 1
    return n


def history_to_vector(history: str | Sequence[str], *, window: int = HISTORY_WINDOW) -> np.ndarray:
    """Encode B/P/T history only; no hidden card information is used."""
    seq = normalize_history(history)
    tail = seq[-window:]
    one_hot = np.zeros((window, 3), dtype=np.float32)
    offset = window - len(tail)
    token_index = {"B": 0, "P": 1, "T": 2}
    for i, token in enumerate(tail, start=offset):
        one_hot[i, token_index[token]] = 1.0

    def ratios(n: int) -> tuple[float, float, float]:
        block = seq[-n:] if n > 0 else seq
        if not block:
            return 0.0, 0.0, 0.0
        size = float(len(block))
        return (
            block.count("B") / size,
            block.count("P") / size,
            block.count("T") / size,
        )

    bp = [x for x in seq if x in {"B", "P"}]
    transitions = 0
    if len(bp) >= 2:
        transitions = sum(bp[i] != bp[i - 1] for i in range(1, len(bp)))
    turn_rate = transitions / max(1, len(bp) - 1)
    current_run = _run_length(seq)

    r8 = ratios(8)
    r16 = ratios(16)
    r32 = ratios(32)
    rall = ratios(max(1, len(seq)))
    summary = np.asarray(
        [
            min(len(seq), 90) / 90.0,
            min(len(bp), 90) / 90.0,
            min(current_run, 12) / 12.0,
            turn_rate,
            *r8,
            *r16,
            *r32,
            *rall,
            _entropy(r8),
            _entropy(r16),
            _entropy(r32),
            1.0 if bp and bp[-1] == "B" else 0.0,
            1.0 if bp and bp[-1] == "P" else 0.0,
        ],
        dtype=np.float32,
    )
    if summary.size != HISTORY_SUMMARY_DIM:
        raise RuntimeError(f"history summary mismatch: {summary.size}")
    vector = np.hstack([one_hot.reshape(-1), summary]).astype(np.float32)
    if vector.size != HISTORY_INPUT_DIM:
        raise RuntimeError(f"history input mismatch: {vector.size}")
    return vector


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    @property
    def baccarat_value(self) -> int:
        return self.rank if 1 <= self.rank <= 9 else 0


@dataclass(frozen=True)
class HandResult:
    outcome: str
    player_point: int
    banker_point: int
    cards: tuple[Card, ...]

    @property
    def card_count(self) -> int:
        return len(self.cards)


def new_eight_deck_shoe(rng: np.random.Generator) -> list[Card]:
    cards = [Card(rank, suit) for _ in range(DECKS) for suit in range(4) for rank in RANKS]
    order = rng.permutation(len(cards))
    return [cards[int(i)] for i in order]


def _total(cards: Sequence[Card]) -> int:
    return sum(card.baccarat_value for card in cards) % 10


def _banker_draws(banker_total: int, player_third_value: int | None) -> bool:
    if player_third_value is None:
        return banker_total <= 5
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return player_third_value != 8
    if banker_total == 4:
        return 2 <= player_third_value <= 7
    if banker_total == 5:
        return 4 <= player_third_value <= 7
    if banker_total == 6:
        return 6 <= player_third_value <= 7
    return False


def deal_baccarat_hand(shoe: list[Card], cursor: int) -> tuple[HandResult, int]:
    """Deal one hand with standard baccarat third-card rules."""
    if cursor + 6 > len(shoe):
        raise IndexError("not enough cards left to safely deal a baccarat hand")

    player = [shoe[cursor], shoe[cursor + 2]]
    banker = [shoe[cursor + 1], shoe[cursor + 3]]
    consumed = [shoe[cursor], shoe[cursor + 1], shoe[cursor + 2], shoe[cursor + 3]]
    cursor += 4

    player_total = _total(player)
    banker_total = _total(banker)

    if player_total not in {8, 9} and banker_total not in {8, 9}:
        player_third_value: int | None = None
        if player_total <= 5:
            card = shoe[cursor]
            cursor += 1
            player.append(card)
            consumed.append(card)
            player_third_value = card.baccarat_value
            player_total = _total(player)

        if _banker_draws(banker_total, player_third_value):
            card = shoe[cursor]
            cursor += 1
            banker.append(card)
            consumed.append(card)
            banker_total = _total(banker)

    if banker_total > player_total:
        outcome = "B"
    elif player_total > banker_total:
        outcome = "P"
    else:
        outcome = "T"

    return (
        HandResult(
            outcome=outcome,
            player_point=player_total,
            banker_point=banker_total,
            cards=tuple(consumed),
        ),
        cursor,
    )


def _remaining_rank_density(shoe: Sequence[Card], cursor: int) -> tuple[float, float]:
    remaining = shoe[cursor:]
    if not remaining:
        return 0.0, 0.0
    low = sum(card.rank <= 5 for card in remaining)
    high = sum(card.rank >= 9 for card in remaining)
    n = float(len(remaining))
    return low / n, high / n


def build_physics_target(hand: HandResult, shoe: Sequence[Card], cursor_before: int) -> np.ndarray:
    """Create the 48D supervised target for one simulated next hand."""
    y = np.zeros(PHYSICS_DIM, dtype=np.float32)
    k = 0

    count_index = {4: 0, 5: 1, 6: 2}[hand.card_count]
    y[k + count_index] = 1.0
    k += 3

    y[k + hand.player_point] = 1.0
    k += 10
    y[k + hand.banker_point] = 1.0
    k += 10

    winner_index = {"B": 0, "P": 1, "T": 2}[hand.outcome]
    y[k + winner_index] = 1.0
    k += 3

    rank_counts = np.zeros(13, dtype=np.float32)
    suit_counts = np.zeros(4, dtype=np.float32)
    for card in hand.cards:
        rank_counts[card.rank - 1] += 1.0
        suit_counts[card.suit] += 1.0
    y[k : k + 13] = rank_counts
    k += 13
    y[k : k + 4] = suit_counts / max(1.0, float(hand.card_count))
    k += 4

    low_density, high_density = _remaining_rank_density(shoe, cursor_before)
    y[k] = cursor_before / float(TOTAL_CARDS)
    y[k + 1] = low_density
    y[k + 2] = high_density
    k += 3

    diff = hand.banker_point - hand.player_point
    y[k] = diff / 9.0
    y[k + 1] = abs(diff) / 9.0
    k += 2

    if k != PHYSICS_DIM:
        raise RuntimeError(f"physics target mismatch: {k}")
    return y


@dataclass
class SimulationDataset:
    x: np.ndarray
    y: np.ndarray
    shoe_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[1] != HISTORY_INPUT_DIM:
            raise ValueError("invalid simulator x shape")
        if self.y.ndim != 2 or self.y.shape[1] != PHYSICS_DIM:
            raise ValueError("invalid simulator y shape")
        if len(self.x) != len(self.y) or len(self.x) != len(self.shoe_ids):
            raise ValueError("simulator arrays have inconsistent lengths")


class OfflineBaccaratSimulator:
    """Generate supervised rows from physically shuffled 8-deck shoes."""

    def __init__(
        self,
        *,
        cut_cards: int = 60,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_hands_per_shoe: int = 90,
    ) -> None:
        self.cut_cards = int(max(14, min(120, cut_cards)))
        self.random_state = int(random_state)
        self.max_hands_per_shoe = int(max(1, max_hands_per_shoe))

    def generate(self, n_shoes: int) -> SimulationDataset:
        rng = np.random.default_rng(self.random_state)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        shoe_ids: list[int] = []

        for shoe_id in range(int(n_shoes)):
            shoe = new_eight_deck_shoe(rng)
            cursor = 0
            history: list[str] = []

            for _ in range(self.max_hands_per_shoe):
                if len(shoe) - cursor <= self.cut_cards + 6:
                    break
                before = cursor
                hand, cursor = deal_baccarat_hand(shoe, cursor)
                xs.append(history_to_vector(history))
                ys.append(build_physics_target(hand, shoe, before))
                shoe_ids.append(shoe_id)
                history.append(hand.outcome)

        if not xs:
            raise ValueError("simulation produced no training rows")
        return SimulationDataset(
            x=np.vstack(xs).astype(np.float32),
            y=np.vstack(ys).astype(np.float32),
            shoe_ids=np.asarray(shoe_ids, dtype=np.int32),
        )


def _normalise_nonnegative(block: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(block, dtype=np.float64), 0.0, None)
    total = float(x.sum())
    if not math.isfinite(total) or total <= 1e-12:
        return fallback.astype(np.float32, copy=True)
    return (x / total).astype(np.float32)


def sanitize_physics_prediction(raw: Sequence[float]) -> np.ndarray:
    """Map raw multi-output regression values to valid probability/expectation ranges."""
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if x.size != PHYSICS_DIM:
        raise ValueError(f"expected {PHYSICS_DIM} physics outputs, got {x.size}")

    out = np.zeros(PHYSICS_DIM, dtype=np.float32)
    k = 0
    out[k : k + 3] = _normalise_nonnegative(x[k : k + 3], np.array([0.58, 0.34, 0.08]))
    k += 3
    out[k : k + 10] = _normalise_nonnegative(x[k : k + 10], np.full(10, 0.1))
    k += 10
    out[k : k + 10] = _normalise_nonnegative(x[k : k + 10], np.full(10, 0.1))
    k += 10
    out[k : k + 3] = _normalise_nonnegative(x[k : k + 3], np.array([0.4586, 0.4462, 0.0952]))
    k += 3

    out[k : k + 13] = np.clip(x[k : k + 13], 0.0, 6.0).astype(np.float32)
    k += 13
    out[k : k + 4] = _normalise_nonnegative(x[k : k + 4], np.full(4, 0.25))
    k += 4

    out[k : k + 3] = np.clip(x[k : k + 3], 0.0, 1.0).astype(np.float32)
    k += 3
    out[k] = float(np.clip(x[k], -1.0, 1.0))
    out[k + 1] = float(np.clip(x[k + 1], 0.0, 1.0))
    return out


class PhysicsFeatureExtractor:
    """Offline-trained multi-task predictor exposed as a 48D plug-in feature extractor."""

    def __init__(
        self,
        model: XGBRegressor | None = None,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.random_state = int(random_state)
        self.model = model or XGBRegressor(
            objective="reg:squarederror",
            n_estimators=220,
            max_depth=5,
            learning_rate=0.045,
            min_child_weight=8.0,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=6.0,
            random_state=self.random_state,
            n_jobs=1,
            tree_method="hist",
            multi_strategy="one_output_per_tree",
            verbosity=0,
        )
        self.is_fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PhysicsFeatureExtractor":
        xx = np.asarray(x, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32)
        if xx.ndim != 2 or xx.shape[1] != HISTORY_INPUT_DIM:
            raise ValueError(f"x must be N x {HISTORY_INPUT_DIM}")
        if yy.ndim != 2 or yy.shape[1] != PHYSICS_DIM:
            raise ValueError(f"y must be N x {PHYSICS_DIM}")
        self.model.fit(xx, yy)
        self.is_fitted = True
        return self

    def train_from_simulation(
        self,
        *,
        n_shoes: int = 2000,
        cut_cards: int = 60,
        validation_fraction: float = 0.20,
    ) -> dict[str, float]:
        simulator = OfflineBaccaratSimulator(
            cut_cards=cut_cards,
            random_state=self.random_state,
        )
        data = simulator.generate(n_shoes)
        unique_shoes = np.unique(data.shoe_ids)
        split_at = max(1, int(round(len(unique_shoes) * (1.0 - validation_fraction))))
        train_shoes = set(int(x) for x in unique_shoes[:split_at])
        train_mask = np.asarray([int(s) in train_shoes for s in data.shoe_ids], dtype=bool)
        valid_mask = ~train_mask
        if not valid_mask.any():
            valid_mask[-min(100, len(valid_mask)) :] = True
            train_mask = ~valid_mask

        self.fit(data.x[train_mask], data.y[train_mask])
        pred = np.vstack([sanitize_physics_prediction(row) for row in self.model.predict(data.x[valid_mask])])
        truth = data.y[valid_mask]

        card_acc = float(np.mean(np.argmax(pred[:, :3], axis=1) == np.argmax(truth[:, :3], axis=1)))
        winner_slice = slice(23, 26)
        winner_acc = float(
            np.mean(np.argmax(pred[:, winner_slice], axis=1) == np.argmax(truth[:, winner_slice], axis=1))
        )
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        self.metadata = {
            "n_shoes": int(n_shoes),
            "training_rows": int(train_mask.sum()),
            "validation_rows": int(valid_mask.sum()),
            "cut_cards": int(cut_cards),
            "validation_rmse": rmse,
            "card_count_accuracy": card_acc,
            "winner_accuracy": winner_acc,
            "history_input_dim": HISTORY_INPUT_DIM,
            "physics_dim": PHYSICS_DIM,
            "feature_names": list(PHYSICS_FEATURE_NAMES),
            "semantic_note": "conditional expectations from B/P/T history; not actual unseen-card reconstruction",
        }
        return {
            "validation_rmse": rmse,
            "card_count_accuracy": card_acc,
            "winner_accuracy": winner_acc,
            "training_rows": float(train_mask.sum()),
            "validation_rows": float(valid_mask.sum()),
        }

    def predict_features(self, history_path: str | Sequence[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PhysicsFeatureExtractor is not fitted; train or load a physics model first")
        x = history_to_vector(history_path).reshape(1, -1)
        raw = np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1)
        return sanitize_physics_prediction(raw)

    def predict_dict(self, history_path: str | Sequence[str]) -> dict[str, float]:
        vector = self.predict_features(history_path)
        return {name: float(value) for name, value in zip(PHYSICS_FEATURE_NAMES, vector)}

    def save(self, model_path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted physics model")
        path = Path(model_path)
        self.model.save_model(path)
        meta_path = path.with_name(path.name + ".meta.json")
        payload = {
            "schema_version": 1,
            "model_type": "baccarat_physics_multitask_xgb",
            "history_input_dim": HISTORY_INPUT_DIM,
            "physics_dim": PHYSICS_DIM,
            "feature_names": list(PHYSICS_FEATURE_NAMES),
            "metadata": self.metadata,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path: str | Path) -> "PhysicsFeatureExtractor":
        path = Path(model_path)
        model = XGBRegressor()
        model.load_model(path)
        obj = cls(model=model)
        obj.is_fitted = True
        meta_path = path.with_name(path.name + ".meta.json")
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            names = tuple(payload.get("feature_names") or ())
            if names and names != PHYSICS_FEATURE_NAMES:
                raise ValueError("physics feature schema mismatch")
            obj.metadata = dict(payload.get("metadata") or {})
        return obj


_DEFAULT_EXTRACTOR: PhysicsFeatureExtractor | None = None


def get_default_extractor() -> PhysicsFeatureExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        model_path = os.environ.get("BGS_PHYSICS_MODEL_PATH", DEFAULT_MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"physics model not found: {model_path}. Train it offline first or set BGS_PHYSICS_MODEL_PATH."
            )
        _DEFAULT_EXTRACTOR = PhysicsFeatureExtractor.load(model_path)
    return _DEFAULT_EXTRACTOR


def prepare_xgboost_input(
    core_pb: float,
    original_7d: Sequence[float],
    history_path: str | Sequence[str],
    *,
    extractor: PhysicsFeatureExtractor | None = None,
) -> np.ndarray:
    """Concatenate Core P(B) + untouched original 7D + 48D physics features.

    The original 7D is copied value-for-value after numeric conversion; this
    function does not reorder, normalize, or overwrite those seven inputs.
    """
    original = np.asarray(original_7d, dtype=np.float32).reshape(-1)
    if original.size != 7:
        raise ValueError(f"original_7d must contain exactly 7 values, got {original.size}")
    if not np.all(np.isfinite(original)):
        raise ValueError("original_7d contains non-finite values")

    core = np.asarray([_clip(core_pb, 0.0, 1.0)], dtype=np.float32)
    physics = (extractor or get_default_extractor()).predict_features(history_path)
    merged = np.hstack([core, original, physics]).astype(np.float32)
    expected = 1 + 7 + PHYSICS_DIM
    if merged.size != expected:
        raise RuntimeError(f"extended feature mismatch: {merged.size} != {expected}")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBB non-invasive physics feature extractor")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="simulate 8-deck shoes and train the multi-task physics predictor")
    train.add_argument("--shoes", type=int, default=2000)
    train.add_argument("--cut-cards", type=int, default=60)
    train.add_argument("--output", default=DEFAULT_MODEL_PATH)
    train.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    simulate = sub.add_parser("simulate", help="generate a simulator dataset as compressed NPZ")
    simulate.add_argument("--shoes", type=int, default=100)
    simulate.add_argument("--cut-cards", type=int, default=60)
    simulate.add_argument("--output", default="physics_simulation_dataset.npz")
    simulate.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "simulate":
        simulator = OfflineBaccaratSimulator(cut_cards=args.cut_cards, random_state=args.random_state)
        data = simulator.generate(args.shoes)
        np.savez_compressed(args.output, x=data.x, y=data.y, shoe_ids=data.shoe_ids)
        print(json.dumps({"rows": len(data.x), "output": args.output}, ensure_ascii=False))
        return 0

    extractor = PhysicsFeatureExtractor(random_state=args.random_state)
    metrics = extractor.train_from_simulation(n_shoes=args.shoes, cut_cards=args.cut_cards)
    extractor.save(args.output)
    print(json.dumps({"metrics": metrics, "model": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
