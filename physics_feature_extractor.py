#!/usr/bin/env python3
"""Non-invasive 48D baccarat physics feature plug-in.

Production architecture:
    B/P/T history -> lightweight multi-task MLP -> 48D conditional expectations
    Core P(B) + untouched original 7D + 48D -> residual XGBoost

The 256D/V23 core is never modified.

Important:
B/P/T history cannot reconstruct actual unseen ranks or suits. The 48D output is
an offline-simulation-trained conditional expectation, not a real card counter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DECKS = 8
TOTAL_CARDS = 52 * DECKS
RANKS = tuple(range(1, 14))
RANK_LABELS = ("A","2","3","4","5","6","7","8","9","10","J","Q","K")
SUITS = ("spades","hearts","diamonds","clubs")
OUTCOMES = ("B","P","T")
HISTORY_WINDOW = 64
HISTORY_SUMMARY_DIM = 21
HISTORY_INPUT_DIM = HISTORY_WINDOW * 3 + HISTORY_SUMMARY_DIM

PHYSICS_FEATURE_NAMES: tuple[str, ...] = (
    ("cards_p4","cards_p5","cards_p6")
    + tuple(f"player_point_p{i}" for i in range(10))
    + tuple(f"banker_point_p{i}" for i in range(10))
    + ("winner_p_b","winner_p_p","winner_p_t")
    + tuple(f"next_rank_expected_{x}" for x in RANK_LABELS)
    + tuple(f"next_suit_ratio_{x}" for x in SUITS)
    + ("shoe_consumed_cards","remaining_low_rank_density","remaining_high_rank_density")
    + ("expected_point_diff_norm","expected_abs_point_diff_norm")
)
PHYSICS_DIM = len(PHYSICS_FEATURE_NAMES)
assert PHYSICS_DIM == 48

DEFAULT_MODEL_PATH = "physics_multitask_model.joblib"
DEFAULT_BROWSER_BUNDLE = "physics_multitask_model.json"
DEFAULT_RANDOM_STATE = 20260922


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    v = float(v)
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    values = [c for c in history.upper() if c in OUTCOMES] if isinstance(history, str) else history
    out: list[str] = []
    for item in values:
        token = str(item or "").strip().upper()
        if token in OUTCOMES:
            out.append(token)
    return out


def _entropy(p: Sequence[float]) -> float:
    a = np.asarray(p, dtype=np.float64)
    a = a[a > 0]
    return 0.0 if a.size <= 1 else float(-(a * np.log(a)).sum() / np.log(3.0))


def _run_length(seq: Sequence[str]) -> int:
    bp = [x for x in seq if x in {"B","P"}]
    if not bp:
        return 0
    side, n = bp[-1], 1
    for x in reversed(bp[:-1]):
        if x != side:
            break
        n += 1
    return n


def history_to_vector(history: str | Sequence[str], *, window: int = HISTORY_WINDOW) -> np.ndarray:
    seq = normalize_history(history)
    tail = seq[-window:]
    one_hot = np.zeros((window, 3), dtype=np.float32)
    idx = {"B":0,"P":1,"T":2}
    offset = window - len(tail)
    for i, token in enumerate(tail, start=offset):
        one_hot[i, idx[token]] = 1.0

    def ratios(n: int) -> tuple[float,float,float]:
        block = seq[-n:] if n else seq
        if not block:
            return 0.0,0.0,0.0
        z = float(len(block))
        return block.count("B")/z, block.count("P")/z, block.count("T")/z

    bp = [x for x in seq if x in {"B","P"}]
    turns = sum(bp[i] != bp[i-1] for i in range(1,len(bp)))
    turn_rate = turns / max(1, len(bp)-1)
    r8,r16,r32,rall = ratios(8),ratios(16),ratios(32),ratios(max(1,len(seq)))
    summary = np.asarray([
        min(len(seq),90)/90.0,
        min(len(bp),90)/90.0,
        min(_run_length(seq),12)/12.0,
        turn_rate,
        *r8,*r16,*r32,*rall,
        _entropy(r8),_entropy(r16),_entropy(r32),
        1.0 if bp and bp[-1]=="B" else 0.0,
        1.0 if bp and bp[-1]=="P" else 0.0,
    ], dtype=np.float32)
    if summary.size != HISTORY_SUMMARY_DIM:
        raise RuntimeError(f"history summary mismatch: {summary.size}")
    out = np.hstack([one_hot.reshape(-1), summary]).astype(np.float32)
    if out.size != HISTORY_INPUT_DIM:
        raise RuntimeError(f"history vector mismatch: {out.size}")
    return out


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    @property
    def baccarat_value(self) -> int:
        return self.rank if self.rank <= 9 else 0


@dataclass(frozen=True)
class HandResult:
    outcome: str
    player_point: int
    banker_point: int
    cards: tuple[Card,...]

    @property
    def card_count(self) -> int:
        return len(self.cards)


def new_eight_deck_shoe(rng: np.random.Generator) -> list[Card]:
    cards = [Card(rank,suit) for _ in range(DECKS) for suit in range(4) for rank in RANKS]
    order = rng.permutation(len(cards))
    return [cards[int(i)] for i in order]


def _total(cards: Sequence[Card]) -> int:
    return sum(c.baccarat_value for c in cards) % 10


def _banker_draws(total: int, player_third: int | None) -> bool:
    if player_third is None:
        return total <= 5
    if total <= 2: return True
    if total == 3: return player_third != 8
    if total == 4: return 2 <= player_third <= 7
    if total == 5: return 4 <= player_third <= 7
    if total == 6: return 6 <= player_third <= 7
    return False


def deal_baccarat_hand(shoe: list[Card], cursor: int) -> tuple[HandResult,int]:
    """標準百家樂補牌規則；只做物理模擬，不參與 Core 方向決策。"""
    if cursor + 6 > len(shoe):
        raise IndexError("not enough cards")
    player = [shoe[cursor], shoe[cursor+2]]
    banker = [shoe[cursor+1], shoe[cursor+3]]
    consumed = [shoe[cursor],shoe[cursor+1],shoe[cursor+2],shoe[cursor+3]]
    cursor += 4
    pt,bt = _total(player),_total(banker)
    if pt not in {8,9} and bt not in {8,9}:
        third = None
        if pt <= 5:
            c=shoe[cursor]; cursor+=1; player.append(c); consumed.append(c)
            third=c.baccarat_value; pt=_total(player)
        if _banker_draws(bt, third):
            c=shoe[cursor]; cursor+=1; banker.append(c); consumed.append(c); bt=_total(banker)
    outcome = "B" if bt>pt else "P" if pt>bt else "T"
    return HandResult(outcome,pt,bt,tuple(consumed)),cursor


def _remaining_rank_density(shoe: Sequence[Card], cursor: int) -> tuple[float,float]:
    rem=shoe[cursor:]
    if not rem: return 0.0,0.0
    z=float(len(rem))
    return sum(c.rank<=5 for c in rem)/z, sum(c.rank>=9 for c in rem)/z


def build_physics_target(hand: HandResult, shoe: Sequence[Card], cursor_before: int) -> np.ndarray:
    y=np.zeros(PHYSICS_DIM,dtype=np.float32); k=0
    y[k+{4:0,5:1,6:2}[hand.card_count]]=1; k+=3
    y[k+hand.player_point]=1; k+=10
    y[k+hand.banker_point]=1; k+=10
    y[k+{"B":0,"P":1,"T":2}[hand.outcome]]=1; k+=3
    ranks=np.zeros(13,dtype=np.float32); suits=np.zeros(4,dtype=np.float32)
    for c in hand.cards:
        ranks[c.rank-1]+=1; suits[c.suit]+=1
    y[k:k+13]=ranks; k+=13
    y[k:k+4]=suits/max(1.0,float(hand.card_count)); k+=4
    low,high=_remaining_rank_density(shoe,cursor_before)
    y[k]=float(cursor_before); y[k+1]=low; y[k+2]=high; k+=3
    diff=hand.banker_point-hand.player_point
    y[k]=diff/9.0; y[k+1]=abs(diff)/9.0
    return y


@dataclass
class SimulationDataset:
    x: np.ndarray
    y: np.ndarray
    shoe_ids: np.ndarray
    histories: list[str]
    actual_outcomes: list[str]


class OfflineBaccaratSimulator:
    """8 副牌離線 supervision；輸入永遠只有當下 B/P/T history。"""
    def __init__(self, *, cut_cards: int=60, random_state: int=DEFAULT_RANDOM_STATE, max_hands_per_shoe: int=90):
        self.cut_cards=int(max(14,min(120,cut_cards)))
        self.random_state=int(random_state)
        self.max_hands_per_shoe=int(max(1,max_hands_per_shoe))

    def generate(self, n_shoes: int) -> SimulationDataset:
        rng=np.random.default_rng(self.random_state)
        xs=[]; ys=[]; ids=[]; histories=[]; actual=[]
        for shoe_id in range(int(n_shoes)):
            shoe=new_eight_deck_shoe(rng); cursor=0; history=[]
            for _ in range(self.max_hands_per_shoe):
                if len(shoe)-cursor <= self.cut_cards+6: break
                before=cursor
                hand,cursor=deal_baccarat_hand(shoe,cursor)
                xs.append(history_to_vector(history))
                ys.append(build_physics_target(hand,shoe,before))
                ids.append(shoe_id)
                histories.append("".join(history))
                actual.append(hand.outcome)
                history.append(hand.outcome)
        if not xs: raise ValueError("simulation produced no rows")
        return SimulationDataset(np.vstack(xs).astype(np.float32),np.vstack(ys).astype(np.float32),
                                 np.asarray(ids,dtype=np.int32),histories,actual)


def _norm(block: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    a=np.clip(np.asarray(block,dtype=np.float64),0,None); s=float(a.sum())
    return fallback.astype(np.float32) if not math.isfinite(s) or s<=1e-12 else (a/s).astype(np.float32)


def sanitize_physics_prediction(raw: Sequence[float]) -> np.ndarray:
    x=np.asarray(raw,dtype=np.float64).reshape(-1)
    if x.size != PHYSICS_DIM: raise ValueError(f"expected {PHYSICS_DIM}, got {x.size}")
    out=np.zeros(PHYSICS_DIM,dtype=np.float32); k=0
    out[k:k+3]=_norm(x[k:k+3],np.array([.58,.34,.08])); k+=3
    out[k:k+10]=_norm(x[k:k+10],np.full(10,.1)); k+=10
    out[k:k+10]=_norm(x[k:k+10],np.full(10,.1)); k+=10
    out[k:k+3]=_norm(x[k:k+3],np.array([.4586,.4462,.0952])); k+=3
    out[k:k+13]=np.clip(x[k:k+13],0,6); k+=13
    out[k:k+4]=_norm(x[k:k+4],np.full(4,.25)); k+=4
    out[k]=np.clip(x[k],0,TOTAL_CARDS)
    out[k+1:k+3]=np.clip(x[k+1:k+3],0,1); k+=3
    out[k]=np.clip(x[k],-1,1); out[k+1]=np.clip(x[k+1],0,1)
    return out


class PhysicsFeatureExtractor:
    """輕量 multi-task MLP：一個模型一次輸出完整 48D，適合 browser export。"""
    def __init__(self, *, random_state: int=DEFAULT_RANDOM_STATE):
        self.random_state=int(random_state)
        self.scaler=StandardScaler()
        # 中文：48D target 同時包含 0~1 機率與 0~416 張數，必須做 target scaling，\n        # 否則 MLP loss 會被 consumed-card 維度支配。\n        self.target_scaler=StandardScaler()
        self.model=MLPRegressor(
            hidden_layer_sizes=(64,32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=8e-4,
            max_iter=80,
            early_stopping=True,
            validation_fraction=.1,
            n_iter_no_change=8,
            random_state=self.random_state,
            verbose=False,
        )
        self.is_fitted=False
        self.metadata: dict[str,Any]={}

    def fit(self,x: np.ndarray,y: np.ndarray) -> "PhysicsFeatureExtractor":
        xx=np.asarray(x,dtype=np.float32); yy=np.asarray(y,dtype=np.float32)
        scaled=self.scaler.fit_transform(xx)
        target_scaled=self.target_scaler.fit_transform(yy)
        self.model.fit(scaled,target_scaled)
        self.is_fitted=True
        return self

    def train_from_simulation(self, *, n_shoes: int=800, cut_cards: int=60, validation_fraction: float=.2) -> dict[str,float]:
        data=OfflineBaccaratSimulator(cut_cards=cut_cards,random_state=self.random_state).generate(n_shoes)
        unique=np.unique(data.shoe_ids); split=max(1,int(round(len(unique)*(1-validation_fraction))))
        train_ids=set(int(x) for x in unique[:split])
        train=np.asarray([int(s) in train_ids for s in data.shoe_ids]); valid=~train
        self.fit(data.x[train],data.y[train])
        raw_scaled=self.model.predict(self.scaler.transform(data.x[valid]))
        raw=self.target_scaler.inverse_transform(raw_scaled)
        pred=np.vstack([sanitize_physics_prediction(v) for v in raw])
        truth=data.y[valid]
        metrics={
            "validation_rmse":float(np.sqrt(np.mean((pred-truth)**2))),
            "card_count_accuracy":float(np.mean(np.argmax(pred[:,:3],1)==np.argmax(truth[:,:3],1))),
            "winner_accuracy":float(np.mean(np.argmax(pred[:,23:26],1)==np.argmax(truth[:,23:26],1))),
            "training_rows":float(train.sum()),"validation_rows":float(valid.sum())
        }
        self.metadata={**metrics,"n_shoes":int(n_shoes),"cut_cards":int(cut_cards),
                       "history_input_dim":HISTORY_INPUT_DIM,"physics_dim":PHYSICS_DIM,
                       "feature_names":list(PHYSICS_FEATURE_NAMES),
                       "semantic_note":"conditional expectations; not unseen-card reconstruction"}
        return metrics

    def predict_features(self,history_path: str | Sequence[str]) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("physics model not fitted")
        x=history_to_vector(history_path).reshape(1,-1)
        raw_scaled=self.model.predict(self.scaler.transform(x))
        raw=self.target_scaler.inverse_transform(raw_scaled)[0]
        return sanitize_physics_prediction(raw)

    def save(self,path: str | Path) -> None:
        if not self.is_fitted: raise RuntimeError("cannot save unfitted model")
        joblib.dump({"schema_version":3,"scaler":self.scaler,"target_scaler":self.target_scaler,"model":self.model,"metadata":self.metadata},path)

    @classmethod
    def load(cls,path: str | Path) -> "PhysicsFeatureExtractor":
        payload=joblib.load(path); obj=cls()
        obj.scaler=payload["scaler"]; obj.target_scaler=payload["target_scaler"]; obj.model=payload["model"]; obj.metadata=dict(payload.get("metadata") or {})
        obj.is_fitted=True; return obj

    def export_browser_bundle(self,path: str | Path) -> dict[str,Any]:
        if not self.is_fitted: raise RuntimeError("physics model not fitted")
        bundle={
            "schema_version":3,"model_type":"baccarat_physics_multitask_mlp","trained":True,
            "history_input_dim":HISTORY_INPUT_DIM,"physics_dim":PHYSICS_DIM,
            "feature_names":list(PHYSICS_FEATURE_NAMES),
            "scaler":{"mean":self.scaler.mean_.tolist(),"scale":self.scaler.scale_.tolist()},
            "target_scaler":{"mean":self.target_scaler.mean_.tolist(),"scale":self.target_scaler.scale_.tolist()},
            "activation":"relu",
            "coefs":[w.tolist() for w in self.model.coefs_],
            "intercepts":[b.tolist() for b in self.model.intercepts_],
            "metadata":self.metadata,
        }
        Path(path).write_text(json.dumps(bundle,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
        return bundle


_DEFAULT: PhysicsFeatureExtractor | None=None

def get_default_extractor() -> PhysicsFeatureExtractor:
    global _DEFAULT
    if _DEFAULT is None:
        path=os.environ.get("BGS_PHYSICS_MODEL_PATH",DEFAULT_MODEL_PATH)
        _DEFAULT=PhysicsFeatureExtractor.load(path)
    return _DEFAULT


# 中文：Core 與 original_7d 不重算、不改值、不改順序。
def prepare_xgboost_input(core_pb: float, original_7d: Sequence[float], history_path: str | Sequence[str],
                          *, extractor: PhysicsFeatureExtractor | None=None) -> np.ndarray:
    original=np.asarray(original_7d,dtype=np.float32).reshape(-1)
    if original.size != 7: raise ValueError("original_7d must contain exactly 7 values")
    if not np.all(np.isfinite(original)): raise ValueError("original_7d contains non-finite values")
    physics=(extractor or get_default_extractor()).predict_features(history_path)
    merged=np.hstack([[ _clip(core_pb) ],original,physics]).astype(np.float32)
    if merged.size != 56: raise RuntimeError(f"extended feature mismatch: {merged.size}")
    return merged


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train"); t.add_argument("--shoes",type=int,default=800); t.add_argument("--cut-cards",type=int,default=60)
    t.add_argument("--output",default=DEFAULT_MODEL_PATH); t.add_argument("--browser-output",default=DEFAULT_BROWSER_BUNDLE)
    t.add_argument("--random-state",type=int,default=DEFAULT_RANDOM_STATE)
    s=sub.add_parser("simulate"); s.add_argument("--shoes",type=int,default=100); s.add_argument("--cut-cards",type=int,default=60)
    s.add_argument("--output",default="physics_simulation_dataset.npz"); s.add_argument("--rows-output",default="")
    s.add_argument("--random-state",type=int,default=DEFAULT_RANDOM_STATE)
    return p


def main() -> int:
    a=build_parser().parse_args()
    if a.command=="simulate":
        d=OfflineBaccaratSimulator(cut_cards=a.cut_cards,random_state=a.random_state).generate(a.shoes)
        np.savez_compressed(a.output,x=d.x,y=d.y,shoe_ids=d.shoe_ids)
        if a.rows_output:
            rows=[{"shoe_id":int(s),"history":h,"actual_outcome":o} for s,h,o in zip(d.shoe_ids,d.histories,d.actual_outcomes)]
            Path(a.rows_output).write_text(json.dumps({"rows":rows},separators=(",",":")),encoding="utf-8")
        print(json.dumps({"rows":len(d.x),"output":a.output}))
        return 0
    m=PhysicsFeatureExtractor(random_state=a.random_state)
    metrics=m.train_from_simulation(n_shoes=a.shoes,cut_cards=a.cut_cards)
    m.save(a.output); m.export_browser_bundle(a.browser_output)
    print(json.dumps({"metrics":metrics,"model":a.output,"browser":a.browser_output},ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
