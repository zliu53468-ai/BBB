#!/usr/bin/env python3
"""Non-invasive 48D baccarat physics feature plug-in.

Production architecture:
    B/P/T history -> lightweight multi-task MLP -> 48D conditional expectations
    Core P(B) + fixed 57D bridge -> direct-classification XGBoost

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
from sklearn.isotonic import IsotonicRegression
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
DEFAULT_PHYSICS_SHOES = 5000
# Multi-task emphasis: winner / point-distribution fidelity first, then card-count and composition.
# Architecture remains fixed at 213D input -> ReLU MLP -> 48D output.
PHYSICS_LOSS_WEIGHTS = np.asarray([1.25]*3+[1.10]*20+[2.50]*3+[.80]*13+[.65]*4+[.55]*5,dtype=np.float32)
PROBABILITY_BLOCKS = {"card_count":slice(0,3),"player_points":slice(3,13),"banker_points":slice(13,23),"winner":slice(23,26),"suit":slice(39,43)}
AFFINE_OUTPUT_INDICES = tuple(range(26,39)) + tuple(range(43,48))
assert PHYSICS_LOSS_WEIGHTS.size == PHYSICS_DIM


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


def augment_213d(x: np.ndarray,y: np.ndarray,*,ratio: float=.20,random_state: int=DEFAULT_RANDOM_STATE) -> tuple[np.ndarray,np.ndarray]:
    """Class-balanced history dropout plus tiny summary jitter; dimensions stay 213D."""
    xx=np.asarray(x,dtype=np.float32); yy=np.asarray(y,dtype=np.float32)
    count=int(round(len(xx)*max(0.0,min(.5,float(ratio)))))
    if count<=0: return xx,yy
    rng=np.random.default_rng(random_state); labels=np.argmax(yy[:,23:26],axis=1)
    pools=[np.flatnonzero(labels==k) for k in range(3)]; picks=[]
    for i in range(count):
        pool=pools[i%3] if len(pools[i%3]) else np.arange(len(xx)); picks.append(int(rng.choice(pool)))
    aug=xx[np.asarray(picks)].copy(); one=aug[:,:HISTORY_WINDOW*3].reshape(-1,HISTORY_WINDOW,3)
    one[rng.random(one.shape[:2])<.03]=0.0
    aug[:,HISTORY_WINDOW*3:]=np.clip(aug[:,HISTORY_WINDOW*3:]+rng.normal(0,.01,(count,HISTORY_SUMMARY_DIM)),0,1)
    return np.vstack((xx,aug)).astype(np.float32),np.vstack((yy,yy[picks])).astype(np.float32)


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


def _temperature_norm(block: np.ndarray,fallback: np.ndarray,temperature: float) -> np.ndarray:
    p=np.clip(_norm(block,fallback).astype(np.float64),1e-8,1); z=np.log(p)/max(.25,float(temperature)); z-=z.max()
    return (np.exp(z)/np.exp(z).sum()).astype(np.float32)


def fit_probability_temperatures(raw: np.ndarray,truth: np.ndarray) -> dict[str,float]:
    fallbacks={"card_count":np.array([.58,.34,.08]),"player_points":np.full(10,.1),"banker_points":np.full(10,.1),"winner":np.array([.4586,.4462,.0952]),"suit":np.full(4,.25)}
    result={}
    for name,block in PROBABILITY_BLOCKS.items():
        base=np.vstack([_norm(row[block],fallbacks[name]) for row in raw]); target=np.asarray(truth[:,block],dtype=float)
        candidates=np.linspace(.60,1.80,25); losses=[]
        for t in candidates:
            z=np.log(np.clip(base,1e-8,1))/t; z-=z.max(axis=1,keepdims=True); p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
            losses.append(float(-np.mean(np.sum(target*np.log(np.clip(p,1e-8,1)),axis=1))))
        result[name]=float(candidates[int(np.argmin(losses))])
    return result


def fit_output_affine(raw: np.ndarray,truth: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Per-output calibration for the non-probability 48D targets."""
    raw=np.asarray(raw,dtype=float); truth=np.asarray(truth,dtype=float)
    slope=np.ones(PHYSICS_DIM,dtype=np.float32); intercept=np.zeros(PHYSICS_DIM,dtype=np.float32)
    for index in AFFINE_OUTPUT_INDICES:
        source=raw[:,index]; target=truth[:,index]; variance=float(np.var(source))
        fitted=float(np.cov(source,target,bias=True)[0,1]/variance) if variance>1e-10 else 1.0
        slope[index]=np.float32(np.clip(fitted,.25,4.0))
        intercept[index]=np.float32(np.mean(target)-float(slope[index])*np.mean(source))
    return slope,intercept


def apply_output_affine(raw: np.ndarray,slope: np.ndarray,intercept: np.ndarray) -> np.ndarray:
    return np.asarray(raw,dtype=float)*np.asarray(slope,dtype=float)+np.asarray(intercept,dtype=float)


def physics_uncertainty_proxy(physics_48d: Sequence[float], round_index: float = 70.0) -> float:
    """Deterministic pre-calibration uncertainty proxy from the existing 48D output."""
    physics=np.asarray(physics_48d,dtype=np.float64).reshape(-1)
    if physics.size!=PHYSICS_DIM: raise ValueError(f"expected {PHYSICS_DIM}, got {physics.size}")
    def entropy(block: np.ndarray) -> float:
        p=np.clip(block,1e-8,None);p/=p.sum()
        return float(-np.sum(p*np.log(p))/math.log(len(p)))
    winner=np.sort(physics[23:26])[::-1];winner_gap=float(winner[0]-winner[1])
    density_gap=abs(_clip(physics[44])-_clip(physics[45]));density_ambiguity=1.0-min(1.0,density_gap/.25)
    raw=_clip(.10*entropy(physics[0:3])+.075*entropy(physics[3:13])+.075*entropy(physics[13:23])+
              .15*entropy(physics[23:26])+.25*entropy(physics[26:39])+.15*entropy(physics[39:43])+
              .075*(1.0-winner_gap)+.125*density_ambiguity)
    compressed=.50+.35*math.tanh((raw-.75)/.20)
    influence=.35 if round_index<=40 else .65 if round_index<=50 else 1.0
    return _clip(.50+(compressed-.50)*influence)


def fit_uncertainty_calibration(pred: np.ndarray, truth: np.ndarray, rounds: Sequence[float]) -> dict[str,Any]:
    """Calibrate the proxy against empirical 48D residual magnitude on held-out shoes."""
    pp=np.asarray(pred,dtype=np.float64);tt=np.asarray(truth,dtype=np.float64);rr=np.asarray(rounds,dtype=np.float64)
    if len(pp)!=len(tt) or len(pp)!=len(rr) or len(pp)<32:
        return {"method":"identity","x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.0]}
    weights=PHYSICS_LOSS_WEIGHTS.astype(np.float64);weights/=max(1e-12,float(np.mean(weights)))
    residual=np.mean(((pp-tt)**2)*weights.reshape(1,-1),axis=1)
    q10,q90=np.quantile(residual,[.10,.90]);span=max(1e-9,float(q90-q10))
    target=np.clip((residual-q10)/span,0.0,1.0)
    proxy=np.asarray([physics_uncertainty_proxy(row,rnd) for row,rnd in zip(pp,rr)],dtype=np.float64)
    if len(np.unique(proxy))<8:
        return {"method":"identity","x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.0],"residual_q10":float(q10),"residual_q90":float(q90)}
    iso=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip")
    iso.fit(proxy,target)
    calibrated=np.asarray(iso.predict(proxy),dtype=np.float64)
    corr=float(np.corrcoef(calibrated,residual)[0,1]) if np.std(calibrated)>1e-10 and np.std(residual)>1e-10 else 0.0
    return {"method":"isotonic","x_thresholds":iso.X_thresholds_.tolist(),"y_thresholds":iso.y_thresholds_.tolist(),
            "residual_q10":float(q10),"residual_q90":float(q90),"calibration_error_correlation":corr}


def apply_uncertainty_calibration(values: Sequence[float], calibration: Mapping[str,Any] | None) -> np.ndarray:
    v=np.clip(np.asarray(values,dtype=np.float64),0.0,1.0)
    if not calibration or calibration.get("method")!="isotonic": return v
    x=np.asarray(calibration.get("x_thresholds") or [],dtype=np.float64);y=np.asarray(calibration.get("y_thresholds") or [],dtype=np.float64)
    return np.clip(np.interp(v,x,y),0.0,1.0) if len(x)>1 and len(x)==len(y) else v


def shoe_bootstrap_ci(values: Sequence[float],shoe_ids: Sequence[int],*,samples: int=1000,random_state: int=DEFAULT_RANDOM_STATE) -> list[float]:
    values=np.asarray(values,dtype=float); groups=np.asarray(shoe_ids); unique=np.unique(groups)
    if len(unique)<2 or samples<=0:
        mean=float(np.mean(values)); return [mean,mean]
    rng=np.random.default_rng(random_state); estimates=[]
    for _ in range(int(samples)):
        chosen=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([np.flatnonzero(groups==g) for g in chosen])
        estimates.append(float(np.mean(values[idx])))
    return [float(x) for x in np.percentile(estimates,[2.5,97.5])]


def sanitize_physics_prediction(raw: Sequence[float],temperatures: dict[str,float] | None=None) -> np.ndarray:
    x=np.asarray(raw,dtype=np.float64).reshape(-1)
    if x.size != PHYSICS_DIM: raise ValueError(f"expected {PHYSICS_DIM}, got {x.size}")
    temperatures=temperatures or {}
    out=np.zeros(PHYSICS_DIM,dtype=np.float32); k=0
    out[k:k+3]=_temperature_norm(x[k:k+3],np.array([.58,.34,.08]),temperatures.get("card_count",1)); k+=3
    out[k:k+10]=_temperature_norm(x[k:k+10],np.full(10,.1),temperatures.get("player_points",1)); k+=10
    out[k:k+10]=_temperature_norm(x[k:k+10],np.full(10,.1),temperatures.get("banker_points",1)); k+=10
    out[k:k+3]=_temperature_norm(x[k:k+3],np.array([.4586,.4462,.0952]),temperatures.get("winner",1)); k+=3
    out[k:k+13]=np.clip(x[k:k+13],0,6); k+=13
    out[k:k+4]=_temperature_norm(x[k:k+4],np.full(4,.25),temperatures.get("suit",1)); k+=4
    out[k]=np.clip(x[k],0,TOTAL_CARDS)
    out[k+1:k+3]=np.clip(x[k+1:k+3],0,1); k+=3
    out[k]=np.clip(x[k],-1,1); out[k+1]=np.clip(x[k+1],0,1)
    return out


class PhysicsFeatureExtractor:
    """輕量 multi-task MLP：一個模型一次輸出完整 48D，適合 browser export。"""
    def __init__(self, *, random_state: int=DEFAULT_RANDOM_STATE):
        self.random_state=int(random_state)
        self.scaler=StandardScaler()
        # 中文：48D target 同時包含 0~1 機率與 0~416 張數，必須做 target scaling，
        # 否則 MLP loss 會被 consumed-card 維度支配。
        self.target_scaler=StandardScaler()
        self.loss_scale=np.sqrt(PHYSICS_LOSS_WEIGHTS).astype(np.float32)
        self.calibration_temperatures: dict[str,float]={}
        self.output_slope=np.ones(PHYSICS_DIM,dtype=np.float32)
        self.output_intercept=np.zeros(PHYSICS_DIM,dtype=np.float32)
        self.uncertainty_calibration: dict[str,Any]={"method":"identity","x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.0]}
        self.model=MLPRegressor(
            hidden_layer_sizes=(64,32),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            batch_size=512,
            learning_rate_init=5e-4,
            max_iter=160,
            early_stopping=True,
            validation_fraction=.15,
            n_iter_no_change=15,
            random_state=self.random_state,
            verbose=False,
        )
        self.is_fitted=False
        self.metadata: dict[str,Any]={}

    def fit(self,x: np.ndarray,y: np.ndarray) -> "PhysicsFeatureExtractor":
        xx=np.asarray(x,dtype=np.float32); yy=np.asarray(y,dtype=np.float32)
        scaled=self.scaler.fit_transform(xx)
        target_scaled=self.target_scaler.fit_transform(yy)*self.loss_scale
        self.model.fit(scaled,target_scaled)
        self.is_fitted=True
        return self

    def _decode_scaled(self,raw_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(raw_scaled,dtype=float)/self.loss_scale)

    def train_from_simulation(self, *, n_shoes: int=DEFAULT_PHYSICS_SHOES, cut_cards: int=60, validation_fraction: float=.2,
                              calibration_fraction: float=.1, augment_ratio: float=.25, bootstrap_samples: int=1000) -> dict[str,Any]:
        data=OfflineBaccaratSimulator(cut_cards=cut_cards,random_state=self.random_state).generate(n_shoes)
        unique=np.unique(data.shoe_ids)
        if len(unique)<3: raise ValueError("strict train/calibration/holdout split requires at least three shoes")
        holdout_count=max(1,int(math.ceil(len(unique)*validation_fraction))); calibration_count=max(1,int(math.ceil(len(unique)*calibration_fraction)))
        calibration_at=max(1,len(unique)-holdout_count-calibration_count); holdout_at=len(unique)-holdout_count
        train_ids=set(int(x) for x in unique[:calibration_at]); calibration_ids=set(int(x) for x in unique[calibration_at:holdout_at])
        train=np.asarray([int(s) in train_ids for s in data.shoe_ids]); calibration=np.asarray([int(s) in calibration_ids for s in data.shoe_ids]); valid=~(train|calibration)
        train_x,train_y=augment_213d(data.x[train],data.y[train],ratio=augment_ratio,random_state=self.random_state)
        self.fit(train_x,train_y)
        calibration_raw=self._decode_scaled(self.model.predict(self.scaler.transform(data.x[calibration])))
        self.output_slope,self.output_intercept=fit_output_affine(calibration_raw,data.y[calibration])
        calibration_raw=apply_output_affine(calibration_raw,self.output_slope,self.output_intercept)
        self.calibration_temperatures=fit_probability_temperatures(calibration_raw,data.y[calibration])
        calibration_pred=np.vstack([sanitize_physics_prediction(v,self.calibration_temperatures) for v in calibration_raw])
        calibration_rows=np.flatnonzero(calibration)
        calibration_rounds=np.asarray([len(data.histories[int(i)])+1 for i in calibration_rows],dtype=np.float64)
        self.uncertainty_calibration=fit_uncertainty_calibration(calibration_pred,data.y[calibration],calibration_rounds)
        raw=self._decode_scaled(self.model.predict(self.scaler.transform(data.x[valid])))
        raw=apply_output_affine(raw,self.output_slope,self.output_intercept)
        pred=np.vstack([sanitize_physics_prediction(v,self.calibration_temperatures) for v in raw])
        truth=data.y[valid]
        card_ok=(np.argmax(pred[:,:3],1)==np.argmax(truth[:,:3],1)).astype(float); winner_ok=(np.argmax(pred[:,23:26],1)==np.argmax(truth[:,23:26],1)).astype(float)
        row_mse=np.mean((pred-truth)**2,axis=1); valid_shoes=data.shoe_ids[valid]
        valid_rows=np.flatnonzero(valid);valid_rounds=np.asarray([len(data.histories[int(i)])+1 for i in valid_rows],dtype=np.float64)
        raw_noise=np.asarray([physics_uncertainty_proxy(row,rnd) for row,rnd in zip(pred,valid_rounds)],dtype=np.float64)
        calibrated_noise=apply_uncertainty_calibration(raw_noise,self.uncertainty_calibration)
        noise_error_corr=float(np.corrcoef(calibrated_noise,row_mse)[0,1]) if np.std(calibrated_noise)>1e-10 and np.std(row_mse)>1e-10 else 0.0
        metrics={
            "validation_rmse":float(np.sqrt(np.mean(row_mse))),"validation_mse_ci95":shoe_bootstrap_ci(row_mse,valid_shoes,samples=bootstrap_samples),
            "uncertainty_error_correlation":noise_error_corr,"uncertainty_mean":float(np.mean(calibrated_noise)),
            "card_count_accuracy":float(card_ok.mean()),"card_count_accuracy_ci95":shoe_bootstrap_ci(card_ok,valid_shoes,samples=bootstrap_samples),
            "winner_accuracy":float(winner_ok.mean()),"winner_accuracy_ci95":shoe_bootstrap_ci(winner_ok,valid_shoes,samples=bootstrap_samples),
            "training_rows":float(train.sum()),"augmented_training_rows":float(len(train_x)),"calibration_rows":float(calibration.sum()),"validation_rows":float(valid.sum())
        }
        self.metadata={**metrics,"n_shoes":int(n_shoes),"cut_cards":int(cut_cards),
                       "split":{"train_shoes":len(train_ids),"calibration_shoes":len(calibration_ids),"holdout_shoes":int(len(unique)-holdout_at)},
                       "augment_ratio":float(augment_ratio),"loss_weights":PHYSICS_LOSS_WEIGHTS.tolist(),"calibration_temperatures":self.calibration_temperatures,
                       "uncertainty_calibration":self.uncertainty_calibration,
                       "output_calibration":{"slope":self.output_slope.tolist(),"intercept":self.output_intercept.tolist()},
                       "history_input_dim":HISTORY_INPUT_DIM,"physics_dim":PHYSICS_DIM,
                       "feature_names":list(PHYSICS_FEATURE_NAMES),
                       "semantic_note":"conditional expectations; not unseen-card reconstruction"}
        return metrics

    def predict_features(self,history_path: str | Sequence[str]) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("physics model not fitted")
        x=history_to_vector(history_path).reshape(1,-1)
        raw_scaled=self.model.predict(self.scaler.transform(x))
        raw=apply_output_affine(self._decode_scaled(raw_scaled)[0],self.output_slope,self.output_intercept)
        return sanitize_physics_prediction(raw,self.calibration_temperatures)

    def save(self,path: str | Path) -> None:
        if not self.is_fitted: raise RuntimeError("cannot save unfitted model")
        joblib.dump({"schema_version":6,"scaler":self.scaler,"target_scaler":self.target_scaler,"loss_scale":self.loss_scale,"calibration_temperatures":self.calibration_temperatures,"uncertainty_calibration":self.uncertainty_calibration,"output_slope":self.output_slope,"output_intercept":self.output_intercept,"model":self.model,"metadata":self.metadata},path)

    @classmethod
    def load(cls,path: str | Path) -> "PhysicsFeatureExtractor":
        payload=joblib.load(path); obj=cls()
        obj.scaler=payload["scaler"]; obj.target_scaler=payload["target_scaler"]; obj.model=payload["model"]; obj.metadata=dict(payload.get("metadata") or {})
        obj.loss_scale=np.asarray(payload.get("loss_scale",np.ones(PHYSICS_DIM)),dtype=np.float32); obj.calibration_temperatures=dict(payload.get("calibration_temperatures") or {})
        obj.uncertainty_calibration=dict(payload.get("uncertainty_calibration") or obj.metadata.get("uncertainty_calibration") or {"method":"identity","x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.0]})
        obj.output_slope=np.asarray(payload.get("output_slope",np.ones(PHYSICS_DIM)),dtype=np.float32); obj.output_intercept=np.asarray(payload.get("output_intercept",np.zeros(PHYSICS_DIM)),dtype=np.float32)
        obj.is_fitted=True; return obj

    def export_browser_bundle(self,path: str | Path) -> dict[str,Any]:
        if not self.is_fitted: raise RuntimeError("physics model not fitted")
        bundle={
            "schema_version":6,"model_type":"baccarat_physics_multitask_mlp","trained":True,
            "history_input_dim":HISTORY_INPUT_DIM,"physics_dim":PHYSICS_DIM,
            "feature_names":list(PHYSICS_FEATURE_NAMES),
            "scaler":{"mean":self.scaler.mean_.tolist(),"scale":self.scaler.scale_.tolist()},
            "target_scaler":{"mean":self.target_scaler.mean_.tolist(),"scale":self.target_scaler.scale_.tolist()},
            "loss_scale":self.loss_scale.tolist(),"calibration_temperatures":self.calibration_temperatures,
            "uncertainty_calibration":self.uncertainty_calibration,
            "output_calibration":{"slope":self.output_slope.tolist(),"intercept":self.output_intercept.tolist()},
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
    t=sub.add_parser("train"); t.add_argument("--shoes",type=int,default=DEFAULT_PHYSICS_SHOES); t.add_argument("--cut-cards",type=int,default=60)
    t.add_argument("--output",default=DEFAULT_MODEL_PATH); t.add_argument("--browser-output",default=DEFAULT_BROWSER_BUNDLE)
    t.add_argument("--validation-fraction",type=float,default=.20); t.add_argument("--calibration-fraction",type=float,default=.10)
    t.add_argument("--augment-ratio",type=float,default=.25); t.add_argument("--bootstrap-samples",type=int,default=1000)
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
    metrics=m.train_from_simulation(n_shoes=a.shoes,cut_cards=a.cut_cards,validation_fraction=a.validation_fraction,
                                    calibration_fraction=a.calibration_fraction,augment_ratio=a.augment_ratio,bootstrap_samples=a.bootstrap_samples)
    m.save(a.output); m.export_browser_bundle(a.browser_output)
    print(json.dumps({"metrics":metrics,"model":a.output,"browser":a.browser_output},ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
