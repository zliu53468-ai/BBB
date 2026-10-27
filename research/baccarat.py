"""精確八副牌百家樂：10 類點數、有限牌組不放回枚舉；不供正式系統 import。"""
from __future__ import annotations
from functools import lru_cache
import random
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda func: func

INITIAL = (128,) + (32,) * 9  # 0:10/J/Q/K, 1:A, 2..9:數字
OUTCOMES = "BPT"
SEED = 20261027

@njit(cache=True)
def _banker_draw(total, third):
    # third=-1: 閒家停牌；total 為莊前兩張點數。
    if third == -1:
        return total <= 5
    if total <= 2:
        return True
    if total == 3:
        return third != 8
    if total == 4:
        return 2 <= third <= 7
    if total == 5:
        return 4 <= third <= 7
    if total == 6:
        return 6 <= third <= 7
    return False

@njit(cache=True)
def _record(out, p, b, cards, weight):
    if b > p:
        out[0] += weight
    elif p > b:
        out[1] += weight
    else:
        out[2] += weight
    out[3 + cards - 4] += weight
    out[6 + p] += weight
    out[16 + b] += weight
    out[26] += abs(p-b) * weight

@njit(cache=True)
def _exact_kernel(start):
    """精確依次放回前四張排列及條件第五/六張，不用 Monte Carlo。"""
    counts = start.copy()
    total = 0
    for v in counts:
        total += v
    if total < 6:
        raise ValueError("需要至少六張剩餘牌")
    out = np.zeros(27, dtype=np.float64)
    for p1 in range(10):
        if not counts[p1]:
            continue
        w1 = counts[p1]/total
        counts[p1] -= 1
        for b1 in range(10):
            if not counts[b1]:
                continue
            w2 = w1*counts[b1]/(total-1)
            counts[b1] -= 1
            for p2 in range(10):
                if not counts[p2]:
                    continue
                w3 = w2*counts[p2]/(total-2)
                counts[p2] -= 1
                for b2 in range(10):
                    if not counts[b2]:
                        continue
                    w4 = w3*counts[b2]/(total-3)
                    counts[b2] -= 1
                    pt = (p1+p2)%10
                    bt = (b1+b2)%10
                    if pt >= 8 or bt >= 8: # 天牌，雙方不補
                        _record(out,pt,bt,4,w4)
                    elif pt <= 5:
                        for p3 in range(10):
                            if not counts[p3]:
                                continue
                            w5 = w4*counts[p3]/(total-4)
                            counts[p3] -= 1
                            finalp=(pt+p3)%10
                            if _banker_draw(bt,p3):
                                for b3 in range(10):
                                    if counts[b3]:
                                        _record(out,finalp,(bt+b3)%10,6,
                                                w5*counts[b3]/(total-5))
                            else:
                                _record(out,finalp,bt,5,w5)
                            counts[p3] += 1
                    elif _banker_draw(bt,-1):
                        for b3 in range(10):
                            if counts[b3]:
                                _record(out,pt,(bt+b3)%10,5,
                                        w4*counts[b3]/(total-4))
                    else:
                        _record(out,pt,bt,4,w4)
                    counts[b2] += 1
                counts[p2] += 1
            counts[b1] += 1
        counts[p1] += 1
    return out

@lru_cache(maxsize=16384)
def exact_distribution(composition):
    """輸出 (B,P,T, cards4..6, player_points0..9, banker_points0..9, abs_diff)。
    前 26 維為機率，最後為期望絕對點差。只依當下剩餘牌組計算。
    """
    key=tuple(int(x) for x in composition)
    if len(key)!=10 or min(key)<0 or sum(key)<6:
        raise ValueError("composition 必須是十種點數的非負張數")
    return tuple(float(x) for x in _exact_kernel(np.asarray(key,dtype=np.int64)))

def draw(counts, rng):
    total=sum(counts)
    if total<=0:
        raise ValueError("牌靴用盡")
    k=rng.randrange(total)
    for i, n in enumerate(counts):
        if k<n:
            counts[i]-=1
            return i
        k-=n
    raise AssertionError("取牌失敗")

def deal(counts, rng):
    """固定發牌順序 P1 B1 P2 B2；回傳觀察結果與張數。"""
    p1=draw(counts,rng); b1=draw(counts,rng)
    p2=draw(counts,rng); b2=draw(counts,rng)
    p=(p1+p2)%10; b=(b1+b2)%10
    n=4
    if p<8 and b<8:
        third=-1
        if p<=5:
            third=draw(counts,rng); p=(p+third)%10; n+=1
        if _banker_draw(b,third):
            b=(b+draw(counts,rng))%10; n+=1
    return ("B" if b>p else "P" if p>b else "T"),n,p,b

def burn(counts, rng, additional=True):
    """揭開首張，依點數再燒 1..10 張；0 點牌(10/J/Q/K)視為 10 張。
    可關閉 additional 模式，對照不同賭場之燒牌慣例。
    """
    first=draw(counts,rng)
    extra=(10 if first==0 else first) if additional else max(0,(10 if first==0 else first)-1)
    for _ in range(extra):
        draw(counts,rng)
    return 1+extra

def simulate_shoe(shoe_id, seed=SEED, decks=8, cut_from_end=14,
                  include_exact=True):
    if decks<1 or cut_from_end<6:
        raise ValueError("decks>=1 且 cut_from_end>=6")
    rng=random.Random(int(seed)+int(shoe_id)*104729)
    counts=[v*decks//8 for v in INITIAL] if decks!=8 else list(INITIAL)
    # 一副牌各點數張數為 (16,4,...,4)；避免整數截斷偏差。
    counts=[16*decks]+[4*decks]*9
    burned=burn(counts,rng)
    if sum(counts)<=cut_from_end:
        raise ValueError("燒牌後沒有可發的牌")
    history=[]
    while sum(counts)>cut_from_end:
        before=tuple(counts)
        probs=exact_distribution(before) if include_exact else None
        outcome,n,ps,bs=deal(counts,rng)
        yield dict(shoe_id=int(shoe_id), round=len(history)+1,
                   history="".join(history), outcome=outcome, cards=n,
                   ground_truth=list(probs) if probs is not None else None,
                   remaining_before=sum(before), burned=burned)
        history.append(outcome)
