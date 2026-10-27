"""與 main/final_probability_runtime.js 的 historyVector 完全相同的 213D ABI（已逐行核對）。"""
import math
import numpy as np

WINDOW=64
DIM=213

def vector(history):
    seq=list(history)
    if any(c not in "BPT" for c in seq):
        raise ValueError("只能使用 B/P/T；輸入不得包含未開出的下一局")
    one=[0.0]*192
    tail=seq[-WINDOW:]
    for i,token in enumerate(tail):
        one[(WINDOW-len(tail)+i)*3+"BPT".index(token)]=1.0
    bp=[x for x in seq if x!="T"]
    nturn=sum(a!=b for a,b in zip(bp,bp[1:]))
    stage=0
    if bp:
        side=bp[-1]
        for x in reversed(bp):
            if x!=side:
                break
            stage+=1
    def ratios(n):
        part=seq[-n:] if n else seq
        return [part.count(x)/len(part) for x in "BPT"] if part else [0.0]*3
    r8=ratios(8);r16=ratios(16);r32=ratios(32);rall=ratios(max(1,len(seq)))
    def entropy(r):
        return -sum(p*math.log(p) for p in r if p>0)/math.log(3)
    summary=[min(len(seq),90)/90,min(len(bp),90)/90,min(stage,12)/12,
             nturn/max(1,len(bp)-1),*r8,*r16,*r32,*rall,
             entropy(r8),entropy(r16),entropy(r32),
             float(bool(bp) and bp[-1]=="B"),float(bool(bp) and bp[-1]=="P")]
    out=one+summary
    assert len(out)==DIM,(len(out),len(summary))
    return np.asarray(out,dtype=np.float32)
