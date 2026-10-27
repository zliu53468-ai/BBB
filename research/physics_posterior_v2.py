"""Physics Posterior V2 — 僅供 research/ 使用的 B/P/T 可觀測後驗教師。

train 才能讀取模擬真實牌組生成的 soft labels；
predict 僅允許 213D 歷史結果，絕不可取得真實 composition。
非參數 bucket 平滑不是能重建未知牌序，也不能保證正期望。
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

class PosteriorTeacher:
    def __init__(self, context_length=5, prior_strength=200.0):
        if not (0<=context_length<=16 and prior_strength>0):
            raise ValueError("invalid predeclared hyperparameters")
        self.context_length=int(context_length)
        self.prior_strength=float(prior_strength)
        self.trained=False

    def _key(self,x):
        if len(x)!=213:
            raise ValueError("expects production 213D ABI")
        # 192 個 one-hot；最後五局（含和局）加上觀測局數分層。
        # 空局採 3（不會和 B/P/T 混淆），不能使用未來標籤。
        history=np.asarray(x[:192],dtype=float).reshape(64,3)
        tail=history[-self.context_length:] if self.context_length else history[:0]
        code=tuple(int(np.argmax(v)) if np.sum(v)>0.5 else 3 for v in tail)
        # summary[0] = min(rounds,90)/90，分 1..20、21..45、46..。
        rounds=int(round(float(x[192])*90))
        phase=0 if rounds<=20 else 1 if rounds<=45 else 2
        return (phase,*code)

    def fit(self,x,outcome_soft,point_soft):
        x=np.asarray(x);a=np.asarray(outcome_soft,dtype=float);c=np.asarray(point_soft,dtype=float)
        if x.ndim!=2 or x.shape[1]!=213 or a.shape!=(len(x),3) or c.shape!=(len(x),10):
            raise ValueError("dimension mismatch")
        if not len(x) or not np.isfinite(a).all() or not np.isfinite(c).all():
            raise ValueError("invalid simulation labels")
        if not np.allclose(a.sum(axis=1),1,atol=1e-3) or not np.allclose(c.sum(axis=1),1,atol=1e-3):
            raise ValueError("soft labels must be probability distributions")
        self.global_out=a.mean(axis=0)
        self.global_points=c.mean(axis=0)
        self.buckets=defaultdict(lambda:[0,np.zeros(3),np.zeros(10)])
        for history,prob,composition in zip(x,a,c):
            bucket=self.buckets[self._key(history)]
            bucket[0]+=1
            bucket[1]+=prob
            bucket[2]+=composition
        self.trained=True
        return self

    def predict(self,x):
        if not self.trained:raise ValueError("fit on train shoes first")
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=213:
            raise ValueError("x must be (n,213)")
        win=np.empty((len(x),3));points=np.empty((len(x),10));support=np.empty(len(x),dtype=int)
        for i,row in enumerate(x):
            bucket=self.buckets.get(self._key(row))
            if bucket is None:
                n=0;out=np.zeros(3);comp=np.zeros(10)
            else:
                n,out,comp=bucket
            k=self.prior_strength
            win[i]=(out+k*self.global_out)/(n+k)
            points[i]=(comp+k*self.global_points)/(n+k)
            support[i]=n
        return {"p":win,"remaining_point_probabilities":points,
                "bucket_train_support":support}

    def predict_proba(self,x):
        return self.predict(x)["p"]

    def diagnose(self,x):
        q=self.predict(x)
        return {"test_rows":len(x),
                "median_training_support":float(np.median(q["bucket_train_support"])) if len(x) else 0,
                "unseen_fraction":float(np.mean(q["bucket_train_support"]==0)) if len(x) else 0,
                "prior_strength":self.prior_strength,
                "context_length":self.context_length,
                "note":"per-bucket conditional expectation of exact outcomes; not exact remaining deck reconstruction"}
