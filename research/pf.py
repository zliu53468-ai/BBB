"""研究用 bootstrap PF：完整組成機率、結果條件下的隱藏發牌抽樣、ESS 重抽樣。

注意：使用 outcome 對「已結束局」做過濾，只能於下一局事前預測。
不得對當局真實剩餘組成洩漏；與既有 stateless PF500 獨立。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .baccarat import INITIAL, SEED, burn, deal, exact_distribution
from .run import split, metrics, returns, shoe_rows, bootstrap, pooled, write_json

class CompositionPF:
    def __init__(self,particles=500,seed=SEED,threshold=.5,rejuvenate=True):
        self.n=int(particles)
        if self.n<2:
            raise ValueError("粒子數太少")
        self.rng=np.random.default_rng(seed)
        # 使用 numpy Generator 直接當 randrange 不支援；建立相容類。
        class Adapter:
            def __init__(self,rng):self.rng=rng
            def randrange(self,n):return int(self.rng.integers(n))
        self.adapter=Adapter(self.rng)
        self.counts=[]
        for _ in range(self.n):
            c=list(INITIAL);burn(c,self.adapter)
            self.counts.append(c)
        self.weights=np.full(self.n,1/self.n)
        self.threshold=float(threshold)
        self.rejuvenate=bool(rejuvenate)
        self.ess_history=[]

    def forecast(self):
        out=np.zeros(3)
        for weight,c in zip(self.weights,self.counts):
            out+=weight*np.asarray(exact_distribution(tuple(c))[:3])
        return out/out.sum()

    def observe(self,outcome):
        """先 likelihood，後用拒絕取樣抽取條件隱藏牌；不使用未來或真實發牌。"""
        if outcome not in "BPT":
            raise ValueError("outcome must be B/P/T")
        previous=[c.copy() for c in self.counts]
        likelihoods=np.asarray([exact_distribution(tuple(c))["BPT".index(outcome)]
                                for c in previous],dtype=float)
        post=self.weights*likelihoods
        mass=post.sum()
        if mass<=0:
            raise RuntimeError("觀測似然為零，不可任意重設成均勻權重")
        self.weights=post/mass
        for i,pre in enumerate(previous):
            # rejection sampling 保證 P(隱藏牌組 | 觀測結果, precomposition) 正確。
            for trial in range(100000):
                proposal=pre.copy()
                if deal(proposal,self.adapter)[0]==outcome:
                    self.counts[i]=proposal
                    break
            else:
                raise RuntimeError("無法取得條件隱藏牌樣本")
        ess=float(1.0/np.sum(self.weights**2))
        self.ess_history.append(ess)
        if ess<self.threshold*self.n:
            # 系統重抽樣（以加權 CDF）
            positions=(self.rng.random()+np.arange(self.n))/self.n
            index=np.searchsorted(np.cumsum(self.weights),positions,side="left")
            index=np.minimum(index,self.n-1)
            self.counts=[self.counts[i].copy() for i in index]
            previous=[previous[i].copy() for i in index]
            self.weights=np.full(self.n,1/self.n)
            if self.rejuvenate:
                # 合法的最後一局條件轉移 Gibbs refresh：
                # 只使用前態與已觀察 outcome，不直接交換組成張數（會改壞後驗）。
                for i,pre in enumerate(previous):
                    for trial in range(100000):
                        proposal=pre.copy()
                        if deal(proposal,self.adapter)[0]==outcome:
                            self.counts[i]=proposal
                            break
                    else:
                        raise RuntimeError("rejuvenation 條件抽樣失敗")
        return ess

def run_stage3(data,reports="research/reports",seed=SEED,max_shoes=100):
    gate=json.loads((Path(reports)/"stage1.json").read_text(encoding="utf-8"))["gate_1"]
    if not gate["passed"]:
        write_json(Path(reports)/"stage3.json",{"status":"SKIPPED_GATE_1","seed":seed})
        return
    from .run import chunks
    from scipy.stats import pearsonr
    # 小型 deterministic pilot 先做 500/2000/10000 收斂比較。
    # max_shoes 與粒子數為研究運算預算，未執行滿量不得宣稱所有資料已驗證。
    grouped={}
    for part in chunks(data):
        ids=part["shoe"]
        for i,sid in enumerate(ids):
            if split(sid,seed)!=2:
                continue
            d=grouped.setdefault(int(sid),[])
            d.append((int(part["round"][i]),int(part["y"][i]),part["p"][i]))
            if len(grouped)>max_shoes*4:break
        if len(grouped)>max_shoes*4:break
    shoe_ids=sorted(grouped)[:max_shoes]
    summary={"seed":seed,"pilot_test_shoes":len(shoe_ids),
             "particle_results":{},"status":"PILOT_ONLY"}
    for count in (500,2000,10000):
        preds=[];truth=[];actual=[];all_ess=[];used=[]
        for sid in shoe_ids:
            pf=CompositionPF(count,seed+sid)
            for position,label,exact in sorted(grouped[sid]):
                pred=pf.forecast()
                preds.append(pred);truth.append(exact);actual.append(label);used.append(sid)
                pf.observe("BPT"[label])
            all_ess.extend(pf.ess_history)
        if not actual:continue
        y=np.asarray(actual,dtype=np.int8);p=np.asarray(preds)
        ground=np.asarray(truth);ids=np.asarray(used)
        # 防止 test 偷選阈值：只以機率 log loss、全時段永遠莊 vs PF 作無調參比較。
        direction=np.where(.95*p[:,0]-p[:,1]>p[:,1]-p[:,0],0,1).astype(np.int8)
        rows=shoe_rows(ids,y,p,direction)
        coef=pearsonr(p[:,0],ground[:,0]).statistic if np.std(p[:,0])>0 and np.std(ground[:,0])>0 else None
        summary["particle_results"][str(count)]={
           **metrics(y,p),
           "mae_pB":float(np.abs(p[:,0]-ground[:,0]).mean()),
           "pearson_pB":float(coef) if coef is not None else None,
           "ess_mean":float(np.mean(all_ess)),
           "ess_min":float(np.min(all_ess)),
           "always_choose_higher_ev":pooled(rows),
           "ev_ci95":bootstrap(rows,seed),
           "n_hands":len(y)}
    write_json(Path(reports)/"stage3.json",summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.bar(list(summary["particle_results"]),
               [v["ess_mean"]/int(k) for k,v in summary["particle_results"].items()])
        ax.set(ylabel="平均 ESS / N",xlabel="粒子數",title="後驗粒子有效率")
        fig.tight_layout();fig.savefig(Path(reports)/"ess_curve.png",dpi=150);plt.close(fig)
    except ImportError:pass
