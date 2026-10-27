"""研究階段一：完整靴模擬、整靴 holdout、bootstrap、預註冊擇時閘門。

用法: python -m research.run generate --shoes 200000 --out research/data
      python -m research.run stage1 --data research/data --reports research/reports
未執行長算時絕不輸出成功結論。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .baccarat import SEED, INITIAL, exact_distribution, simulate_shoe
from .features import vector

OUT="BPT"
QUANTILES=(0.01,0.02,0.05,0.10,0.20,0.50,1.0)
BOOTSTRAPS=1500

def split(shoe_id,seed=SEED):
    b=hashlib.sha256(f"{seed}:{int(shoe_id)}".encode()).digest()
    v=int.from_bytes(b[:8],"big")/2**64
    return 0 if v<0.70 else 1 if v<0.85 else 2

def generate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(seed=args.seed,decks=8,cut_from_end=args.cut,shoes=args.shoes,
                  start_shoe=args.start_shoe,shard_size=args.shard_size,
                  exact=True,abi="213D:64*3+21",version=1)
    (out/"config.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for start in range(args.start_shoe,args.start_shoe+args.shoes,args.shard_size):
        stop=min(start+args.shard_size,args.start_shoe+args.shoes)
        path=out/f"shoe_{start:09d}_{stop:09d}.npz"
        if path.exists():
            raise FileExistsError(f"不能靜默覆蓋既有實驗：{path}")
        rows=[];yy=[];pp=[];aux=[];ids=[];rounds=[]
        for sid in range(start,stop):
            for hand in simulate_shoe(sid,args.seed,cut_from_end=args.cut):
                rows.append(vector(hand["history"]))
                yy.append(OUT.index(hand["outcome"]))
                gt=hand["ground_truth"]
                pp.append(gt[:3]);aux.append(gt[3:])
                ids.append(sid);rounds.append(hand["round"])
        np.savez_compressed(path,x=np.asarray(rows,dtype=np.float32),
            y=np.asarray(yy,dtype=np.uint8),p=np.asarray(pp,dtype=np.float32),
            aux=np.asarray(aux,dtype=np.float32),shoe=np.asarray(ids,dtype=np.int32),
            round=np.asarray(rounds,dtype=np.int16))
        print(f"{path.name}: {len(yy)} 局 / {stop-start} 靴",flush=True)

def chunks(data):
    paths=sorted(Path(data).glob("shoe_*.npz"))
    if not paths:
        raise FileNotFoundError(f"{data} 找不到模擬分片，先執行 generate")
    for path in paths:
        with np.load(path) as z:
            yield {k:z[k] for k in ("x","y","p","aux","shoe","round")}

def collect_train_valid(data,seed,max_train=300000,max_valid=80000):
    # 第一趟統計靴分組筆數，第二趟按固定機率抽樣，避免載入千萬局造成 OOM。
    counts=np.zeros(3,dtype=np.int64)
    for d in chunks(data):
        groups=np.fromiter((split(i,seed) for i in d["shoe"]),dtype=np.int8,count=len(d["shoe"]))
        counts+=np.bincount(groups,minlength=3)
    rng=np.random.default_rng(seed)
    xs=[[],[]];ys=[[],[]];ps=[[],[]]
    for d in chunks(data):
        groups=np.fromiter((split(i,seed) for i in d["shoe"]),dtype=np.int8,count=len(d["shoe"]))
        for g,limit in ((0,max_train),(1,max_valid)):
            ix=np.flatnonzero(groups==g)
            if len(ix) and counts[g]>limit:
                ix=ix[rng.random(len(ix))<limit/counts[g]]
            if len(ix):
                xs[g].append(d["x"][ix]);ys[g].append(d["y"][ix]);ps[g].append(d["p"][ix])
    if not all(xs):
        raise ValueError("train/valid 不足，需增加整靴數")
    return [(np.concatenate(xs[g]),np.concatenate(ys[g]),np.concatenate(ps[g])) for g in (0,1)],counts

def metrics(y,p):
    picked=np.maximum(1e-12,p[np.arange(len(y)),y])
    ll=-float(np.log(picked).mean())
    one=np.eye(3)[y]
    brier=float(np.sum((p-one)**2,axis=1).mean())
    return {"log_loss":ll,"brier":brier}

def scores(prob):
    return np.column_stack((0.95*prob[:,0]-prob[:,1],
                            prob[:,1]-prob[:,0]))

def returns(y,decision):
    # -1 表示觀望，0 莊，1 閒；和局不輸不贏。
    r=np.zeros(len(y),dtype=np.float64)
    b=decision==0;p=decision==1
    r[b & (y==0)]=0.95
    r[b & (y==1)]=-1.
    r[p & (y==1)]=1.
    r[p & (y==0)]=-1.
    return r

def decisions(prob,threshold=-math.inf,require_positive=True):
    ev=scores(prob);side=ev.argmax(axis=1)
    maximum=ev[np.arange(len(ev)),side]
    accepted=maximum>=threshold
    if require_positive:
        accepted &= maximum>0
    return np.where(accepted,side,-1).astype(np.int8)

def pooled(rows):
    # rows: (靴, sum_payout, count_bets, sum_logloss, hands, wins)
    a=np.asarray(list(rows.values()),dtype=float)
    count=a[:,2].sum();num=a[:,1].sum()
    return {"ev_per_bet":float(num/count) if count else None,
            "ev_per_hand":float(num/a[:,4].sum()),
            "bet_rate":float(count/a[:,4].sum()),
            "hit_rate_ex_tie":float(a[:,5].sum()/a[:,6].sum()) if a[:,6].sum() else None,
            "log_loss":float(a[:,3].sum()/a[:,4].sum()),
            "bets":int(count),"hands":int(a[:,4].sum()),"shoes":len(rows)}

def shoe_rows(shoes,y,p,dec):
    r=returns(y,dec)
    ll=-np.log(np.maximum(1e-12,p[np.arange(len(y)),y]))
    rows={}
    for sid in np.unique(shoes):
        mask=shoes==sid;d=dec[mask];s=(d>=0)
        won=(d==0)&(y[mask]==0)|(d==1)&(y[mask]==1)
        rows[int(sid)]=[int(sid),float(r[mask].sum()),int(s.sum()),
                        float(ll[mask].sum()),int(mask.sum()),int(won.sum()),
                        int((s&(y[mask]!=2)).sum())]
    return rows

def bootstrap(rows,seed=SEED,n=BOOTSTRAPS):
    a=np.asarray(list(rows.values()),dtype=np.float64)
    if len(a)<20 or a[:,2].sum()==0:
        return None
    rng=np.random.default_rng(seed)
    outcomes=[]
    # 同一靴完整重抽；每靴所有下注與結果一起抽樣。
    for _ in range(n):
        sub=a[rng.integers(0,len(a),len(a))]
        bet=sub[:,2].sum()
        if bet:
            outcomes.append(sub[:,1].sum()/bet)
    if len(outcomes)<n*.95:
        return None
    return [float(x) for x in np.quantile(outcomes,[.025,.975])]

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def stage1(args):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    (tr,va),counts=collect_train_valid(args.data,args.seed)
    xtr,ytr,_=tr;xva,yva,_=va
    if len(np.unique(ytr))!=3:
        raise ValueError("訓練集缺少 B/P/T 類別")
    # 固定模型/訓練超參數；不讀取測試資料選模型或參數。
    model=make_pipeline(StandardScaler(),
                       MLPClassifier(hidden_layer_sizes=(256,128),
                         activation="relu",alpha=0.001,
                         early_stopping=True,n_iter_no_change=7,
                         max_iter=80,random_state=args.seed))
    model.fit(xtr,ytr)
    val=model.predict_proba(xva)
    ev=scores(val).max(axis=1)
    possible=[float(np.quantile(ev,1-q)) for q in QUANTILES]
    policies=[]
    for q,t in zip(QUANTILES,possible):
        d=decisions(val,t)
        n=(d>=0).sum()
        gain=returns(yva,d).sum()/max(1,n)
        # 門檻只用驗證資料選擇；至少 200 筆下注防止樣本太小。
        if n>=200:
            policies.append((float(gain),-q,t,q,int(n)))
    if not policies:
        threshold=math.inf;selected="none"
    else:
        best=max(policies)
        threshold=float(best[2]);selected=float(best[3])
    # 真正 OOS：固定模型及閾值後，逐 shard 只讀測試靴。
    evals={name:{} for name in ("prior","outcome_model","composition_oracle","always_banker")}
    curves={str(q):{} for q in QUANTILES}
    test_count=0
    prior=np.asarray(exact_distribution(INITIAL)[:3],dtype=float)
    for chunk in chunks(args.data):
        mask=np.fromiter((split(i,args.seed)==2 for i in chunk["shoe"]),
                         dtype=bool,count=len(chunk["shoe"]))
        if not mask.any():
            continue
        test_count+=int(mask.sum())
        x,y,p,sid=chunk["x"][mask],chunk["y"][mask],chunk["p"][mask],chunk["shoe"][mask]
        pred=model.predict_proba(x)
        scenarios={
            "prior":(np.tile(prior,(len(y),1)),decisions(np.tile(prior,(len(y),1)))),
            "outcome_model":(pred,decisions(pred,threshold)),
            "composition_oracle":(p,decisions(p)),
            "always_banker":(np.tile(prior,(len(y),1)),np.zeros(len(y),dtype=np.int8))
        }
        for name,(prob,dec) in scenarios.items():
            evals[name].update(shoe_rows(sid,y,prob,dec))
        for q,t in zip(QUANTILES,possible):
            curves[str(q)].update(shoe_rows(sid,y,pred,decisions(pred,t)))
    if not test_count:
        raise ValueError("測試靴不足")
    report={"seed":args.seed,"split":"SHA256(seed:shoe_id), 70/15/15",
            "counts":counts.tolist(),"train_sample":len(ytr),"valid_sample":len(yva),
            "test_hands":test_count,"validation_selected_rate":selected,
            "threshold_from_validation":threshold if math.isfinite(threshold) else None,
            "models":{},"curves":{}}
    for key,rows in evals.items():
        report["models"][key]={**pooled(rows),"ev_ci95":bootstrap(rows,args.seed)}
    for key,rows in curves.items():
        report["curves"][key]={**pooled(rows),"ev_ci95":bootstrap(rows,args.seed)}
    # Gate 1 不將無觀察到正值偷換成「數學已證明不可能」。
    target=report["models"]["outcome_model"]
    ci=target["ev_ci95"]
    passed=bool(ci and ci[0]>0)
    report["gate_1"]={"passed":passed,"reason":"CI lower>0" if passed else
      "沒有證明正期望；停止階段二/三；必須繼續階段四/五",
      "note":"有限模型的實測值不是資訊理論上限；正值仍需獨立重複驗證"}
    dest=Path(args.reports);write_json(dest/"stage1.json",report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys=list(report["curves"])
        vals=[report["curves"][k]["ev_per_bet"] for k in keys]
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot([100*float(k) for k in keys],
                [np.nan if v is None else 100*v for v in vals],marker="o")
        ax.axhline(0,color="black",linewidth=.8)
        ax.axhline(-1.06,linestyle="--",label="莊固定賭注理論約 -1.06%")
        ax.axhline(-1.24,linestyle=":",label="閒固定賭注理論約 -1.24%")
        ax.set(xlabel="驗證集預設最高下注比例(%)",ylabel="測試集每注 ROI(%)",
               title="結果紀錄模型：下注頻率對扣佣報酬")
        ax.legend();fig.tight_layout();fig.savefig(dest/"ev_curve.png",dpi=160);plt.close(fig)
    except ImportError:
        pass
    (dest/"stage1.md").write_text(
      "# 第一階段報告\n\n"+json.dumps(report,ensure_ascii=False,indent=2)+
      "\n\n注意：此為模型可達實證基準而非嚴格理論上界。\n",encoding="utf-8")
    print(json.dumps(report["gate_1"],ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("generate")
    a.add_argument("--out",default="research/data")
    a.add_argument("--shoes",type=int,default=200000)
    a.add_argument("--start-shoe",type=int,default=0)
    a.add_argument("--shard-size",type=int,default=250)
    a.add_argument("--cut",type=int,default=14)
    a.add_argument("--seed",type=int,default=SEED)
    b=sub.add_parser("stage1")
    b.add_argument("--data",default="research/data")
    b.add_argument("--reports",default="research/reports")
    b.add_argument("--seed",type=int,default=SEED)
    args=p.parse_args()
    if args.command=="generate":
        generate(args)
    else:
        stage1(args)
if __name__=="__main__":
    main()
