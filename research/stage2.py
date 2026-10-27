"""階段二：必須先通過 Gate 1；固定架構 213→64→32→48，硬/軟標籤成對比較。"""
import json
import math
from pathlib import Path
import numpy as np
from .baccarat import SEED
from .run import QUANTILES, chunks, split, metrics, scores, decisions, returns, shoe_rows, pooled, bootstrap, write_json

def _sample(data,seed,group,rate=.025):
    # 從整靴分組內按固定種子抽取局；標籤只來自該局開始前的真實牌組。
    rng=np.random.default_rng(seed+group)
    xs=[];ys=[];ps=[];aux=[];point_probs=[];point_means=[]
    for d in chunks(data):
        use=np.asarray([split(i,seed)==group for i in d["shoe"]])
        use &= rng.random(len(use))<rate
        if use.any():
            xs.append(d["x"][use]);ys.append(d["y"][use])
            ps.append(d["p"][use]);aux.append(d["aux"][use])
            point_probs.append(d["point_probs"][use]);point_means.append(d["point_means"][use])
    if not xs:raise ValueError("階段二訓練/驗證靴樣本不足")
    return [np.concatenate(a) for a in (xs,ys,ps,aux,point_probs,point_means)]

def _train(samples,seed,soft):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    net=nn.Sequential(nn.Linear(213,64),nn.ReLU(),
                      nn.Linear(64,32),nn.ReLU(),nn.Linear(32,48))
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    x,y,p,aux,point_probs,point_means=samples
    batch=2048
    xx=torch.from_numpy(x).float()
    yy=torch.from_numpy(y.astype(np.int64))
    pp=torch.from_numpy(p).float()
    aa=torch.from_numpy(aux).float()
    vv=torch.from_numpy(point_probs).float()
    mm=torch.from_numpy(point_means).float()
    for epoch in range(25):
        # 固定亂數 shuffle，只從訓練組取樣
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch))
        for ix in order.split(batch):
            z=net(xx[ix])
            if soft:
                main=-(pp[ix]*torch.log_softmax(z[:,23:26],dim=1)).sum(dim=1).mean()
            else:
                main=nn.functional.cross_entropy(z[:,23:26],yy[ix])
            aux_loss=0.
            for sl,gt in ((slice(0,3),slice(0,3)),
                          (slice(3,13),slice(3,13)),
                          (slice(13,23),slice(13,23))):
                aux_loss+=-(aa[ix,gt]*torch.log_softmax(z[:,sl],dim=1)).sum(dim=1).mean()
            # 26..35: research-only 十點數類別輔助 head；不宣稱是正式 rank ABI。
            point_loss=-(vv[ix]*torch.log_softmax(z[:,26:36],dim=1)).sum(dim=1).mean()
            mean_loss=((torch.sigmoid(z[:,47])*9-mm[ix])**2).mean()
            opt.zero_grad();(main+.2*aux_loss+.1*point_loss+.01*mean_loss).backward();opt.step()
    net.eval()
    return net

def _predict(net,x):
    import torch
    outs=[]
    with torch.no_grad():
        for batch in np.array_split(x,max(1,math.ceil(len(x)/8192))):
            if len(batch):
                outputs=net(torch.from_numpy(batch).float())[:,23:26]
                outs.append(torch.softmax(outputs,dim=1).numpy())
    return np.concatenate(outs)

def _threshold(y,p):
    expected=scores(p).max(axis=1)
    choices=[]
    for q in QUANTILES:
        t=float(np.quantile(expected,1-q))
        d=decisions(p,t)
        count=int((d!=-1).sum())
        if count>=200:
            choices.append((float(returns(y,d).sum()/count),-q,t,q))
    if not choices:
        return math.inf,None
    z=max(choices)
    return z[2],z[3]

def stage2(data="research/data",reports="research/reports",seed=SEED):
    dest=Path(reports)
    g=json.loads((dest/"stage1.json").read_text(encoding="utf-8"))["gate_1"]
    if not g["passed"]:
        write_json(dest/"stage2.json",{"status":"SKIPPED_GATE_1","seed":seed})
        return
    train=_sample(data,seed,0);valid=_sample(data,seed,1,rate=.10)
    soft=_train(train,seed,soft=True)
    hard=_train(train,seed,soft=False)
    from .reference import PhysicsReference
    from .physics_posterior_v2 import PosteriorTeacher
    reference=PhysicsReference()
    teacher=PosteriorTeacher(context_length=5,prior_strength=200).fit(train[0],train[2],train[4])
    models={
        "soft_MLP":lambda x:_predict(soft,x),
        "hard_MLP":lambda x:_predict(hard,x),
        "physics_direct_only":reference.predict_proba,
        "posterior_teacher_BPT_only":teacher.predict_proba
    }
    thresholds={}
    for name,predict in models.items():
        thresholds[name]=_threshold(valid[1],predict(valid[0]))
    outputs={k:{} for k in models}
    yall=[];probs={k:[] for k in models};idsall=[]
    for d in chunks(data):
        use=np.asarray([split(i,seed)==2 for i in d["shoe"]])
        if not use.any():continue
        x,y,ids=d["x"][use],d["y"][use],d["shoe"][use]
        yall.append(y);idsall.append(ids)
        for name,predict in models.items():
            p=predict(x);probs[name].append(p)
            outputs[name].update(shoe_rows(ids,y,p,decisions(p,thresholds[name][0])))
    y=np.concatenate(yall);sid=np.concatenate(idsall)
    probs={k:np.concatenate(v) for k,v in probs.items()}
    report={"seed":seed,"status":"COMPLETED","models":{},"comparisons":{},
      "note":"Physics 比較器只重現直接 48D head，非正式 PF/EMA/決策整體鏈。"}
    for name,p in probs.items():
        report["models"][name]={**metrics(y,p),**pooled(outputs[name]),
             "ev_ci95":bootstrap(outputs[name],seed),
             "validation_quantile":thresholds[name][1]}
    # 同一測試靴配對重抽，估計兩模型每局 net profit 差，避免不同下注數扭曲 ROI。
    rng=np.random.default_rng(seed)
    unique=np.unique(sid)
    for name in ("hard_MLP","physics_direct_only","posterior_teacher_BPT_only"):
        r1=outputs["soft_MLP"];r0=outputs[name]
        diffs=np.asarray([r1[int(s)][1]-r0[int(s)][1] for s in unique])
        hands=np.asarray([r1[int(s)][4] for s in unique])
        sims=[]
        for _ in range(1500):
            j=rng.integers(0,len(unique),len(unique))
            sims.append(float(diffs[j].sum()/hands[j].sum()))
        ci=np.quantile(sims,[.025,.975])
        report["comparisons"]["soft_minus_"+name]={
            "profit_per_hand_ci95":ci.tolist(),
            "profit_per_hand_significant_at_95pct":bool(ci[0]>0 or ci[1]<0),
            "profit_per_hand_improved_significantly":bool(ci[0]>0),
            "profit_per_hand_mean":float(diffs.sum()/hands.sum()),
            "log_loss_delta":report["models"]["soft_MLP"]["log_loss"]-
                             report["models"][name]["log_loss"]}
    report["posterior_teacher"]={"validation":teacher.diagnose(valid[0]),"test":teacher.diagnose(np.concatenate([d["x"][np.asarray([split(i,seed)==2 for i in d["shoe"]])] for d in chunks(data) if np.any([split(i,seed)==2 for i in d["shoe"]])])) if len(y) else None}
    write_json(dest/"stage2.json",report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for name,p in probs.items():
        conf=p[:,0];edges=np.quantile(conf,np.linspace(0,1,11))
        centers=[];freq=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(conf>=lo)&(conf<=hi)
            if mask.any():
                centers.append(float(conf[mask].mean()))
                freq.append(float((y[mask]==0).mean()))
        ax.plot(centers,freq,marker=".",label=name)
    ax.plot([0,1],[0,1],"k--",linewidth=.8)
    ax.set(xlabel="預測 P(B)",ylabel="實際莊比例",title="整靴測試集校準曲線")
    ax.legend();fig.tight_layout();fig.savefig(dest/"calibration.png",dpi=160);plt.close(fig)
