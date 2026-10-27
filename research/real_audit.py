"""階段四：真實牌靴與模擬序列檢定；沒有真實資料時僅輸出 BLOCKED，不杜撰。"""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from .baccarat import SEED, INITIAL, exact_distribution
from .run import chunks, returns, shoe_rows, bootstrap, pooled, write_json

def _read_real(path):
    if not path or not Path(path).exists():
        return None
    shoes={};order={}
    with open(path,encoding="utf-8-sig",newline="") as file:
        reader=csv.DictReader(file)
        needed={"shoe_id","round","outcome","shoe_start"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("真實資料必須有 shoe_id, round, outcome, shoe_start 四欄")
        for row in reader:
            sid=row["shoe_id"]
            o=row["outcome"].upper().strip()
            if o not in "BPT":raise ValueError("outcome 僅 B/P/T")
            n=int(row["round"])
            if n<=0:raise ValueError("round 必須 >=1")
            if not row["shoe_start"]:
                raise ValueError("時間樣本外評估需要鞋開始的 shoe_start ISO 時間")
            if sid in order and order[sid]!=row["shoe_start"]:
                raise ValueError("同一靴 shoe_start 必須相同")
            order[sid]=row["shoe_start"]
            shoes.setdefault(sid,[]).append((n,o,row))
    for sid,seq in shoes.items():
        seq.sort(key=lambda x:x[0])
        rounds=[x[0] for x in seq]
        if len(rounds)!=len(set(rounds)) or rounds!=list(range(1,len(rounds)+1)):
            raise ValueError(f"靴 {sid} 局數不連續，請補齊而非推測")
    return sorted(shoes.items(),key=lambda item:(order[item[0]],item[0]))

def _sim_sequences(folder,max_shoes=2000):
    if not folder or not Path(folder).exists():
        return None
    grouped={}
    for d in chunks(folder):
        for sid,y,r in zip(d["shoe"],d["y"],d["round"]):
            s=int(sid)
            if s not in grouped and len(grouped)>=max_shoes:continue
            grouped.setdefault(s,[]).append((int(r),"BPT"[int(y)],None))
    return [(str(k),sorted(v,key=lambda x:x[0])) for k,v in sorted(grouped.items())]

def _run_lengths(shoes):
    lengths=[]
    for _,seq in shoes:
        bp=[x[1] for x in seq if x[1]!="T"]
        if not bp:continue
        length=1
        for a,b in zip(bp,bp[1:]):
            if a==b:length+=1
            else:lengths.append(length);length=1
        lengths.append(length)
    return lengths

def _runs_z(shoes):
    runs=expect=var=0
    for _,seq in shoes:
        bp=[x[1] for x in seq if x[1]!="T"]
        b=bp.count("B");p=bp.count("P");n=b+p
        if b*p==0 or n<2:continue
        runs+=1+sum(a!=z for a,z in zip(bp,bp[1:]))
        expect+=1+2*b*p/n
        var+=2*b*p*(2*b*p-n)/(n*n*(n-1))
    return float((runs-expect)/math.sqrt(var)) if var>0 else None

def _autocorr(shoes):
    x=[];y=[]
    for _,seq in shoes:
        bp=[int(a[1]=="B") for a in seq if a[1]!="T"]
        x.extend(bp[:-1]);y.extend(bp[1:])
    return float(np.corrcoef(x,y)[0,1]) if len(x)>3 and np.std(x)>0 and np.std(y)>0 else None

def _summary(shoes):
    outcomes=Counter(token for _,seq in shoes for _,token,*_ in seq)
    stages={"early":Counter(),"middle":Counter(),"late":Counter()}
    for _,seq in shoes:
        for i,(_,token,*_) in enumerate(seq):
            group="early" if i<20 else "middle" if i<45 else "late"
            stages[group][token]+=1
    count=sum(outcomes.values())
    return {"shoes":len(shoes),"hands":count,
            "outcomes":{c:int(outcomes[c]) for c in "BPT"},
            "ratios":{c:outcomes[c]/count for c in "BPT"} if count else None,
            "run_lengths":_run_lengths(shoes),
            "runs_z":_runs_z(shoes),"autocorr_lag1_bp":_autocorr(shoes),
            "phase":{k:{v:int(c[v]) for v in "BPT"} for k,c in stages.items()},
            "shoe_lengths":[len(s) for _,s in shoes]}

def _tests(real,sim):
    from scipy.stats import chi2_contingency,ks_2samp,norm
    ra=np.asarray([real["outcomes"][c] for c in "BPT"])
    sa=np.asarray([sim["outcomes"][c] for c in "BPT"])
    _,pchi,_,_=chi2_contingency(np.vstack([ra,sa]))
    # KS 用於離散 run lengths 僅是探索性；應另以整靴區塊置換做穩健性。
    kr=ks_2samp(real["run_lengths"],sim["run_lengths"])
    kl=ks_2samp(real["shoe_lengths"],sim["shoe_lengths"])
    p_runs=2*norm.sf(abs(real["runs_z"])) if real["runs_z"] is not None else None
    phases=np.asarray([[real["phase"][stage][c] for c in "BPT"] for stage in ("early","middle","late")])
    active=phases.sum(axis=1)>0
    p_phase=float(chi2_contingency(phases[active])[1]) if active.sum()>1 else None
    tests={"chi_square_outcomes":float(pchi),"ks_run_lengths_exploratory":float(kr.pvalue),
           "ks_shoe_length":float(kl.pvalue),"runs_test":float(p_runs) if p_runs is not None else None,
           "phase_drift_real":p_phase}
    # Benjamini-Hochberg FDR = 0.05；不將未拒絕 H0 等同於證明相同。
    available=sorted((v,k) for k,v in tests.items() if v is not None)
    threshold=0
    for i,(p,_) in enumerate(available,1):
        if p<=.05*i/len(available):threshold=p
    rejected=[k for k,v in tests.items() if v is not None and v<=threshold]
    return {"p_values":tests,"fdr05_rejections":rejected,
           "no_significant_difference":len(rejected)==0}

def _backtest(shoes,seed):
    # 實際資料以完整靴依時間排序，最後 15% 為真正樣本外。
    count=len(shoes)
    start=max(1,int(math.floor(.85*count)))
    test=shoes[start:]
    if not test:return {"status":"INSUFFICIENT_SHOES"}
    rng=np.random.default_rng(seed)
    scenarios={key:{} for key in ("always_banker","random_direction","prior_no_bet","logged_system")}
    system_complete=True
    baseline=exact_distribution(INITIAL)[:3]
    for i,(_,hands) in enumerate(test):
        outcome=np.asarray(["BPT".index(row[1]) for row in hands],dtype=np.uint8)
        n=len(outcome)
        pred=np.tile(np.asarray(baseline),(n,1))
        sid=np.full(n,i,dtype=int)
        directions=np.zeros(n,dtype=np.int8)
        scenarios["always_banker"].update(shoe_rows(sid,outcome,pred,directions))
        directions=rng.integers(0,2,n,dtype=np.int8)
        scenarios["random_direction"].update(shoe_rows(sid,outcome,pred,directions))
        scenarios["prior_no_bet"].update(shoe_rows(sid,outcome,pred,np.full(n,-1,dtype=np.int8)))
        logged=[]
        for _,_,row in hands:
            direction=(row or {}).get("direction","").upper().strip()
            if direction not in ("B","P","SKIP"):
                system_complete=False
            logged.append({"B":0,"P":1,"SKIP":-1}.get(direction,-1))
        if system_complete:
            scenarios["logged_system"].update(shoe_rows(sid,outcome,pred,np.asarray(logged,dtype=np.int8)))
    result={"train_shoes":start,"test_shoes":len(test),"split":"shoe_start chronological 85/15","metrics":{}}
    for name,rows in scenarios.items():
        if name=="logged_system" and not system_complete:
            result["metrics"][name]={"status":"MISSING_PRE_BET_LOGS"};continue
        result["metrics"][name]={**pooled(rows),"ev_ci95":bootstrap(rows,seed)}
    return result

def stage4(real_csv=None,sim_dir="research/data",reports="research/reports",seed=SEED):
    dest=Path(reports);dest.mkdir(parents=True,exist_ok=True)
    real=_read_real(real_csv)
    if real is None:
        report={"status":"MISSING_REAL_DATA","message":"階段四必須執行檢查，但沒有按靴原始檔，不得虛構判斷",
                "required_columns":["shoe_id","round","outcome","shoe_start"],
                "optional_prebet_column":"direction (B/P/SKIP)"}
        write_json(dest/"stage4.json",report)
        return report
    sim=_sim_sequences(sim_dir)
    rs=_summary(real)
    ss=_summary(sim) if sim else None
    # 不在輸出塞入全長 run 序列，避免難以閱讀。
    report={"status":"COMPLETED_WITH_SIM" if ss else "PARTIAL_NO_SIM",
            "real":{k:v for k,v in rs.items() if k not in ("run_lengths","shoe_lengths")},
            "simulation":{k:v for k,v in ss.items() if k not in ("run_lengths","shoe_lengths")} if ss else None,
            "tests":_tests(rs,ss) if ss else None,
            "out_of_sample":_backtest(real,seed)}
    if ss:
        result=report["tests"]
        report["environment_conclusion"]=(
          "未觀察到牌組消耗效應；但無顯著差異不等於證明牌組真的連續消耗"
          if result["no_significant_difference"] else
          "真實數據與八副牌連續消耗模擬出現統計差異；需排查 CSM、重洗與切牌設定")
    else:
        report["environment_conclusion"]="沒有模擬資料，不能檢定牌組消耗假設"
    write_json(dest/"stage4.json",report)
    return report
