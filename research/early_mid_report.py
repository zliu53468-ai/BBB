"""Single500-shoe comparison: fixed v1 vs v2, with live-main reference."""
import json,hashlib,math,sys
from pathlib import Path
import numpy as np
from .priority_report import blank,metrics,runs,streak_report
V=['main','before','after'];FILTERS={'all':lambda n:True,'1-30':lambda n:n<=30,'31-45':lambda n:31<=n<=45,'1-45':lambda n:n<=45,'46-70':lambda n:n>=46}
def main():
    d=Path(sys.argv[1]);p=json.loads((d/'protocol.json').read_text());assert json.loads((d/'progress.json').read_text())['complete']
    totals={s:{v:blank() for v in V} for s in FILTERS};tables={h:{v:blank() for v in V} for h in [50,60,70]};sequences={};groups=[];within={v:{True:{},False:{}} for v in V};reasons={};checks=0
    import collections
    within={v:{True:collections.Counter(),False:collections.Counter()} for v in V}
    for line in (d/'predictions.jsonl').open():
        shoe=json.loads(line);sid=shoe['shoe_id'];assert sid not in sequences;seq={v:[] for v in V};local={s:{v:blank() for v in V} for s in FILTERS}
        for i,r in enumerate(shoe['rows']):
            assert r['round']==i+1
            if i<30:
                assert r['pf_weight']<=.08+1e-12 and r['alpha']<=.40 and abs(r['residual']['applied_delta'])<=.015+1e-12 and r['calibration']['delta']==0;checks+=1
            for reason in r['reasons']:reasons[reason]=reasons.get(reason,0)+1
            for v in V:
                side=r[v];actual=r['actual'];bet=side in ['B','P'];win=bet and actual==side;loss=bet and actual!='T' and not win;push=bet and actual=='T';profit=(.95 if side=='B' else 1) if win else -1 if loss else 0
                dest=[z[v] for s,z in totals.items() if FILTERS[s](i+1)]+[z[v] for s,z in local.items() if FILTERS[s](i+1)]+[z[v] for h,z in tables.items() if i<h]
                for z in dest:
                    z['hands']+=1;z['recommendations']+=int(bet);z['wins']+=int(win);z['losses']+=int(loss);z['pushes']+=int(push);z['profit']+=profit
                    if bet:z[side]+=1
                if win or loss:seq[v].append(bool(win))
        for v in V:
            h=runs(seq[v])
            for w in [True,False]:within[v][w].update(h[w])
        sequences[sid]=seq;groups.append((sid,local))
    assert len(sequences)==500
    stages={s:{v:metrics(x) for v,x in z.items()} for s,z in totals.items()};horizons={str(h):{v:metrics(x) for v,x in z.items()} for h,z in tables.items()}
    streaks={}
    for v in V:
        seq=[x for sid in sorted(sequences) for x in sequences[sid][v]];global_hist=runs(seq)
        streaks[v]={'within_shoe':{'win':streak_report(within[v][True],len(seq)),'loss':streak_report(within[v][False],len(seq))},'across_shoes':{'win':streak_report(global_hist[True],len(seq)),'loss':streak_report(global_hist[False],len(seq))}}
    ci={};rng=np.random.default_rng(2026102922)
    for s in FILTERS:
        a=np.asarray([[[g[s][v]['wins'],g[s][v]['losses']] for v in ['main','before','after']] for _,g in sorted(groups)])
        samples=[]
        for _ in range(2000):
            z=a[rng.integers(len(a),size=len(a))].sum(axis=0);n=z.sum(axis=1);hit=np.divide(z[:,0],n,out=np.full(3,np.nan),where=n>0);samples.append([hit[2],hit[2]-hit[1],hit[2]-hit[0]])
        out=[]
        for column in np.asarray(samples).T:
            valid=column[np.isfinite(column)];out.append(np.quantile(valid,[.025,.975]).tolist() if len(valid) else None)
        ci[s]={'after_hit95':out[0],'paired_hit_change95':out[1],'paired_hit_change_vs_main95':out[2]}
    report={'protocol':p,'synthetic_only':True,'stages':stages,'horizons':horizons,'streaks':streaks,'whole_shoe_bootstrap':ci,'early_cap_checks_passed':checks,'skip_reasons':reasons,'raw_sha256':hashlib.sha256((d/'predictions.jsonl').read_bytes()).hexdigest(),'deployed':False,'limitations':['500 shoes only; low action counts cannot establish improved accuracy.','Points not provided in this comparison; early point posterior disabled by design.','Push ties and Skip do not erase the placed losing streak. All stakes flat1, Banker commission5%.','Primary horizon70;50/60 are prespecified causal prefix checks, not additional independent shoes.']}
    (d/'statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    fmt=lambda x:'—（無有效推薦）' if x is None else f'{x*100:.3f}%'
    md='# 前中段保守修正｜500靴配對驗證\n\n全新seed2026102921、500靴／35,000手；每桌最多70手，另評估同一批前50／60手。固定參數，一次評估；不提供真實牌階／當手尚未公布結果。主比較為上一候選PR73(116ffc9)對新v2，另列正式main(c6f4a0a)，三者不是同一版本。\n\n|範圍|修改前推薦數／出手率|修改後推薦數／出手率|前命中率|後命中率|\n|---|---:|---:|---:|---:|\n'
    for s,z in stages.items():
        b=z['before'];a=z['after'];md+=f"|{s}|{b['recommendations']}／{fmt(b['action_rate'])}|{a['recommendations']}／{fmt(a['action_rate'])}|{fmt(b['hit_on_recommendations_excluding_ties'])}|{fmt(a['hit_on_recommendations_excluding_ties'])}|\n"
    md+='\n## 正式main參考（不是前版候選）\n\n|範圍|main出手率／命中率|v2出手率／命中率|\n|---|---:|---:|\n'
    for s,z in stages.items():
        m=z['main'];a=z['after'];md+=f"|{s}|{fmt(m['action_rate'])}／{fmt(m['hit_on_recommendations_excluding_ties'])}|{fmt(a['action_rate'])}／{fmt(a['hit_on_recommendations_excluding_ties'])}|\n"
    md+='\n## 全程連錯與EV\n\n|版本|全程最大連錯（不清零）|靴內最大連錯|EV／推薦|莊／閒推薦數|\n|---|---:|---:|---:|---:|\n'
    for v,name in [('main','目前正式main'),('before','上一候選PR73'),('after','新保守v2')]:
        x=stages['all'][v];md+=f"|{name}|{streaks[v]['across_shoes']['loss']['maximum']}|{streaks[v]['within_shoe']['loss']['maximum']}|{fmt(x['EV_per_recommendation'])}|{x['B']}／{x['P']}|\n"
    md+='\n## 50／60／70手環境\n\n|桌長|修改前出手率／命中率|修改後出手率／命中率|\n|---|---:|---:|\n'
    for h,z in horizons.items():
        a=z['after'];b=z['before'];md+=f"|{h}|{fmt(b['action_rate'])}／{fmt(b['hit_on_recommendations_excluding_ties'])}|{fmt(a['action_rate'])}／{fmt(a['hit_on_recommendations_excluding_ties'])}|\n"
    md+='\n## 不確定性與解讀\n\n'
    for s,x in ci.items():
        md+=f"- {s}：後命中率整靴95%區間 {x['after_hit95']}；與前版配對差區間 {x['paired_hit_change95']}（JSON數值為0–1）。\n"
    md+=f'\n前30局硬性PF／alpha／Residual／校準限制共驗證{checks:,}手。觀望不計勝負、和局退回不計命中率分母；不把沒有出手稱作100%命中。門檻變嚴只代表篩選更少，不保證篩選更準；本次不自動合併／部署。完整原始統計及預註冊參數見同目錄JSON。\n'
    (d/'REPORT.zh-TW.md').write_text(md,encoding='utf-8')
    print(json.dumps({'stages':stages,'global_losses':{v:streaks[v]['across_shoes']['loss']['maximum'] for v in V},'ci':ci},ensure_ascii=True))
if __name__=='__main__':main()
