"""Per-shoe paired audit, with recommendation-sequence and clock-time streaks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
VERSIONS=['next','old','always_B','always_B_on_new_schedule']
STAGES={'all':lambda n:True,'1-20':lambda n:n<=20,'21-50':lambda n:21<=n<=50,'51+':lambda n:n>=51}
def blank():return dict(hands=0,recommendations=0,wins=0,losses=0,pushes=0,profit=0.,B=0,P=0)
def metrics(a):
    total=a['hands'];bets=a['recommendations'];n=a['wins']+a['losses']
    return dict(**a,skip=total-bets,action_rate=bets/total if total else None,hit_on_recommendations_excluding_ties=a['wins']/n if n else None,wins_per_all_hands=a['wins']/total if total else None,EV_per_recommendation=a['profit']/bets if bets else None,EV_per_opportunity=a['profit']/total if total else None)
def runs(seq):
    out={True:collections.Counter(),False:collections.Counter()};last=None;length=0
    for value in seq+[None]:
        if value is not None and value==last:length+=1
        else:
            if last is not None:out[last][length]+=1
            last=value;length=int(value is not None)
    return out
def streak_report(counter,non_tie):
    return dict(maximum=max(counter,default=0),histogram=dict(sorted(counter.items())),thresholds={str(k):dict(maximal_run_count=sum(v for length,v in counter.items() if length>=k),mean_run_length=sum(length*v for length,v in counter.items() if length>=k)/sum(v for length,v in counter.items() if length>=k) if sum(v for length,v in counter.items() if length>=k) else None,overlapping_windows=sum((length-k+1)*v for length,v in counter.items() if length>=k),windows_per_1000_non_tie_recommendations=1000*sum((length-k+1)*v for length,v in counter.items() if length>=k)/non_tie if non_tie else None) for k in [3,5,7]})
def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory)
    protocol=json.loads((d/'protocol.json').read_text());status=json.loads((d/'progress.json').read_text());assert status['phase']=='complete'
    totals={s:{v:blank() for v in VERSIONS} for s in STAGES};groups=[];hist={v:{'placed':{True:collections.Counter(),False:collections.Counter()},'clock':{True:collections.Counter(),False:collections.Counter()}} for v in VERSIONS};reasons=collections.Counter();seen=set();across={};common=dict(opportunities=0,conflicting_directions=0,new_wins=0,old_wins=0)
    for line in (d/'predictions.jsonl').open():
        shoe=json.loads(line);assert shoe['shoe_id'] not in seen;seen.add(shoe['shoe_id']);local={v:blank() for v in VERSIONS};sequences={v:{'placed':[],'clock':[]} for v in VERSIONS}
        for i,row in enumerate(shoe['rows']):
            assert row['round']==i+1;actual=row['actual'];directions={'next':row['next'],'old':row['old'],'always_B':'B','always_B_on_new_schedule':'B' if row['next'] in 'BP' else 'Skip'}
            reasons.update(row['reasons'])
            if directions['next'] in ['B','P'] and directions['old'] in ['B','P']:
                common['opportunities']+=1;common['conflicting_directions']+=int(directions['next']!=directions['old']);common['new_wins']+=int(directions['next']==actual);common['old_wins']+=int(directions['old']==actual)
            for v,side in directions.items():
                bet=side in ['B','P'];win=bet and side==actual;loss=bet and actual!='T' and not win;push=bet and actual=='T';profit=(.95 if side=='B' else 1.) if win else -1. if loss else 0.
                for dst in [local[v]]+[totals[s][v] for s,f in STAGES.items() if f(i+1)]:
                    dst['hands']+=1;dst['recommendations']+=int(bet);dst['wins']+=int(win);dst['losses']+=int(loss);dst['pushes']+=int(push);dst['profit']+=profit
                    if bet:dst[side]+=1
                if win or loss:sequences[v]['placed'].append(bool(win));sequences[v]['clock'].append(bool(win))
                else:sequences[v]['clock'].append(None)
        for v in VERSIONS:
            for scope in ['placed','clock']:
                h=runs(sequences[v][scope])
                for win in [True,False]:hist[v][scope][win].update(h[win])
        across[shoe['shoe_id']]={v:sequences[v]['placed'] for v in VERSIONS}
        groups.append((shoe['shoe_id'],[[local[v][k] for k in ['hands','recommendations','wins','losses','profit']] for v in VERSIONS]))
    assert len(seen)==protocol['shoes'];result={s:{v:metrics(z) for v,z in group.items()} for s,group in totals.items()}
    streaks={v:{scope:{'winning':streak_report(counter[True],totals['all'][v]['wins']+totals['all'][v]['losses']),'losing':streak_report(counter[False],totals['all'][v]['wins']+totals['all'][v]['losses'])} for scope,counter in scopes.items()} for v,scopes in hist.items()}
    for v in VERSIONS:
        sequence=[x for sid in sorted(across) for x in across[sid][v]];h=runs(sequence)
        streaks[v]['chronological_across_shoes']={'winning':streak_report(h[True],len(sequence)),'losing':streak_report(h[False],len(sequence))}
    g=np.asarray([row for sid,row in sorted(groups,key=lambda x:x[0])],dtype=float);rng=np.random.default_rng(2026102914);samples=[]
    for _ in range(2000):
        z=g[rng.integers(len(g),size=len(g))].sum(axis=0);non_tie=z[:,2]+z[:,3];hit=np.divide(z[:,2],non_tie,out=np.full(4,np.nan),where=non_tie>0);ev=z[:,4]/z[:,0];evbet=np.divide(z[:,4],z[:,1],out=np.full(4,np.nan),where=z[:,1]>0)
        samples.append([hit[0],hit[0]-hit[1],hit[0]-hit[3],ev[0]-ev[1],evbet[0]-evbet[1]])
    ci=np.nanquantile(samples,[.025,.975],axis=0).T.tolist();n=result['all']['next'];old=result['all']['old'];fail=[]
    if not .20<=n['action_rate']<=.35:fail.append('出手率未落在建議20–35%')
    if n['hit_on_recommendations_excluding_ties'] is None or old['hit_on_recommendations_excluding_ties'] is None or n['hit_on_recommendations_excluding_ties']<old['hit_on_recommendations_excluding_ties']:fail.append('條件命中率未優於舊版或推薦不足以估計')
    if not np.isfinite(ci[1][0]) or ci[1][0]<=0:fail.append('整靴配對區間未證實條件命中率提升')
    if not np.isfinite(ci[2][0]) or ci[2][0]<=0:fail.append('未證實優於相同推薦時機的固定莊')
    if n['EV_per_recommendation'] is None or old['EV_per_recommendation'] is None or n['EV_per_recommendation']<old['EV_per_recommendation']:fail.append('每次推薦扣佣EV未優於舊版或推薦不足以估計')
    if streaks['next']['placed']['losing']['maximum']>=streaks['old']['placed']['losing']['maximum']:fail.append('推薦序列最大連錯未下降')
    if streaks['next']['chronological_across_shoes']['losing']['maximum']>=streaks['old']['chronological_across_shoes']['losing']['maximum']:fail.append('跨靴連續推薦最大連錯未下降')
    if (streaks['next']['chronological_across_shoes']['losing']['thresholds']['3']['windows_per_1000_non_tie_recommendations'] or 0)>=(streaks['old']['chronological_across_shoes']['losing']['thresholds']['3']['windows_per_1000_non_tie_recommendations'] or 0):fail.append('跨靴每千次非和推薦的三連錯頻率未下降')
    if n['recommendations']<1000 and protocol['shoes']>=10000:fail.append('推薦數太少，不能宣稱穩定改善')
    report=dict(schema_version=1,data_kind='synthetic_only',protocol=protocol,shoes=len(seen),all_hands_and_recommendations=result,streaks=streaks,common_recommendations=common,skip_reasons=dict(reasons),whole_shoe_bootstrap=dict(samples=2000,seed=2026102914,new_hit95=ci[0],conditional_hit_change_vs_old95=ci[1],conditional_hit_change_vs_scheduled_B95=ci[2],EV_per_opportunity_change_vs_old95=ci[3],EV_per_recommendation_change_vs_old95=ci[4]),gate=dict(passed=not fail,failures=fail,production_deployed=False),notes=['Placed streaks ignore Skip and push ties. Cooldown does not reset the recommended losing sequence.','Clock streaks break at any Skip or tie; do not use this alone to claim fewer losses.','Threshold event counts are maximal runs; overlapping windows and rate/1000 resolved recommendations are also reported.','Maximum streaks are within a shoe, not concatenated across independent shoes.','Full holdout is BPT button mode only. Point-mode functionality is tested separately, not claimed as a 10000-shoe point-mode benefit.','No ground-truth ranks/composition/points/card counts are fed to this BPT holdout.','Original PF may exhaust hypotheses near a14-card cut; new gate fails closed on missing ESS. Legacy behavior is preserved for comparison.'])
    report['notes'].append('EV uses a flat1 unit for every recommendation in both policies, not the legacy variable stake multiplier.')
    report['notes'].append('chronological_across_shoes additionally concatenates placed non-tie recommendations by generation shoe_id, WITHOUT resetting at shoe boundaries. Runtime protection itself still resets per shoe.')
    import platform
    report['environment']={'python':platform.python_version(),'numpy':np.__version__}
    report['reporter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest=hashlib.sha256()
    with (d/'predictions.jsonl').open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    report['raw_predictions_sha256']=digest.hexdigest()
    report['simulation']={'seed':int(next(iter(seen)).split('-')[0]),'decks':8,'cut_from_end':14,'burn':'reveal1 + burn point value; zero=10','full_shoes':True,'max_hands_cap':None,'latent_card_composition_input':False,'point_input_in_holdout':False}
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    import math
    report=clean(report)
    (d/'statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    labels={'next':'新純算牌閘門','old':'舊正式完整runtime','always_B':'每手固定莊','always_B_on_new_schedule':'相同新推薦時機固定莊'}
    fmt=lambda x:'—' if x is None or not math.isfinite(x) else f'{x*100:.4f}%'
    md='# 純算牌優先：完整牌靴、選擇性與連勝連敗回測\n\n'+f"{len(seen):,}靴，{n['hands']:,}手。8副牌、揭牌後燒牌、靴底切14張、完整結束跨切牌的一手。只輸入已公布B/P/T；不輸入模擬真實牌組、點數或張數。沒有重訓／替換模型權重，未部署。\n\n"
    md+='以下每次推薦統一平注1單位、莊勝扣5%佣金、和局退回；不套用舊版變動stakeMultiplier，因此是方向／時機的平注對比，不是個人實盤收益。\n\n## 全部手與有推薦手\n\n|版本|推薦數／出手率|推薦命中率（排除和）|勝局／全部手|扣佣EV／推薦|扣佣EV／機會|靴內最大連勝／連錯|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in VERSIONS:
        z=result['all'][v];ss=streaks[v]['placed'];md+=f"|{labels[v]}|{z['recommendations']:,}／{fmt(z['action_rate'])}|{fmt(z['hit_on_recommendations_excluding_ties'])}|{fmt(z['wins_per_all_hands'])}|{fmt(z['EV_per_recommendation'])}|{fmt(z['EV_per_opportunity'])}|{ss['winning']['maximum']}／{ss['losing']['maximum']}|\n"
    md+=f"\n新版本命中率整靴95%區間：{fmt(ci[0][0])}–{fmt(ci[0][1])}；與舊版配對命中率差：{fmt(ci[1][0])}–{fmt(ci[1][1])}（此處差為百分點）。\n\n"
    md+='## 連續推薦的連勝／連錯（觀望不清零）\n\n|版本|類型|≥3次數／平均長度|≥5次數／平均長度|≥7次數／平均長度|三連窗口／每千次非和推薦|\n|---|---|---:|---:|---:|---:|\n'
    for v in VERSIONS:
        for kind,name in [('winning','連勝'),('losing','連錯')]:
            t=streaks[v]['placed'][kind]['thresholds'];fields=[f"{t[str(k)]['maximal_run_count']}／{t[str(k)]['mean_run_length']:.2f}" if t[str(k)]['mean_run_length'] is not None else '0／—' for k in [3,5,7]];md+=f"|{labels[v]}|{name}|"+'|'.join(fields)+f"|{t['3']['overlapping_windows']}／{t['3']['windows_per_1000_non_tie_recommendations'] or 0:.3f}|\n"
    md+='\n次數為極大連續段數；一段7連錯計為一段≥3、一段≥5、一段≥7。另報重疊窗口，避免定義含混。和局退回、不計勝負；推薦序列略過和局與觀望，絕不因冷卻把連錯紀錄清零。時鐘序列遇觀望／和局中斷，另列JSON，不能單獨拿來宣稱改善。每靴重置。\n\n## 分段\n\n|階段|新出手率／命中率|舊出手率／命中率|\n|---|---:|---:|\n'
    for s in ['1-20','21-50','51+']:
        x=result[s]['next'];y=result[s]['old'];md+=f"|{s}|{fmt(x['action_rate'])}／{fmt(x['hit_on_recommendations_excluding_ties'])}|{fmt(y['action_rate'])}／{fmt(y['hit_on_recommendations_excluding_ties'])}|\n"
    md+='\n## 跨靴連續推薦（不在新靴清零的額外統計）\n\n按生成shoe_id排序串接所有非和推薦，觀望也不清零；不能把靴內最大7次誤稱為全程上限。實際runtime的風控仍每靴重置，以下是額外風險統計。\n\n|版本|全程最大連勝／連錯|≥3連錯段數／平均長度|三連錯重疊窗口／每千次推薦|\n|---|---:|---:|---:|\n'
    for v in VERSIONS:
        z=streaks[v]['chronological_across_shoes'];t=z['losing']['thresholds']['3'];md+=f"|{labels[v]}|{z['winning']['maximum']}／{z['losing']['maximum']}|{t['maximal_run_count']}／{t['mean_run_length'] or 0:.2f}|{t['overlapping_windows']}／{t['windows_per_1000_non_tie_recommendations'] or 0:.3f}|\n"
    md+='\n## 驗收與限制\n\n'+('研究指標通過；仍需真實獨立資料，不自動部署。\n' if not fail else '\n'.join('- '+v for v in fail)+'\n')
    md+='\n少出手自然減少連錯事件總數，必須同時看每千次推薦的三連錯與推薦序列最大連錯。冷卻只管理風險，沒有證明能預知下一局。20–35%為建議出手率，不能為湊比例放寬不確定性或EV閘門。點數PF為近似後驗、不是觀察到真實牌階；本次10000靴不包含點數模式效果驗證。此模擬有燒牌／切14張，與先前無燒牌、最多70手的實驗不能直接拼成同一獨立樣本。\n'
    (d/'REPORT.zh-TW.md').write_text(md,encoding='utf-8')
    try:
        os.environ['MPLCONFIGDIR']=str((d/'.mpl-cache').resolve());(d/'.mpl-cache').mkdir(exist_ok=True)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for column,kind in enumerate(['winning','losing']):
            for row,normal in enumerate([False,True]):
                ax=axes[row,column]
                for v in VERSIONS:
                    h=hist[v]['placed'][kind=='winning'];xs=sorted(h);resolved=totals['all'][v]['wins']+totals['all'][v]['losses'];scale=1000/resolved if normal and resolved else 1
                    ax.plot(xs,[h[x]*scale for x in xs],marker='o',label=v)
                ax.set(xlabel='Run length',ylabel='Runs per1000 resolved bets' if normal else 'Maximal runs',title='Within-shoe '+kind+' runs',yscale='log');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(d/'streak_distribution.png',dpi=160);plt.close(fig)
        with (d/'REPORT.zh-TW.md').open('a',encoding='utf-8') as stream:stream.write('\n## 靴內分布圖\n\n上排為原始段數，下排按每千次非和推薦正規化；跨靴全程指標另見上表及JSON，不以此圖替代。\n\n![靴內連勝與連錯分布](streak_distribution.png)\n')
    except ImportError:pass
    print(json.dumps(dict(shoes=len(seen),candidate=n,old=old,gate=report['gate']),ensure_ascii=True))
if __name__=='__main__':main()
