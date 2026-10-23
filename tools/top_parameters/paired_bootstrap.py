"""Paired whole-shoe intervals; no model/parameter fitting and no row shuffle."""
import argparse,json
from pathlib import Path
import numpy as np

STAGES={"all":lambda r:True,"1-20":lambda r:1<=r<=20,"21-45":lambda r:21<=r<=45,"46-70":lambda r:46<=r<=70,"<=40":lambda r:r<=40,"41-50":lambda r:41<=r<=50,">50":lambda r:r>50,"50-70":lambda r:50<=r<=70}
def compare(b,c,samples=2000,seed=20261023):
    key=lambda r:(r['shoe_id'],r['round_index'],r['outcome'])
    if not b.get('rows') or [key(r) for r in b['rows']]!=[key(r) for r in c['rows']]:raise ValueError('Unpaired timelines')
    ids=list(dict.fromkeys(r['shoe_id'] for r in b['rows']));lookup={s:i for i,s in enumerate(ids)}
    if len(ids)<2:raise ValueError('At least two independent shoes required')
    values=np.zeros((len(ids),len(STAGES),2,7))
    for version,data in enumerate([b,c]):
        for row in data['rows']:
            if not 0<=row['p_b']<=1 or not np.isfinite(row['p_b']) or row['direction'] not in ('B','P','Skip') or row['outcome'] not in ('B','P','T'):raise ValueError('Invalid prediction')
            bet=row['direction']!='Skip';binary=row['outcome']!='T';win=bet and binary and row['direction']==row['outcome']
            profit=(.95 if row['direction']=='B' else 1.) if win else -1. if bet and binary else 0.
            stats=[1,int(bet),int(bet and binary),int(win),profit,(row['p_b']-int(row['outcome']=='B'))**2 if binary else 0.,int(binary)]
            for j,predicate in enumerate(STAGES.values()):
                if predicate(row['round_index']):values[lookup[row['shoe_id']],j,version]+=stats
    def ratios(total):
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.stack([total[...,3]/total[...,2],total[...,4]/total[...,1],total[...,5]/total[...,6],1-total[...,1]/total[...,0]],axis=-1)
    rng=np.random.default_rng(seed);deltas=[]
    for _ in range(max(100,int(samples))):
        counts=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
        result=ratios(np.tensordot(counts,values,axes=(0,0)));deltas.append(result[:,1]-result[:,0])
    deltas=np.asarray(deltas);point=ratios(values.sum(axis=0));reports={}
    for j,name in enumerate(STAGES):
        reports[name]={}
        for k,metric in enumerate(['hit_rate','ev_per_bet','brier','skip']):
            finite=deltas[:,j,k][np.isfinite(deltas[:,j,k])]
            ci=np.quantile(finite,[.025,.975]).tolist() if len(finite)==len(deltas) else None
            delta=float(point[j,1,k]-point[j,0,k])
            reports[name][metric]={'delta':delta if np.isfinite(delta) else None,'ci95':ci,'valid_replicates':len(finite)}
    missing=[]
    for name in ['21-45','46-70','>50','50-70']:
        h=reports[name]['hit_rate']['ci95'];e=reports[name]['ev_per_bet']['ci95']
        # Point estimates also have to pass gate.py. A wide interval is not evidence of improvement.
        if h is None or e is None or not (h[0]>0 or e[0]>0):missing.append(name)
    return dict(schema_version=1,method='paired whole-shoe percentile bootstrap',samples=len(deltas),seed=seed,shoes=len(ids),stage_deltas=reports,clear_stage_benefit_passed=not missing,stages_without_clear_benefit=missing,production_promotion_allowed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);p.add_argument('--samples',type=int,default=2000);a=p.parse_args()
    result=compare(*[json.loads(f.read_text(encoding='utf-8')) for f in (a.baseline,a.candidate)],samples=a.samples)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    raise SystemExit(0 if result['clear_stage_benefit_passed'] else 2)
