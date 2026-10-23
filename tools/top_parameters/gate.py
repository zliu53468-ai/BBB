"""Read-only paired formal-runtime gate. Exit 2 means DO NOT DEPLOY."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);p.add_argument('--strict',action='store_true');p.add_argument('--min-shoes',type=int,default=1000);a=p.parse_args()
b,c=[json.loads(f.read_text(encoding='utf-8')) for f in (a.baseline,a.candidate)]
if not b.get('rows') or not c.get('rows'):raise SystemExit('Nonempty predictions required')
def key(r):return r['shoe_id'],r['round_index'],r['outcome']
if [key(r) for r in b['rows']]!=[key(r) for r in c['rows']]:raise SystemExit('Unpaired timelines')
if b.get('input_sha256')!=c.get('input_sha256'):raise SystemExit('Different input data')
stages={'all':lambda r:True,'1-20':lambda r:1<=r<=20,'21-45':lambda r:21<=r<=45,'46-70':lambda r:46<=r<=70,'<=40':lambda r:r<=40,'41-50':lambda r:41<=r<=50,'>50':lambda r:r>50,'50-70':lambda r:50<=r<=70}
def metrics(rows):
    bets=non_tie=wins=0;profit=0.;squared=[];longest=run=0
    shoe=None
    for r in rows:
        if r['shoe_id']!=shoe:run=0;shoe=r['shoe_id']
        prob=r['p_b'];out=r['outcome'];d=r['direction']
        if not math.isfinite(prob) or not 0<=prob<=1 or out not in ('B','P','T') or d not in ('B','P','Skip'):raise ValueError('Bad row')
        if out!='T':squared.append((prob-int(out=='B'))**2)
        if d=='Skip':continue
        bets+=1
        if out=='T':continue
        non_tie+=1
        if d==out:wins+=1;profit+=.95 if d=='B' else 1.;run=0
        else:profit-=1;run+=1;longest=max(longest,run)
    return dict(rows=len(rows),wagers=bets,non_tie_wagers=non_tie,absolute_correct_bets=wins,hit_rate=wins/non_tie if non_tie else None,ev_per_bet=profit/bets if bets else None,brier=sum(squared)/len(squared) if squared else None,skip=1-bets/len(rows) if rows else None,max_loss_streak=longest)
fail=[];report={}
for name,predicate in stages.items():
    x,y=[metrics([r for r in data['rows'] if predicate(r['round_index'])]) for data in (b,c)]
    report[name]={'baseline':x,'candidate':y}
    if name not in ('all','1-20','<=40','21-45','46-70'):continue
    if any(z[k] is None for z in (x,y) for k in ('hit_rate','ev_per_bet','brier','skip')):fail.append(name+': insufficient observations/bets');continue
    if y['hit_rate']+1e-12<x['hit_rate']:fail.append(name+': hit regressed')
    if y['ev_per_bet']+1e-12<x['ev_per_bet']:fail.append(name+': EV regressed')
    brier_limit=.0015 if name=='all' else .0005 if name in ('1-20','<=40') else .0015
    if y['brier']>x['brier']+brier_limit+1e-12:fail.append(name+': Brier regressed beyond limit')
    if name=='all':
        if y['absolute_correct_bets']+.000000001<.95*x['absolute_correct_bets']:fail.append('all: correct bets <95% baseline')
        delta=y['skip']-x['skip']
        if delta>.05+1e-12 or (a.strict and delta<.02-1e-12):fail.append('all: Skip constraint failed')
    if name in ('21-45','46-70') and not (y['hit_rate']>x['hit_rate']+1e-12 or y['ev_per_bet']>x['ev_per_bet']+1e-12):fail.append(name+': no observed bet quality benefit')
shoes=len({r['shoe_id'] for r in b['rows']})
if shoes<max(1,a.min_shoes):fail.append('Insufficient holdout shoes')
if 'real' not in str(b.get('data_kind','')).lower() or 'synthetic' in str(b.get('data_kind','')).lower():fail.append('Real independent holdout provenance required')
locked=['app256forward.js','app256continuation.js','particle_filter_runtime.js','final_probability_runtime.js','macro_ema.js']
for file in locked:
    if not b.get('runtime_sha256',{}).get(file) or b['runtime_sha256'][file]!=c.get('runtime_sha256',{}).get(file):fail.append('Locked code differs: '+file)
result=dict(schema_version=1,production_promotion_allowed=False,numerical_gates_passed=not fail,failures=fail,shoes=shoes,stages=report,definitions={'hit':'wins/non-tie placed bets','EV':'fixed-unit profit/all placed bets, tie=0, banker win=.95','Brier':'conditional B probability, non-ties only','Skip':'all eligible hands including ties; first hand excluded','max_loss_streak':'stage-sliced bet subsequence, ties/skips ignored'},remaining_required=['Verify frozen production manifest, model topology and 57D order','Verify real data provenance, chronological separation and no holdout tuning','Paired whole-shoe bootstrap confidence intervals and temporal subperiod stability','Pattern report from prefix only; browser/device lifecycle checks; engineering sign-off'])
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'numerical_gates_passed':not fail,'failures':fail,'production_promotion_allowed':False},ensure_ascii=False))
raise SystemExit(2 if fail else 0)
