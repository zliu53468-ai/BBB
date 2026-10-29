"""Offline KNN feasibility pipeline: factual metrics even for small SYNTHETIC runs."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from jsonschema import Draft202012Validator, FormatChecker

HERE = Path(__file__).resolve().parent
METRICS = ['standard_hit_rate', 'pct_shoes_max_losses_le_2',
           'pct_shoes_with_four_streak', 'mean_shoe_five_window_coverage',
           'mean_shoe_six_window_coverage', 'action_rate']


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        f.write('\n')


def validate(payload):
    schema = json.loads((HERE/'data.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
    kind = payload['data_kind']
    if kind not in ('SYNTHETIC', 'real_observations'):
        raise ValueError('Explicit data kind required')
    if payload['chronological_order'] != 'oldest_to_newest' or not payload['source']:
        raise ValueError('Source and chronological order required')
    shoes = {}; closed = set(); last = None
    for row in payload['rows']:
        sid = row['shoe_id']
        if not isinstance(sid, str) or not sid: raise ValueError('shoe_id')
        if row['data_kind'] != kind: raise ValueError('mixed provenance')
        if sid != last:
            if sid in closed: raise ValueError('interleaved shoes')
            if last is not None: closed.add(last)
            last = sid
        events = shoes.setdefault(sid, [])
        if row['round_index'] != len(events)+1 or row['outcome'] not in 'BPT' or len(row['outcome']) != 1:
            raise ValueError('incomplete chronological observations')
        events.append(row)
    good = {}; split = []; previous = None
    for sid, events in shoes.items():
        record = dict(shoe_id=sid, rounds=len(events), split=None,
                      status='ELIGIBLE' if 50 <= len(events) <= 70 else 'OUT_OF_RANGE_SHOE')
        split.append(record)
        if record['status'] != 'ELIGIBLE': continue
        start = datetime.fromisoformat(events[0]['shoe_start'].replace('Z', '+00:00'))
        if start.tzinfo is None or previous is not None and start <= previous: raise ValueError('shoe time order')
        previous = start
        expected = events[0]['initial_counts_after_burn']
        for row in events:
            before, consumed = row['remaining_counts_before'], row['consumed_counts']
            for c in (before, consumed, row['initial_counts_after_burn']):
                if len(c) != 10 or any(type(v) is not int or v < 0 for v in c): raise ValueError('10 counts required')
            if row['initial_counts_after_burn'] != events[0]['initial_counts_after_burn'] or sum(expected) > 416:
                raise ValueError('initial composition inconsistent')
            if before != expected or any(a < b for a, b in zip(before, consumed)) or not 4 <= sum(consumed) <= 6:
                raise ValueError('composition conservation')
            if row['shoe_start'] != events[0]['shoe_start']: raise ValueError('shoe time inconsistent')
            expected = [a-b for a, b in zip(before, consumed)]
            if row['composition_through_round'] != row['round_index']-1 or not row['composition_evidence']:
                raise ValueError('prefix evidence required')
            expected_source = 'synthetic_fully_observed_prefix' if kind == 'SYNTHETIC' else 'observed_prefix'
            if row['composition_source'] != expected_source: raise ValueError('oracle/unknown composition prohibited')
        good[sid] = events
    ids = list(good); cut = int(len(ids)*.7); tune = cut+(len(ids)-cut)//2
    if cut < 1 or tune == cut or tune == len(ids): raise ValueError('Need library, fit and test shoes')
    groups = {'library': ids[:cut], 'validation_fit': ids[cut:tune], 'validation_test': ids[tune:]}
    for row in split:
        row['split'] = next((g for g, ids in groups.items() if row['shoe_id'] in ids), None)
    return good, groups, split


def feature(row):
    c = np.asarray(row['remaining_counts_before'], dtype=float)
    return np.r_[c.sum(), c/c.sum()]


def build_signals(shoes, groups, k, cap, seed):
    library = [r for sid in groups['library'] for r in shoes[sid][1:]]
    if len(library) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(len(library), cap, replace=False))
        library = [library[i] for i in idx]
    if len(library) < k: raise ValueError('library smaller than k')
    x = np.array([feature(r) for r in library]); mu = x.mean(0); scale = x.std(0); scale[scale == 0] = 1
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=1).fit((x-mu)/scale)
    queries = [r for g in ('validation_fit', 'validation_test') for sid in groups[g] for r in shoes[sid][1:]]
    # Stable full distance sort resolves ties deterministically in library order.
    signals = {}
    for row in queries:
        q = (feature(row)-mu)/scale
        distances, indices = nn.kneighbors(q[None], n_neighbors=len(library))
        order = np.lexsort((indices[0], distances[0]))[:k]
        ix = indices[0][order]; ds = distances[0][order]
        counts = Counter(library[i]['outcome'] for i in ix)
        p = np.array([counts[o]/k for o in 'BPT']); nonzero = p[p > 0]
        signals[f"{row['shoe_id']}:{row['round_index']}"] = dict(
            b_count=counts['B'], p_count=counts['P'], tie_count=counts['T'],
            distribution=dict(zip('BPT', p.tolist())),
            p_b=(counts['B']+1)/(counts['B']+counts['P']+2),
            mean_distance=float(ds.mean()), distance_variance=float(ds.var()),
            entropy=float(-(nonzero*np.log(nonzero)).sum()/np.log(3)),
            neighbor_keys=[f"{library[i]['shoe_id']}:{library[i]['round_index']}" for i in ix])
    return signals, dict(rows=len(library), mean=mu.tolist(), scale=scale.tolist(), k=k,
                         dimensions=['remaining_total']+[f'remaining_point_{i}_ratio' for i in range(10)])


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))


def fit(rows, signals, gate=None):
    data = [r for r in rows if r['standard_eligible'] and r['outcome'] != 'T' and
            (gate is None or signals[f"{r['shoe_id']}:{r['round_index']}"]['mean_distance'] <= gate[0] and
             signals[f"{r['shoe_id']}:{r['round_index']}"]['entropy'] <= gate[1])]
    if len(data) < 2 or len({r['outcome'] for r in data}) < 2: raise ValueError('insufficient fit labels')
    x = np.array([[1, logit(r['p_b']), logit(signals[f"{r['shoe_id']}:{r['round_index']}"]['p_b'])] for r in data])
    y = np.array([r['outcome'] == 'B' for r in data], dtype=float)
    def objective(w):
        z=x@w
        return float(np.mean(np.logaddexp(0,z)-y*z) + .001*np.sum(w*w))
    result = minimize(objective, np.zeros(3), method='L-BFGS-B', bounds=[(None,None),(0,None),(0,None)])
    if not result.success: raise ValueError('fusion optimizer failed: '+result.message)
    return result.x.tolist(), len(data)


def models(rows, signals, fit_ids):
    weights, n = fit(rows, signals)
    inner_cut = max(1, len(fit_ids)//2); train_ids = fit_ids[:inner_cut]; valid_ids = fit_ids[inner_cut:]
    if not valid_ids: raise ValueError('Need at least two fitting shoes for forward validation')
    train = [r for r in rows if r['shoe_id'] in train_ids]
    valid = [r for r in rows if r['shoe_id'] in valid_ids and r['outcome'] != 'T' and r['standard_eligible']]
    if not valid: raise ValueError('No forward-validation strong labels')
    # Threshold values drawn only from past fitting shoes, never final test.
    ss = [signals[f"{r['shoe_id']}:{r['round_index']}"] for r in train]
    distances = np.quantile([s['mean_distance'] for s in ss], [.5,.75,1])
    entropies = np.quantile([s['entropy'] for s in ss], [.5,.75,1])
    candidates = []
    for d in distances:
        for h in entropies:
            try: w, _ = fit(train, signals, (d,h))
            except ValueError: continue
            losses = []
            for r in valid:
                s=signals[f"{r['shoe_id']}:{r['round_index']}"]
                z=w[0]+w[1]*logit(r['p_b'])+w[2]*logit(s['p_b']) if s['mean_distance'] <= d and s['entropy'] <= h else logit(r['p_b'])
                losses.append(float(np.logaddexp(0,z)-(r['outcome']=='B')*z))
            candidates.append((float(np.mean(losses)), float(d), float(h)))
    if not candidates: raise ValueError('No fit-only gate candidate')
    score, d, h = min(candidates); gated_w, gn = fit(rows, signals, (d,h))
    return dict(ungated=dict(weights=weights,fit_labels=n),
                gated=dict(weights=gated_w,fit_labels=gn,max_distance=d,max_entropy=h,forward_logloss=score),
                fitting_shoes=fit_ids, inner_train_shoes=train_ids, inner_validation_shoes=valid_ids)


def metrics(rows):
    actions=[r for r in rows if r['direction'] != 'Skip']; resolved=[r for r in actions if r['outcome'] != 'T']
    wins=sum(r['direction']==r['outcome'] for r in resolved); losses=run=streak=maxloss=0; four=False
    for r in resolved:
        if r['direction']==r['outcome']: run=0; streak+=1; four |= streak>=4
        else: losses+=1; run+=1; maxloss=max(maxloss,run); streak=0
    def coverage(size):
        # Includes first hand as a non-action original-round slot.
        timeline=[dict(direction='Skip',outcome='T')]+rows; covered=0
        for i in range(len(timeline)-size+1):
            seq=[r for r in timeline[i:i+size] if r['direction']!='Skip' and r['outcome']!='T']
            w=0; found=False
            for r in seq:
                w=w+1 if r['direction']==r['outcome'] else 0; found |= w>=4
            covered+=found
        return covered/(len(timeline)-size+1)
    return dict(opportunities=len(rows),standard_actions=len(actions),non_tie_standard_actions=len(resolved),
                wins=wins,losses=losses,max_losses=maxloss,has_four_streak=int(four),
                five_coverage=coverage(5),six_coverage=coverage(6),
                standard_hit_rate=wins/len(resolved) if resolved else 0.,hit_rate_estimable=bool(resolved),
                action_rate=len(actions)/len(rows),light_actions_counted=0)


def aggregate(shoes):
    n=len(shoes); opportunities=sum(s['opportunities'] for s in shoes)
    actions=sum(s['standard_actions'] for s in shoes); resolved=sum(s['non_tie_standard_actions'] for s in shoes)
    return dict(standard_hit_rate=sum(s['wins'] for s in shoes)/resolved if resolved else 0.,
                hit_rate_estimable=bool(resolved),pct_shoes_max_losses_le_2=sum(s['max_losses']<=2 for s in shoes)/n,
                pct_shoes_with_four_streak=sum(s['has_four_streak'] for s in shoes)/n,
                mean_shoe_five_window_coverage=sum(s['five_coverage'] for s in shoes)/n,
                mean_shoe_six_window_coverage=sum(s['six_coverage'] for s in shoes)/n,
                action_rate=actions/opportunities,standard_actions=actions,non_tie_standard_actions=resolved,
                shoes=n,light_actions_counted=0)


def bootstrap(per_shoe, seed, samples):
    rng=np.random.default_rng(seed); n=len(per_shoe['baseline']); result={}
    for group in ('ungated','gated'):
        distribution={m:[] for m in METRICS}; undefined=0
        for _ in range(samples):
            ix=rng.integers(0,n,n); b=aggregate([per_shoe['baseline'][i] for i in ix]); c=aggregate([per_shoe[group][i] for i in ix])
            if not b['hit_rate_estimable'] or not c['hit_rate_estimable']: undefined+=1; continue
            for m in METRICS: distribution[m].append(c[m]-b[m])
        point_b=aggregate(per_shoe['baseline']); point_c=aggregate(per_shoe[group])
        result[group]=dict(undefined_replicates=undefined,metrics={})
        for m, values in distribution.items():
            if not values: raise ValueError('No estimable bootstrap replicates')
            result[group]['metrics'][m]=dict(delta=point_c[m]-point_b[m],
                ci95=np.quantile(values,[.025,.975]).tolist(),
                simultaneous_ci=np.quantile(values,[.05/12,1-.05/12]).tolist())
    return result


def annotate(obj, kind):
    if isinstance(obj,dict): return {k:annotate(v,kind) for k,v in obj.items()}
    if isinstance(obj,list): return [annotate(v,kind) for v in obj]
    if isinstance(obj,(int,float)) and not isinstance(obj,bool): return dict(value=obj,data_kind=kind)
    return obj


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--k',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261029); p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args()
    if not 20<=a.k<=50: p.error('k must be 20..50')
    if a.output.exists(): p.error('Use a fresh output directory')
    payload=json.loads(a.input.read_text()); kind=payload['data_kind']; shoes,groups,split=validate(payload)
    if kind=='SYNTHETIC' and 'SYNTHETIC' not in str(a.output): p.error('SYNTHETIC output path required')
    a.output.mkdir(parents=True)
    signals,library=build_signals(shoes,groups,a.k,500000,a.seed)
    if kind!='SYNTHETIC' and library['rows']<50000: raise ValueError('Real library needs 50000 rows')
    write(a.output/'library.json',dict(data_kind=kind,**library))
    # Baseline traces only fitting/test shoes; no future outcomes enter retrieval.
    replay_input=dict(payload,rows=[r for g in ('validation_fit','validation_test') for sid in groups[g] for r in shoes[sid]])
    write(a.output/'replay_input.json',replay_input)
    def replay(group):
        out=a.output/(group+'_trace.json')
        cmd=['node',str(HERE/'replay.js'),str(a.output/'replay_input.json'),str(out)]
        if group!='baseline':cmd += [str(a.output/'signals_and_fit.json'),group]
        subprocess.run(cmd,check=True)
        return json.loads(out.read_text())
    baseline=replay('baseline'); fit_rows=[r for r in baseline['rows'] if r['shoe_id'] in groups['validation_fit']]
    model=models(fit_rows,signals,groups['validation_fit'])
    write(a.output/'signals_and_fit.json',dict(data_kind=kind,models=model,signals=signals))
    traces=dict(baseline=baseline,ungated=replay('ungated'),gated=replay('gated'))
    per_shoe={g:[dict(shoe_id=sid,**metrics([r for r in trace['rows'] if r['shoe_id']==sid])) for sid in groups['validation_test']] for g,trace in traces.items()}
    summary={g:aggregate(rows) for g,rows in per_shoe.items()}
    for trace in traces.values():
        assert trace['runtime_sha256']==baseline['runtime_sha256']
        assert all(r['direction']=='Skip' or r['entry_tier']=='strong' and r['stake_multiplier']==1 for r in trace['rows'])
        assert all(not r['halted_before'] or r['direction']=='Skip' for r in trace['rows'])
    assert all(m['pct_shoes_max_losses_le_2']==1. for m in summary.values())
    ci=bootstrap(per_shoe,a.seed,a.bootstrap)
    insufficient=any(m['standard_actions']<100 for m in summary.values())
    if kind!='SYNTHETIC':
        insufficient |= any(m['shoes']<30 or m['non_tie_standard_actions']<500 for m in summary.values())
    primary=['standard_hit_rate','mean_shoe_five_window_coverage','mean_shoe_six_window_coverage']
    increment={g:all(ci[g]['metrics'][m]['simultaneous_ci'][0]>0 for m in primary) for g in ('ungated','gated')}
    status='INSUFFICIENT_SAMPLE' if insufficient else 'SYNTHETIC_PIPELINE_ONLY' if kind=='SYNTHETIC' else 'INCREMENT_IN_REPLAY' if any(increment.values()) else 'NO_INCREMENT'
    report=dict(data_kind=kind,claim_scope='回放統計',status=status,pipeline_complete=True,
                warning='合成資料僅用於管線驗證，不得用於宣稱真實增量' if kind=='SYNTHETIC' else '僅限回放統計',
                true_increment_evaluated=kind!='SYNTHETIC',expansion_allowed=False,
                library_below_target=library['rows']<50000,seed=a.seed,k=a.k,bootstrap_replicates=a.bootstrap,
                source=payload['source'],input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                source_time_range=[next(iter(shoes.values()))[0]['shoe_start'],list(shoes.values())[-1][0]['shoe_start']],
                split=split,library=library,models=model,summary=summary,per_shoe=per_shoe,paired_bootstrap=ci,
                runtime_sha256=baseline['runtime_sha256'],real_NO_INCREMENT=None if kind=='SYNTHETIC' else not any(increment.values()),
                definitions=dict(standard='PR72 strong AND stake_multiplier=1; weak excluded; additional irreversible two-loss per-shoe stop',
                    zero_denominator='hit_rate=0 by reporting convention only; hit_rate_estimable=false; never evidence of 0% true rate',
                    bootstrap='paired shoe clusters; ratio of pooled wins/non-tie bets; coverage equal-shoe mean; six-test Bonferroni simultaneous intervals'))
    write(a.output/'report.json',annotate(report,kind))
    tag=lambda x:f'{x:.8g} {kind}' if isinstance(x,(int,float)) else str(x)
    lines=['# SYNTHETIC 管線回放統計' if kind=='SYNTHETIC' else '# 回放統計', '', report['warning'],
           '', '合成資料的結果不能用來判斷真實增量，只能用來證明管線可運行。' if kind=='SYNTHETIC' else '',
           '', f'狀態：{status}。建庫→檢索→離線擬合→三組獨立 runtime 回放已完成。',
           f'建庫樣本：{tag(library["rows"])}；k：{tag(a.k)}；seed：{tag(a.seed)}。',
           '本次小庫低於真實實驗規模要求，只驗證機制。' if library['rows']<50000 else '',
           '', '| 指標 | baseline | ungated | gated |','|---|---|---|---|']
    for m in METRICS+['standard_actions','non_tie_standard_actions','shoes','light_actions_counted']:
        lines.append('| '+m+' | '+' | '.join(tag(summary[g][m]) for g in traces)+' |')
    lines+=['','## 配對整靴 bootstrap 增量（候選減基線）','',f'| 組別/指標 | 增量 | 95% CI（{kind}） | 多重比較同時 CI（{kind}） |','|---|---|---|---|']
    for g in ('ungated','gated'):
        for m,v in ci[g]['metrics'].items():
            lines.append(f'| {g}/{m} | {tag(v["delta"])} | '+', '.join(map(tag,v['ci95']))+' | '+', '.join(map(tag,v['simultaneous_ci']))+' |')
    lines+=['','## 逐靴表','','| 組別 | 靴 | standard 出手 | 勝 | 負 | 最大連敗 | 四連勝 | 五窗口 | 六窗口 |','|---|---|---|---|---|---|---|---|---|']
    for g,ss in per_shoe.items():
        for s in ss: lines.append('| '+g+' | '+s['shoe_id']+' | '+' | '.join(tag(s[k]) for k in ['standard_actions','wins','losses','max_losses','has_four_streak','five_coverage','six_coverage'])+' |')
    lines+=['','## 資料切分','','| 靴 | 局數 | 分組 | 狀態 |','|---|---|---|---|']
    for r in split:lines.append(f'| {r["shoe_id"]} | {tag(r["rounds"])} | {r["split"]} | {r["status"]} |')
    lines+=['','## 離線擬合（截距、基線 logit 係數、檢索 logit 係數）','']
    for g in ('ungated','gated'):
        lines.append(g+': '+', '.join(tag(v) for v in model[g]['weights']))
    if all(abs(model[g]['weights'][2])<1e-12 for g in ('ungated','gated')):
        lines+=['','兩組檢索係數皆由離線擬合降為 '+tag(0)+'；數值差異不能歸因於檢索增益。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=status,data_kind=kind,summary=summary),ensure_ascii=False))


if __name__=='__main__': main()
