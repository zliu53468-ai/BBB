"""Isolated full retraining orchestration. No production writes or automatic promotion."""
import argparse,hashlib,importlib.metadata,json,math,shutil,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def split_hands(payload,fraction=.20,min_dev_shoes=1000,min_holdout_shoes=300):
    if payload.get('data_kind')!='real_observations' or payload.get('chronological_order')!='oldest_to_newest':raise ValueError('Need real_observations and chronological_order=oldest_to_newest, with externally verified provenance')
    rows=payload.get('rows');shoes=[];groups={};previous=None
    if not isinstance(rows,list) or not rows:raise ValueError('Complete chronological rows required')
    for row in rows:
        sid=str(row.get('shoe_id',''))
        if not sid:raise ValueError('shoe_id required')
        if sid!=previous:
            if sid in groups:raise ValueError('Interleaved shoe')
            shoes.append(sid);groups[sid]=[];previous=sid
        if type(row.get('round_index')) is not int or row['round_index']!=len(groups[sid])+1:raise ValueError('Missing/duplicate observed hand, including skipped hands and ties')
        if row.get('outcome') not in ('B','P','T'):raise ValueError('B/P/T outcome required')
        for key,lo,hi in [('card_count',4,6),('player_score',0,9),('banker_score',0,9)]:
            v=row.get(key)
            if v is not None and (type(v) is not int or not lo<=v<=hi):raise ValueError('Invalid measured '+key)
        if row.get('player_score') is not None and row.get('banker_score') is not None:
            ps,bs=row['player_score'],row['banker_score'];actual='B' if bs>ps else 'P' if ps>bs else 'T'
            if row['outcome']!=actual:raise ValueError('Outcome/points mismatch')
        groups[sid].append({k:row[k] for k in ['shoe_id','round_index','outcome','card_count','player_score','banker_score'] if k in row})
    n=max(1,math.ceil(len(shoes)*fraction));development=shoes[:-n];holdout=shoes[-n:]
    if len(development)<min_dev_shoes or len(holdout)<min_holdout_shoes:raise ValueError(f'Need >= {min_dev_shoes} development and {min_holdout_shoes} independent holdout shoes; got {len(development)}/{len(holdout)}')
    binary=sum(r['outcome']!='T' and r['round_index']>1 for s in development for r in groups[s])
    binary_shoes=sum(any(r['outcome']!='T' and r['round_index']>1 for r in groups[s]) for s in development)
    if binary<12000 or binary_shoes<min_dev_shoes:raise ValueError('Development input fails 12000 binary rows / 1000 directional shoes floor')
    def pack(ids):return dict(data_kind='real_observations',chronological_order='oldest_to_newest',rows=[r for s in ids for r in groups[s]])
    return pack(development),pack(holdout),{'development_shoes':development,'outer_holdout_shoes':holdout,'development_binary_rows':binary}

def stage_models(base,candidate,physics_path,xgb_path):
    physics=read(physics_path);xgb=read(xgb_path);oldp=read(base/'physics_multitask_model.json');oldx=read(base/'final_probability_model.json')
    shape=lambda model:[(len(w),len(w[0])) for w in model['coefs']]
    if (physics.get('history_input_dim'),physics.get('physics_dim'),physics.get('activation'))!=(213,48,'relu') or shape(physics)!=shape(oldp) or physics['feature_names']!=oldp['feature_names']:raise ValueError('Physics architecture/order changed')
    if xgb.get('model_type')!='xgb_final_probability_classifier' or xgb.get('link')!='sigmoid' or len(xgb.get('feature_names',[]))!=57 or xgb['feature_names']!=oldx['feature_names']:raise ValueError('57D binary architecture/order changed')
    # Keep runtime decision settings and PF enabled exactly as production; evaluate weights first.
    for key in ['decision_policy','smoothing','ev_thresholds','probability_bounds','particle_filter_enabled','particle_filter_version','physics_direct_version','physics_primary_version','early35_version','shoe_error_correction_version']:
        if key in oldx:xgb[key]=oldx[key]
    # Keep runtime algorithm switches; retain candidate-fitted probability/noise calibration.
    for key in ['stage_progress_version','noise_score_version']:
        if key in oldx.get('training',{}):xgb.setdefault('training',{})[key]=oldx['training'][key]
    shutil.copy2(physics_path,candidate/'physics_multitask_model.json')
    write(candidate/'final_probability_model.json',xgb)
    return {'physics_shape':shape(physics),'features':57,'policy_and_pf_pinned_to_production':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--hands',type=Path);p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--strict',action='store_true');p.add_argument('--min-holdout-shoes',type=int,default=300);a=p.parse_args()
    work=a.work_dir.resolve()
    if work==ROOT or ROOT in work.parents:raise SystemExit('work-dir must be outside production checkout')
    work.mkdir(parents=True,exist_ok=True)
    result={'schema_version':1,'baseline_main':'39d7a6102ace1da8f5cbd7d99f32a5e565ca99ee','training_profile':'middle_late_focus','status':'preflight','training_executed':False,'production_modified':False,'promotion_allowed':False,'blockers':[],'checks':{},'holdout_metrics':None,'stages':['1-20','21-45','46-70','<=40','41-50','>50','50-70']}
    report=work/'execution_report.json'
    def save():write(report,result)
    expected=read(HERE/'production_manifest.json')['sha256']
    for name,value in expected.items():
        if not (ROOT/name).is_file() or digest(ROOT/name)!=value:result['blockers'].append('Frozen production differs: '+name)
    result['production_sha256']={name:digest(ROOT/name) for name in ['app256forward.js','app256continuation.js','particle_filter_runtime.js','cardcount_profile.json','physics_multitask_model.json','final_probability_model.json','physics_multitask_model.joblib'] if (ROOT/name).exists()}
    versions={}
    for name,minimum in [('numpy',(1,26)),('scikit-learn',(1,5)),('xgboost',(2,1)),('joblib',(1,3))]:
        try:
            version=importlib.metadata.version(name);versions[name]=version
            parts=tuple(int(x) for x in version.split('.')[:2])
            if parts<minimum or (name in ('scikit-learn','xgboost') and parts[0]>=(2 if name=='scikit-learn' else 4)):result['blockers'].append('Unsupported dependency: '+name+' '+version)
        except importlib.metadata.PackageNotFoundError:versions[name]=None;result['blockers'].append('Missing dependency: '+name)
    if not shutil.which('node'):result['blockers'].append('Node.js required')
    result['dependency_versions']=versions
    if not a.hands or not a.hands.is_file():result['blockers'].append('需要使用者提供完整 chronological 牌靴資料')
    else:
        try:
            dev,hold,split=split_hands(read(a.hands),min_holdout_shoes=max(300,a.min_holdout_shoes));write(work/'development_hands.json',dev);write(work/'strict_holdout_hands.json',hold);write(work/'shoe_split.json',split);result['input_sha256']=digest(a.hands);result['split']={k:len(v) if isinstance(v,list) else v for k,v in split.items()}
        except (ValueError,KeyError,TypeError) as e:result['blockers'].append(str(e))
    if result['blockers']:result['status']='blocked_before_training';save();print(json.dumps(result,ensure_ascii=True));return 2
    if not a.execute:result['status']='preflight_passed_not_started';save();print(report);return 0
    if (work/'baseline').exists() or (work/'candidate').exists():raise SystemExit('Use a fresh work-dir; do not overwrite a previous run')
    base=work/'baseline';candidate=work/'candidate'
    for dest in (base,candidate):
        dest.mkdir()
        for file in ROOT.iterdir():
            if file.is_file() and file.suffix in ('.json','.js','.py','.joblib','.txt'):shutil.copy2(file,dest/file.name)
    result['checks']['frozen_baseline_saved']=True
    result['commands']=[];save()
    def run(name,command,cwd=candidate,allowed=(0,)):
        result['commands'].append({'name':name,'argv':[str(x) for x in command]});save()
        with (work/(name+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([str(x) for x in command],cwd=cwd,stdout=log,stderr=subprocess.STDOUT).returncode
        result['commands'][-1]['returncode']=code;save()
        if code not in allowed:raise RuntimeError(name+' failed, see '+str(work/(name+'.log')))
        return code
    try:
        result['training_executed']=True;result['status']='training';save()
        run('physics',[sys.executable,HERE/'train_physics_chronological.py','train','--training-profile','middle_late_focus','--shoes','6000','--validation-fraction','.20','--calibration-fraction','.10','--augment-ratio','.25','--bootstrap-samples','2000','--random-state','20261023','--output',candidate/'physics_middle_late_candidate.joblib','--browser-output',candidate/'physics_middle_late_candidate.json'])
        run('core_rows',['node',HERE/'build_core_rows.js',base,work/'development_hands.json',work/'development_core_rows.json'])
        run('xgb',[sys.executable,'xgb_final_probability.py','train','--training-profile','middle_late_focus','--input',work/'development_core_rows.json','--physics-model',candidate/'physics_middle_late_candidate.joblib','--min-samples','12000','--min-shoes','1000','--validation-fraction','.20','--calibration-fraction','.15','--tuning-fraction','.15','--ev-tuning-fraction','.30','--bootstrap-samples','2000','--random-state','20261023','--report-output',work/'middle_late_candidate_report.json','--joblib-output',candidate/'final_probability_middle_late_candidate.joblib','--output',candidate/'final_probability_middle_late_candidate.json'])
        result['checks']['architecture']=stage_models(base,candidate,candidate/'physics_middle_late_candidate.json',candidate/'final_probability_middle_late_candidate.json');save()
        for name,directory in [('baseline',base),('candidate',candidate)]:run(name+'_replay',['node',HERE/'replay.js',directory,work/'strict_holdout_hands.json',work/(name+'_predictions.json')])
        gate=run('production_gate',[sys.executable,HERE/'gate.py',work/'baseline_predictions.json',work/'candidate_predictions.json',work/'production_gate.json','--min-shoes',str(max(300,a.min_holdout_shoes))]+(['--strict'] if a.strict else []),allowed=(0,2))
        bootstrap=run('bootstrap',[sys.executable,HERE/'paired_bootstrap.py',work/'baseline_predictions.json',work/'candidate_predictions.json',work/'paired_bootstrap.json','--samples','2000'],allowed=(0,2))
        result['holdout_metrics']=read(work/'production_gate.json')['stages'];result['checks']['numerical_gates_passed']=gate==0;result['checks']['clear_stage_benefit_passed']=bootstrap==0
        result['status']='eligible_for_manual_release_review' if gate==bootstrap==0 else 'holdout_failed_no_promotion'
    except Exception as e:result['status']='execution_failed_no_promotion';result['blockers'].append(str(e))
    # Refuse success if any production file changed during execution.
    for name,value in result['production_sha256'].items():
        if digest(ROOT/name)!=value:result['blockers'].append('Production mutated: '+name);result['status']='production_integrity_failed'
    save();print(report)
    return 0 if result['status']=='eligible_for_manual_release_review' else 2

if __name__=='__main__':raise SystemExit(main())
