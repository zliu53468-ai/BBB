"""Fit two bounded, monotone Platt candidates on development shoes only.
Never loads or modifies production weights. Selection occurs separately on calibration shoes.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def fit(rows,delta):
    pairs=[(r['probe']['raw']['directional']['B'],int(r['outcome']=='B')) for r in rows if r['outcome']!='T']
    if len(pairs)<100:raise ValueError('At least 100 non-tie training rows required')
    data=np.array(pairs,dtype=float);p=np.clip(data[:,0],1e-9,1-1e-9);y=data[:,1];x=np.log(p/(1-p))
    def loss(v):
        q=np.clip(1/(1+np.exp(-(v[0]*x+v[1]))),np.maximum(1e-9,p-delta),np.minimum(1-1e-9,p+delta))
        return float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))+.01*((v[0]-1)**2+v[1]**2))
    result=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(.5,1.5),(-.08,.08)])
    if not result.success:raise ValueError('Calibration optimization failed: '+str(result.message))
    return {'type':'bounded_monotone_platt','slope':float(result.x[0]),'intercept':float(result.x[1]),'max_delta':delta,'identity_l2':.01,'train_non_tie_rows':len(pairs),'train_loss_in_sample_only':float(result.fun),'optimizer_success':bool(result.success)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--training-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Use a fresh candidate path')
    cache=json.loads(a.cache.read_text(encoding='utf-8-sig'));manifest=json.loads(a.training_report.read_text(encoding='utf-8-sig'))
    ids=list(dict.fromkeys(r['shoe_id'] for r in cache['rows']));at=int(len(ids)*.60);selected=set(ids[:at]);train=[r for r in cache['rows'] if r['shoe_id'] in selected]
    if len(ids)<15:raise ValueError('At least 15 chronological whole shoes required')
    result={'schema_version':1,'candidate_only':True,'production_promotion_allowed':False,'training_data_kind':manifest['data_kind'],'training_input_sha256':manifest['input_sha256'],'training_seed':manifest.get('seed'),'runtime_sha256':cache['hashes'],'training_cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'train_shoes':at,'selection_shoes':ids[at:int(len(ids)*.80)],'unused_previous_holdout_shoes':ids[int(len(ids)*.80):],'candidates':[fit(train,d) for d in (.005,.010)],'note':'Only first 60% used for fit; choose on next 20%; final evaluation requires a new independent dataset, never the previous holdout'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(a.output)
if __name__=='__main__':main()
