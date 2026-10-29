"""Fixed tiny GRU candidate; NumPy BPTT; synthetic whole-shoe hold-out, no row split."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import sys,json,hashlib,copy,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
TOKENS="BPT"
def run(output,shoes=1000,epochs=8,seed=20261030):
 if shoes<100 or epochs<1:raise ValueError("need at least 100 shoes and one epoch")
 rng=np.random.default_rng(seed);sequences=[]
 for k in range(shoes):
  shoe=new_eight_deck_shoe(rng);cursor=14;obs=[]
  while cursor<len(shoe)-14 and len(obs)<70:
   hand,cursor=deal_baccarat_hand(shoe,cursor);obs.append(TOKENS.index(hand.outcome))
  sequences.append(obs)
 a=int(shoes*.6);b=int(shoes*.8);parts=[sequences[:a],sequences[a:b],sequences[b:]]
 prior=np.bincount(np.concatenate([np.array(s) for s in parts[0]]),minlength=3).astype(float);prior/=prior.sum()
 def dataset(part):
  x=[];y=[];base=[]
  for s in part:
   for t in range(20,len(s)):
    history=s[:t];x.append(history[-20:]);y.append(s[t]);p=None
    for order in [3,2,1]:
     context=history[-order:];counts=np.zeros(3)
     for i in range(order,len(history)):
      if history[i-order:i]==context:counts[history[i]]+=1
     if counts.sum()>=12:
      p=(counts+24*np.array([.45,.45,.1]))/(counts.sum()+24);break
    if p is None:p=prior.copy()
    else:
     counts=np.zeros(3)
     for i in range(1,len(history)):
      if history[i-1]==history[-1]:counts[history[i]]+=1
     if counts.sum()>=12:
      bp=(counts+24*np.array([.45,.45,.1]))/(counts.sum()+24);q=p[0]/(p[0]+p[1]);z=bp[0]/(bp[0]+bp[1]);delta=np.clip(.02*(z-q),-.001,.001)
      if (q-.5)*(q+delta-.5)<0:delta=0
      mass=p[0]+p[1];p=np.array([mass*(q+delta),mass*(1-q-delta),p[2]])
    base.append(p)
  return np.eye(3,dtype=np.float32)[np.array(x)],np.array(y),np.array(base)
 sets=[dataset(part) for part in parts];H=16
 params={}
 for gate in ['z','r','n']:
  params['W'+gate]=(rng.normal(0,.12,(3+H,H))).astype(np.float32);params['b'+gate]=np.zeros(H,dtype=np.float32)
 params['Wo']=rng.normal(0,.04,(H,3)).astype(np.float32);params['bo']=np.log(prior).astype(np.float32)
 def forward(x,cache=False):
  h=np.zeros((len(x),H),dtype=np.float32);rows=[]
  for t in range(20):
   prev=h;u=np.concatenate([x[:,t],prev],1);z=1/(1+np.exp(-np.clip(u@params['Wz']+params['bz'],-30,30)));r=1/(1+np.exp(-np.clip(u@params['Wr']+params['br'],-30,30)));v=np.concatenate([x[:,t],r*prev],1);n=np.tanh(v@params['Wn']+params['bn']);h=(1-z)*prev+z*n
   if cache:rows.append((prev,u,z,r,v,n))
  logits=h@params['Wo']+params['bo'];logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
  return p,h,rows
 def probs(x):return np.concatenate([forward(x[i:i+256])[0] for i in range(0,len(x),256)])
 def metrics(p,y):
  bp=y!=2;pred=np.argmax(p[:,:2],axis=1);return dict(rows=len(y),non_tie_rows=int(bp.sum()),multiclass_brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,1))),log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),reference_hit_rate=float(np.mean(pred[bp]==y[bp])))
 m={k:np.zeros_like(v) for k,v in params.items()};v=copy.deepcopy(m);step=0;best=None;best_loss=float('inf');history=[]
 trainx,trainy,_=sets[0]
 for epoch in range(epochs):
  # Fixed sample order. Whole-shoe split was completed before making windows.
  for start in range(0,len(trainy),128):
   x=trainx[start:start+128];y=trainy[start:start+128];p,h,rows=forward(x,True);d=p.copy();d[np.arange(len(y)),y]-=1;d/=len(y);g={k:np.zeros_like(v) for k,v in params.items()};g['Wo']=h.T@d;g['bo']=d.sum(0);dh=d@params['Wo'].T
   for prev,u,z,r,q,n in reversed(rows):
    dn=dh*z*(1-n*n);dz=dh*(n-prev)*z*(1-z);dq=dn@params['Wn'].T;dr=dq[:,3:]*prev*r*(1-r);dhprev=dh*(1-z)+dq[:,3:]*r+(dz@params['Wz'].T)[:,3:]+(dr@params['Wr'].T)[:,3:]
    g['Wn']+=q.T@dn;g['bn']+=dn.sum(0);g['Wz']+=u.T@dz;g['bz']+=dz.sum(0);g['Wr']+=u.T@dr;g['br']+=dr.sum(0);dh=dhprev
   norm=np.sqrt(sum(float(np.sum(a*a)) for a in g.values()));scale=min(1,1/max(norm,1e-12));step+=1
   for k in params:
    grad=g[k]*scale+(.001*params[k] if k.startswith('W') else 0);m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad;params[k]-=.001*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
  val=metrics(probs(sets[1][0]),sets[1][1]);history.append(dict(epoch=epoch+1,validation=val));print(json.dumps(history[-1]),flush=True)
  if val['log_loss']<best_loss:best_loss=val['log_loss'];best=copy.deepcopy(params);best_epoch=epoch+1
 params=best;report={}
 for name,(x,y,base) in zip(['validation','test'],sets[1:]):
  report[name]=dict(gru=metrics(probs(x),y),constant=metrics(np.tile(prior,(len(y),1)),y),markov_bayes_proxy=metrics(base,y))
 passed=all(report[s]['gru'][metric]<min(report[s]['constant'][metric],report[s]['markov_bayes_proxy'][metric]) for s in report for metric in ['multiclass_brier','log_loss'])
 report.update(standalone_gate_passed=passed,full_runtime_validated=False,real_world_validated=False,promotion_approved=False,reason='full_pipeline_holdout_not_completed' if passed else 'standalone_holdout_not_better_than_both_references',warning='reference_hit_rate is all-row auxiliary direction, not placed-bet hit rate; no Physics/PF in proxy')
 model=dict(schema_version=1,model_type='bbb_tiny_gru_v1',version='TINY_GRU16_CONTEXT20_V1',context=20,hidden_size=16,tokens=list(TOKENS),weights={k:a.tolist() for k,a in params.items()},training=dict(data_kind='fresh_synthetic_8_decks_cut14_fixed_burn14_cap70',seed=seed,shoes=shoes,train_shoes=int(shoes*.6),validation_shoes=int(shoes*.8)-int(shoes*.6),test_shoes=shoes-int(shoes*.8),epochs=epochs,selected_epoch=best_epoch,split='generation_order_whole_shoe_60_20_20',trained_all_gru_parameters=True,selection='validation_log_loss_only',dataset_sha256=hashlib.sha256(json.dumps(sequences).encode()).hexdigest()),validation=report)
 root=Path(output);root.mkdir(parents=True,exist_ok=True);(root/'gru_aux_candidate.json').write_text(json.dumps(model),encoding='utf8');(root/'gru_aux_report.json').write_text(json.dumps(dict(training=model['training'],validation=report,epochs=history),indent=2),encoding='utf8')
 goldx=sets[2][0][:3];goldp=probs(goldx);(root/'validation/gru_golden.json').write_text(json.dumps(dict(histories=[[TOKENS[i] for i in np.argmax(row,1)] for row in goldx],probabilities=goldp.tolist())),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='.');p.add_argument('--shoes',type=int,default=1000);p.add_argument('--epochs',type=int,default=8);p.add_argument('--seed',type=int,default=20261030);a=p.parse_args();run(a.output,a.shoes,a.epochs,a.seed)
