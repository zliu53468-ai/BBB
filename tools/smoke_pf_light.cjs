/* Real-PF sparse-prefix functional smoke; explicitly not a full-shoe backtest. */
'use strict';const fs=require('fs'),cp=require('child_process'),path=require('path'),assert=require('assert/strict'),{make}=require('../test_pf_primary_runtime');process.chdir(path.join(__dirname,'..'));
const py=`import json,numpy as np
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
rng=np.random.default_rng(20261103);shoes=[]
for i in range(4):
 shoe=new_eight_deck_shoe(rng);cursor=14;rows=[]
 for t in range(50+6*i):
  if cursor>=len(shoe)-14:break
  h,cursor=deal_baccarat_hand(shoe,cursor);rows.append(h.outcome)
 shoes.append(rows)
print(json.dumps(shoes))`;
(async()=>{const shoes=JSON.parse(cp.execFileSync(process.env.PYTHON||'python',['-c',py],{encoding:'utf8'})),on=await make({bayes:true,adaptive:true,physicsCorrection:true,pfLight:true}),off=await make({bayes:true,adaptive:true,physicsCorrection:true}),rows=[];
for(let i=0;i<shoes.length;i++){for(const app of [on,off]){app.store.clear();app.store.set('bgs_xgb_final_shoe_id_v5','pf-prefix-'+i);}for(const n of [...new Set([0,5,15,30,45,shoes[i].length-1])]){const h=shoes[i].slice(0,n),t=performance.now(),a=on.api.applyFinalPrediction(h),ms=performance.now()-t,b=off.api.applyFinalPrediction(h),d=a.bayes_markov.pf_light,repeat=on.api.applyFinalPrediction(h);assert.equal(repeat.finalProbability.finalPB,a.finalProbability.finalPB);assert(Math.abs(d.delta)<=.00250000001);assert.equal(a.finalProbability.error,'');assert.equal(d.diagnostics.particle_count,10000);assert.equal(JSON.stringify(a.finalProbability.extended),JSON.stringify(b.finalProbability.extended));rows.push({shoe:i,round:n+1,ms,base_final_pb:b.finalProbability.finalPB,pf_final_pb:a.finalProbability.finalPB,pf_pb:d.posterior_pb,physics_pb:d.physics_pb,weight:d.weight,delta:d.delta,quality:d.diagnostics.particle_quality,ess:d.diagnostics.recent_ess_ratio,tie_observations:d.diagnostics.tie_observation_count,errors:!!a.finalProbability.error});for(const [app,p]of [[on,a],[off,b]]){app.api.registerPrediction(h,p);app.api.settlePending(shoes[i][n],null,h);app.api.recordCompletedHand([...h,shoes[i][n]]);}}
console.log('completed shoe',i+1);}
const report={kind:'REAL_PF_10000_SPARSE_PREFIX_FUNCTIONAL_SMOKE_NOT_FULL_REPLAY_OR_ACCURACY_VALIDATION',seed:20261103,shoes:4,prefixes:rows.length,errors:rows.filter(x=>x.errors).length,active:rows.filter(x=>Math.abs(x.delta)>1e-12).length,max_absolute_pf_delta:Math.max(...rows.map(x=>Math.abs(x.delta))),mean_ms:rows.reduce((s,x)=>s+x.ms,0)/rows.length,max_ms:Math.max(...rows.map(x=>x.ms)),validated:false,note:'Sparse forecasts have fewer than 12 settled pairs per shoe; XGB adaptation is covered separately by integration tests. No hit-rate claim from this sample.',rows};fs.writeFileSync('validation/pf_light_smoke.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,rows:undefined}));})().catch(e=>{console.error(e);process.exitCode=1;});
