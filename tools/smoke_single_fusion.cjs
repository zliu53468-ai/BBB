/* Complete causal paired smoke only. Two shoes are not accuracy evidence. */
'use strict';const fs=require('fs'),cp=require('child_process'),path=require('path'),{make}=require('../test_pf_primary_runtime');process.chdir(path.join(__dirname,'..'));
const py=`import json,numpy as np
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
rng=np.random.default_rng(20261104);shoes=[]
for limit in [50,70]:
 shoe=new_eight_deck_shoe(rng);cursor=14;rows=[]
 for t in range(limit):
  if cursor>=len(shoe)-14:break
  h,cursor=deal_baccarat_hand(shoe,cursor);rows.append(h.outcome)
 shoes.append(rows)
print(json.dumps(shoes))`;
const stat=()=>({opportunities:0,bets:0,non_tie_bets:0,hits:0,profit:0,non_tie_rows:0,brier_sum:0,errors:0,xgb_active:0,pf_active:0});
(async()=>{const shoes=JSON.parse(cp.execFileSync(process.env.PYTHON||'python',['-c',py],{encoding:'utf8'})),base=await make({bayes:true,adaptive:true,physicsCorrection:true,pfLight:true}),next=await make({bayes:true,adaptive:true,physicsCorrection:true,pfLight:true,fusion:true}),apps=[base,next],stats=[stat(),stat()],driver=base.ctx.__BGS_PARTICLE500__.estimate,start=Date.now();let cached=null,pfCalls=0;
// Both variants use identical raw Direct Physics and deterministic PF inputs.
for(const app of apps)app.ctx.__BGS_PARTICLE500__.estimate=(h,p,m,c)=>{const key=JSON.stringify([h,p,m,c]);if(cached?.key!==key){cached={key,value:driver(h,p,m,c)};pfCalls++;}return cached.value;};
for(let i=0;i<shoes.length;i++){cached=null;for(const app of apps){app.store.clear();app.store.set('bgs_xgb_final_shoe_id_v5','single-paired-'+i);}const h=[];for(const actual of shoes[i]){const predictions=apps.map(app=>app.api.applyFinalPrediction(h));for(let k=0;k<2;k++){const p=predictions[k],s=stats[k],q=p.finalProbability.finalPB,a=p.bayes_markov.single_fusion;s.opportunities++;s.errors+=!!p.finalProbability.error;s.xgb_active+=(p.finalProbability.xgb_influence||0)>0;s.pf_active+=a?a.weights.pf>0:Math.abs(p.bayes_markov.pf_light?.delta||0)>1e-12;if(actual!=='T'){s.non_tie_rows++;s.brier_sum+=(q-(actual==='B'?1:0))**2;}if(p.direction!=='Skip'){s.bets++;if(actual!=='T'){s.non_tie_bets++;s.hits+=p.direction===actual;s.profit+=p.direction===actual?(p.direction==='B'?.95:1):-1;}}apps[k].api.registerPrediction(h,p);apps[k].api.settlePending(actual,null,h);}h.push(actual);for(const app of apps)app.api.recordCompletedHand(h);if(h.length%10===0)console.log('completed shoe',i+1,'round',h.length);}}
for(const s of stats){s.hit_rate_on_bets=s.non_tie_bets?s.hits/s.non_tie_bets:null;s.realized_ev_per_bet=s.bets?s.profit/s.bets:null;s.brier=s.brier_sum/s.non_tie_rows;s.action_rate=s.bets/s.opportunities;}
const report={kind:'TWO_COMPLETE_SYNTHETIC_SHOES_FUNCTIONAL_SMOKE_NOT_ACCURACY_VALIDATION',seed:20261104,shoe_lengths:shoes.map(x=>x.length),baseline_PR98:stats[0],single_fusion:stats[1],real_pf_calls:pfCalls,shared_raw_deterministic_pf_snapshot:true,elapsed_ms:Date.now()-start,validated:false,parameters_tuned_on_this_test:false};fs.writeFileSync('validation/single_fusion_smoke.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(stats.some(s=>s.errors))process.exitCode=1;})().catch(e=>{console.error(e);process.exitCode=1;});
