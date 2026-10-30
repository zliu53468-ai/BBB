/* Fresh synthetic paired replay. No tuning or promotion claims. */
'use strict';const fs=require('fs'),cp=require('child_process'),path=require('path'),{make}=require('../test_pf_primary_runtime');process.chdir(path.join(__dirname,'..'));
const count=Number(process.argv[2]||200);if(!Number.isInteger(count)||count<1||count>10000)throw Error('invalid shoes');
const py=`import json,numpy as np
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
rng=np.random.default_rng(20261102);shoes=[]
for i in range(${count}):
 shoe=new_eight_deck_shoe(rng);cursor=14;rows=[]
 for t in range(50+i%21):
  if cursor>=len(shoe)-14:break
  h,cursor=deal_baccarat_hand(shoe,cursor);rows.append(h.outcome)
 shoes.append(rows)
print(json.dumps(shoes))`;
function stats(){return {opportunities:0,bets:0,non_tie_bets:0,correct_bets:0,profit:0,brier_sum:0,non_tie_rows:0,max_loss_streak:0,errors:0,xgb_active:0,weight_sum:0,max_weight:0,max_delta:0,physics_correction_active:0,max_physics_delta:0};}
(async()=>{const shoes=JSON.parse(cp.execFileSync(process.env.PYTHON||'python',['-c',py],{encoding:'utf8'})),apps=[await make({bayes:true,adaptive:true}),await make({bayes:true,adaptive:true,physicsCorrection:true})],s=[stats(),stats()],start=Date.now();
for(let i=0;i<shoes.length;i++){for(const app of apps){app.store.clear();app.store.set('bgs_xgb_final_shoe_id_v5','paired-'+i);}const h=[],loss=[0,0];
 for(const actual of shoes[i]){const p=apps.map(app=>app.api.applyFinalPrediction(h));
  for(let k=0;k<2;k++){const x=p[k],a=s[k],q=x.finalProbability.finalPB,d=x.bayes_markov.xgb_adaptive,correction=x.bayes_markov.physics_correction;a.opportunities++;if(correction){a.physics_correction_active+=Math.abs(correction.delta)>1e-12;a.max_physics_delta=Math.max(a.max_physics_delta,Math.abs(correction.delta));}a.errors+=!!x.finalProbability.error;if(actual!=='T'){a.non_tie_rows++;a.brier_sum+=(q-(actual==='B'?1:0))**2;}if(d){a.xgb_active+=d.weight>0;a.weight_sum+=d.weight;a.max_weight=Math.max(a.max_weight,d.weight);a.max_delta=Math.max(a.max_delta,Math.abs(d.delta));}
   if(x.direction!=='Skip'){a.bets++;if(actual!=='T'){a.non_tie_bets++;const win=x.direction===actual;a.correct_bets+=win;a.profit+=win?(x.direction==='B'?.95:1):-1;loss[k]=win?0:loss[k]+1;a.max_loss_streak=Math.max(a.max_loss_streak,loss[k]);}}
   apps[k].api.registerPrediction(h,x);apps[k].api.settlePending(actual,null,h);
  }h.push(actual);for(const app of apps)app.api.recordCompletedHand(h);
 }if((i+1)%50===0)console.log('completed',i+1);
}
for(const a of s){a.hit_rate_on_bets=a.correct_bets/a.non_tie_bets;a.realized_ev_per_bet=a.profit/a.bets;a.brier=a.brier_sum/a.non_tie_rows;a.action_rate=a.bets/a.opportunities;a.mean_xgb_weight=a.weight_sum/a.opportunities;}
const report={kind:'FRESH_SYNTHETIC_PAIRED_FUNCTIONAL_SMOKE_NOT_REAL_HOLDOUT',seed:20261102,shoes:count,baseline_PR96:s[0],physics_correction:s[1],elapsed_ms:Date.now()-start,validated:false,parameters_tuned_on_this_test:false};fs.writeFileSync('validation/physics_correction_smoke.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(s.some(x=>x.errors))process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
