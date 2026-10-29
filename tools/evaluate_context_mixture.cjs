/* Sequence-only proxy evaluation, NOT a complete Physics/PF bet backtest. */
'use strict';const fs=require('fs'),cp=require('child_process'),crypto=require('crypto'),path=require('path');process.chdir(path.resolve(__dirname,'..'));
const R=require('../road_primary_runtime'),old=require('../validation/road_primary_pr83.cjs'),c=require('../cardcount_profile.json').road_primary_v1;
const py=`import json,numpy as np
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
rng=np.random.default_rng(5182026);data=[]
for i in range(500):
 shoe=new_eight_deck_shoe(rng);cursor=14;obs=[]
 while cursor<len(shoe)-14 and len(obs)<70:
  hand,cursor=deal_baccarat_hand(shoe,cursor);obs.append(hand.outcome)
 data.append(obs)
print(json.dumps(data))`;
const raw=cp.execFileSync(process.env.PYTHON||'python',['-c',py],{encoding:'utf8',maxBuffer:10e6}),shoes=JSON.parse(raw),tokens=['B','P','T'];
function fresh(){return {rows:0,brier_sum:0,log_sum:0,available:0,reference_actions:0,reference_non_tie_actions:0,reference_correct:0,reference_profit:0,max_reference_loss_streak:0};}
const totals={pr83:fresh(),mixture:fresh(),always_b:fresh()},batches=Array.from({length:5},()=>({pr83:fresh(),mixture:fresh(),always_b:fresh()}));
for(let index=0;index<shoes.length;index++){const seq=shoes[index],streak={pr83:0,mixture:0,always_b:0};for(let t=0;t<seq.length;t++){const prefix=seq.slice(0,t),actual=seq[t],outputs={pr83:old.estimate(prefix,c),mixture:R.estimate(prefix,c),always_b:{available:true,probabilities:c.prior,direction:'B'}};for(const [name,result]of Object.entries(outputs)){const p=result.available?result.probabilities:c.prior,action=result.available&&['B','P'].includes(result.direction)?result.direction:null;if(action&&actual!=='T')streak[name]=action===actual?0:streak[name]+1;for(const stats of [totals[name],batches[Math.floor(index/100)][name]]){stats.rows++;stats.brier_sum+=tokens.reduce((s,k)=>s+(p[k]-(k===actual?1:0))**2,0);stats.log_sum-=Math.log(Math.max(1e-12,p[actual]));if(result.available)stats.available++;if(action){stats.reference_actions++;if(actual!=='T'){stats.reference_non_tie_actions++;const hit=action===actual;if(hit)stats.reference_correct++;stats.reference_profit+=hit?(action==='B'?.95:1):-1;stats.max_reference_loss_streak=Math.max(stats.max_reference_loss_streak,streak[name]);}}}}}}
function finalize(s){return {...s,multiclass_brier:s.brier_sum/s.rows,log_loss:s.log_sum/s.rows,candidate_coverage:s.available/s.rows,reference_hit_rate:s.reference_correct/(s.reference_non_tie_actions||1),reference_profit_per_action:s.reference_profit/(s.reference_actions||1)};}
const report={schema_version:1,data_kind:'fresh_synthetic_8_decks_cut14_burn14_cap70',seed:5182026,shoes:500,whole_shoe_batches:5,sha256:crypto.createHash('sha256').update(raw).digest('hex'),no_training_or_parameter_selection:true,full_pipeline_bet_backtest:false,real_world_validated:false,warning:'Sequence candidates only. Missing candidates use the same fixed prior for Brier/logloss; candidate availability is NOT formal bet eligibility. Profit and streak metrics are reference directions without Physics/PF, 0.5pp, EMA or EV/Volume Guard.',totals:Object.fromEntries(Object.entries(totals).map(([k,v])=>[k,finalize(v)])),batches:batches.map((b,i)=>({batch:i+1,...Object.fromEntries(Object.entries(b).map(([k,v])=>[k,finalize(v)]))}))};fs.writeFileSync('validation/context_mixture_smoke.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report.totals));
