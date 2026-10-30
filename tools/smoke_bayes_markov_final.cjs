/* Synthetic functional smoke only; no tuning and no accuracy promotion. */
'use strict';const fs=require('fs'),cp=require('child_process'),path=require('path'),{make}=require('../test_pf_primary_runtime');process.chdir(path.join(__dirname,'..'));
const count=Number(process.argv[2]||200);if(!Number.isInteger(count)||count<1||count>10000)throw Error('invalid shoe count');
const py=`import json,numpy as np
from physics_feature_extractor import new_eight_deck_shoe,deal_baccarat_hand
rng=np.random.default_rng(20261030);shoes=[]
for i in range(${count}):
 shoe=new_eight_deck_shoe(rng);cursor=14;rows=[];limit=50+i%21
 for t in range(limit):
  if cursor>=len(shoe)-14:break
  h,cursor=deal_baccarat_hand(shoe,cursor);rows.append(h.outcome)
 shoes.append(rows)
print(json.dumps(shoes))`;
(async()=>{const shoes=JSON.parse(cp.execFileSync(process.env.PYTHON||'python',['-c',py],{encoding:'utf8'})),{api,store}=await make({bayes:true}),stats={seed:20261030,data_kind:'SYNTHETIC_FUNCTIONAL_SMOKE_NOT_REAL_HOLDOUT',shoes:count,opportunities:0,bets:0,non_tie_bets:0,correct_bets:0,profit:0,display_recommendations:0,banker_forecasts:0,player_forecasts:0,orders:{0:0,1:0,2:0,3:0},errors:0,validated:false},start=Date.now();
for(let i=0;i<shoes.length;i++){store.clear();store.set('bgs_xgb_final_shoe_id_v5','bayes-smoke-'+i);const h=[];
 for(const actual of shoes[i]){const p=api.applyFinalPrediction(h),f=p.next_hand_forecast;stats.opportunities++;stats.errors+=!!p.finalProbability.error;stats.orders[p.bayes_markov.selected_order||0]++;stats.banker_forecasts+=f.direction==='B';stats.player_forecasts+=f.direction==='P';stats.display_recommendations+=['B','P'].includes(api.displayDirection(f));if(p.direction!=='Skip'){stats.bets++;if(actual!=='T'){stats.non_tie_bets++;stats.correct_bets+=p.direction===actual;stats.profit+=p.direction===actual?(p.direction==='B'?.95:1):-1;}}
 api.registerPrediction(h,p);api.settlePending(actual,null,h);h.push(actual);api.recordCompletedHand(h);}
 if((i+1)%50===0)console.log('completed',i+1);
}
stats.action_rate=stats.bets/stats.opportunities;stats.hit_rate_on_bets=stats.non_tie_bets?stats.correct_bets/stats.non_tie_bets:null;stats.realized_ev_per_bet=stats.bets?stats.profit/stats.bets:null;stats.elapsed_ms=Date.now()-start;
fs.writeFileSync('validation/bayes_markov_smoke.json',JSON.stringify(stats,null,2)+'\n');console.log(JSON.stringify(stats));if(stats.errors)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
