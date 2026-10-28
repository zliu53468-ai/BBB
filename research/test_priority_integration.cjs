'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),{runtime,settle}=require('./priority_runtime.cjs');
(async()=>{
 const old=await runtime({legacy:true}),next=await runtime(),plain=await runtime({memoized:false});
 next.api.getCardcountProfile().physics_priority_v1.enabled=false;
 for(const h of ['', 'B','P','T','BBPPBTBP','BPT'.repeat(13)]){
  const seq=h.split('');for(const r of [old,next,plain]){r.data.clear();r.data.set('bgs_xgb_final_shoe_id_v5','parity');}
  const a=old.api.applyFinalPrediction(seq),b=next.api.applyFinalPrediction(seq);
  for(const key of ['direction','ev_banker','ev_player','probabilities'])assert.deepEqual(b[key],a[key],'flag-off rollback '+key);
  assert.equal(b.finalProbability.finalPB,a.finalProbability.finalPB);
  const p=next.api.predictPhysics(seq),cfg={...next.api.getCardcountProfile().pf,max_blend:next.api.profileValue('pf_max',seq.length+1),forecast_draws:next.api.profileValue('pf_draws',seq.length+1)};
  assert.deepEqual(next.win.__BGS_PARTICLE500__.estimate(seq,p,'quality',cfg),plain.win.__BGS_PARTICLE500__.estimate(seq,p,'quality',cfg),'memoized PF identical');
 }
 next.api.getCardcountProfile().physics_priority_v1.enabled=true;
 const row=JSON.parse(fs.readFileSync('research/data/priority_v1/smoke-observations.jsonl','utf8').split('\n')[0]);
 next.data.clear();next.data.set('bgs_xgb_final_shoe_id_v5','point-test');let history=[];let pointCases=0;
 for(const r of row.rows.slice(0,8)){
  const p=next.api.applyFinalPrediction(history);assert(p.physics_priority);
  if(history.length){assert.equal(p.physics_priority.point_weight,.65);assert.equal(p.physics_priority.point_diagnostics.point_observations_processed,history.length);pointCases++;}
  settle(next,history,p,r.outcome,{outcome:r.outcome,player_score:r.player_score,banker_score:r.banker_score,card_count:r.card_count});history.push(r.outcome);
 }
 const p=next.api.applyFinalPrediction(history),q=next.api.applyFinalPrediction(history);assert.equal(p.direction,q.direction);assert.equal(p.finalProbability.finalPB,q.finalProbability.finalPB);
  next.api.setNoSkipTestMode(true);assert(!next.api.applyFinalPrediction(history).physics_priority,'existing no-Skip experiment stays separate');next.api.setNoSkipTestMode(false);
  const originalPoint=next.api.applyFinalPrediction(history).physics_priority.point_diagnostics.point_vs_bpt_diagnostic.point_conditioned_probabilities;
  const short=history.slice(0,4);
  next.data.set('bgs256d_frozen_6x15_forward_v18',JSON.stringify({history:short}));
  next.api.undoRuntimeHistory();next.api.applyFinalPrediction(short);
  for(let i=4;i<history.length;i++){const event=row.rows[i],before=history.slice(0,i),pred=next.api.applyFinalPrediction(before);settle(next,before,pred,event.outcome,{outcome:event.outcome,player_score:event.player_score,banker_score:event.banker_score,card_count:event.card_count});}
  const replay=next.api.applyFinalPrediction(history).physics_priority.point_diagnostics.point_vs_bpt_diagnostic.point_conditioned_probabilities;
  assert.deepEqual(originalPoint,replay,'point posterior undo/replay must be exact');
 const exportData=JSON.parse(next.api.exportTrainingData());assert(exportData.rows.at(-1).physics_priority);
 next.api.getCardcountProfile().physics_priority_v1.enabled=false;assert(!next.api.applyFinalPrediction(history).physics_priority);
 console.log(JSON.stringify({passed:true,flag_off_legacy_parity:true,pf_memo_exact:true,point_cases:pointCases,point_weight:.65,point_undo_replay:true,no_skip_experiment_preserved:true,causal_observation_count:true,snapshot_export:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
