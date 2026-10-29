'use strict';
const assert=require('assert/strict');
const {make,plain}=require('./test_pf_primary_runtime');
(async()=>{
 const {api,ctx,store}=await make({independent:true});
 const config=api.getCardcountProfile();
 const signature=p=>plain({physics:p.finalProbability.physics,probability:p.finalProbability.finalPB,ev:[p.ev_banker,p.ev_player],direction:p.direction,forecast:p.next_hand_forecast});
 for(const outcome of ['B','P','T']){
  const a=api.applyFinalPrediction([outcome]);
  const b=api.applyFinalPrediction(('BPT'.repeat(12)+outcome).split(''));
  assert.deepEqual(signature(a),signature(b));
  assert.equal(b.latest_hand_pf.history_used,1);
  const d=b.finalProbability.particleDiagnostics;
  assert.equal(d.observation_scope,'latest_hand_only');
  assert.equal(d.history_rounds,1);
  assert.equal(d.particle_count,10000);
  assert(d.latest_hand_reconstruction);
  assert(Math.abs(d.latest_hand_reconstruction.card_count.reduce((a,b)=>a+b,0)-1)<1e-9);
  assert.equal(b.finalProbability.smoothingAlpha,1);
  assert.equal(b.finalProbability.xgb_influence,0);
  assert.equal(b.browser_calibration.applied,false);
  assert(b.pf_primary.physics_weight_effective<=.12);
  api.registerPrediction([outcome],a);
  api.settlePending(outcome,null,[outcome]);
  assert.deepEqual(signature(api.applyFinalPrediction([outcome])),signature(a));
 }
 const a=api.applyFinalPrediction(['P']);
 api.setNoSkipTestMode(true);
 assert.deepEqual(signature(api.applyFinalPrediction(['P'])),signature(a));
 api.setNoSkipTestMode(false);
 const point=api.applyFinalPrediction(['B'],null,{player:3,banker:7});
 assert.equal(point.finalProbability.particleDiagnostics.observed_points.banker,7);
 assert(Math.abs(point.finalProbability.particleDiagnostics.latest_hand_reconstruction.player_points[3]-1)<1e-9);
 assert.equal(api.applyFinalPrediction(['B'],null,{player:8,banker:2}).direction,'Skip');
 assert.equal(api.applyFinalPrediction([]).latest_hand_pf.history_used,0);
 config.latest_hand_pf_v1.enabled=false;
 assert.equal(api.latestHandConfig(),null);
 if(process.env.BBB_BASELINE_ROOT){const old=await make({root:process.env.BBB_BASELINE_ROOT});for(const h of [[],['B'],['B','P','T']])assert.deepEqual(signature(api.applyFinalPrediction(h)),signature(old.api.applyFinalPrediction(h)));}
 const fs=require('fs');
 for(const f of ['app256forward.js','app256continuation.js','auth-client.js','index.html','physics_multitask_model.json','final_probability_model.json'])if(process.env.BBB_BASELINE_ROOT)assert.equal(fs.readFileSync(f,'utf8'),fs.readFileSync(process.env.BBB_BASELINE_ROOT+'/'+f,'utf8'));
 console.log('PASS latest-hand B/P/T independence, reset/replay, no EMA/calibration feedback, points conditioning, fail-closed mismatch, rollback parity, protected files unchanged');
})().catch(e=>{console.error(e);process.exitCode=1;});
