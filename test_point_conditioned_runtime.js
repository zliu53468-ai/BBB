"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const storage=new Map(),history=["B","P","T","B","P","B","T","P"];
const points=[
  [2,6],[8,3],[4,4],[1,7],[9,2],[3,8],[5,5],[7,1],
];
const events=history.map((outcome,i)=>({outcome,player_score:points[i][0],banker_score:points[i][1],card_count:null}));
storage.set("bgs256d_frozen_6x15_forward_v18",JSON.stringify({history,last_selected:"",selection_streak:0}));
storage.set("bgs_xgb_final_shoe_id_v5","shoe_point_runtime_test");
storage.set("bgs_macro_observations_v1",JSON.stringify({shoe_id:"shoe_point_runtime_test",events}));
global.window=global;
global.localStorage={getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,String(v)),removeItem:k=>storage.delete(k)};
global.document={getElementById:()=>null};
global.fetch=async file=>({ok:true,status:200,json:async()=>JSON.parse(fs.readFileSync(file,"utf8"))});
const ctx={console,localStorage:global.localStorage,document:global.document,fetch:global.fetch,Math,Date,Promise,URL,Blob};
ctx.window=ctx;vm.createContext(ctx);
for(const file of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])
  vm.runInContext(fs.readFileSync(file,"utf8"),ctx,{filename:file});
(async()=>{
  const api=ctx.__BGS_FINAL56__;
  await api.loadModels();
  const point=api.applyFinalPrediction(history);
  const diagnostics=point.finalProbability.particleDiagnostics;
  assert.equal(point.finalProbability.point_posterior_active,true);
  assert.equal(diagnostics.posterior_update_method,"ten_class_score_pair_likelihood_and_conditional_hand_sampling");
  assert(point.finalProbability.execution_order.includes("point_conditioned_particle_filter"));
  assert.equal(point.finalProbability.primary_source,"score_conditioned_point_particle_posterior");
  assert.equal(point.direction_bias_audit.observation_basis,"latest_hand_point_scores_only");
  assert.equal(diagnostics.point_history_mode,"latest_hand_only");
  assert.equal(diagnostics.point_source_history_rounds,history.length);
  assert.equal(diagnostics.history_rounds,1,"point PF resets and conditions on one hand only");
  assert.equal(diagnostics.point_observations_processed,1,"older score observations must not accumulate into point PF");
  assert.equal(diagnostics.observed_point_history.length,1);
  assert.equal(point.finalProbability.independent_hand_prediction_active,true);
  assert.equal(point.finalProbability.forecast_history_mode,"independent_hand");
  assert.equal(diagnostics.forecast_source_history_rounds,0,"older B/P/T results must not enter the point-mode forecast prior");
  assert.equal(point.finalProbability.smoothingProfile,"independent_per_hand","point-mode predictions must not inherit the previous hand's EMA");
  assert.equal(point.first_layer_ev.smoothed.initialized,true,"independent first-layer EV starts without a prior hand gap");
  assert.equal(api.prepareCalibrationTicket(history,point),null,"independent predictions must not seed cross-hand calibration");
  assert.equal(api.prepareFirstSmoothingTicket(history,point),null,"independent predictions must not seed cross-hand smoothing");
  const posterior=point.finalProbability.particleDiagnostics.point_vs_bpt_diagnostic.point_conditioned_probabilities;
  assert(Math.abs(point.first_layer_ev.probabilities.B-posterior.B)<1e-12);
  assert(Math.abs(point.first_layer_ev.probabilities.P-posterior.P)<1e-12);
  assert.equal(point.settlement_audit.pf_stage_applications,1);
  if(["B","P"].includes(point.direction))assert.equal(point.direction,point.first_layer_ev.direction,"accepted point-mode direction must match first-layer direction");
  else assert.equal(point.direction_consistency.consistent,false,"point-mode Skip is retained if the first/final direction is not identical");
  const boost=api.selectiveEvBoost(10,.8,.4,true);
  assert.equal(boost.base_boost,.006);assert.equal(boost.extra_boost,.006);
  assert(api.scoreConditionedEntryPolicy().volume_guard_target_action_rate.early<=.25);

  const alternateHistory=["B","B","B","B","B","B","B","P"];
  storage.set("bgs_xgb_final_shoe_id_v5","shoe_point_runtime_alt");
  storage.set("bgs_macro_observations_v1",JSON.stringify({shoe_id:"shoe_point_runtime_alt",events:alternateHistory.map((outcome,i)=>({
    outcome,player_score:i===alternateHistory.length-1?7:null,banker_score:i===alternateHistory.length-1?1:null,card_count:null
  }))}));
  const independentAgain=api.applyFinalPrediction(alternateHistory);
  assert.deepEqual(independentAgain.finalProbability.physicsDirect,point.finalProbability.physicsDirect,"history-free Physics prior must ignore older B/P/T results");
  assert.deepEqual(independentAgain.finalProbability.physics,point.finalProbability.physics,"same latest points must produce the same PF posterior across different outcome histories");
  assert.equal(independentAgain.finalProbability.finalPB,point.finalProbability.finalPB);
  assert.equal(independentAgain.direction,point.direction);

  storage.set("bgs_xgb_final_shoe_id_v5","shoe_point_runtime_test");
  storage.set("bgs_macro_observations_v1",JSON.stringify({shoe_id:"shoe_point_runtime_test",events}));
  const profile=api.getCardcountProfile();profile.pf.independent_point_update_enabled=false;
  const cumulative=api.applyFinalPrediction(history),cumulativeDiagnostics=cumulative.finalProbability.particleDiagnostics;
  assert.equal(cumulativeDiagnostics.point_history_mode,"full_shoe","independent-update switch retains the prior cumulative path when disabled");
  assert.equal(cumulativeDiagnostics.point_observations_processed,history.length);
  profile.pf.independent_point_update_enabled=true;

  profile.pf.independent_hand_prediction_enabled=false;
  const historyBased=api.applyFinalPrediction(history);
  assert.equal(historyBased.finalProbability.independent_hand_prediction_active,false);
  assert.deepEqual(historyBased.finalProbability.physicsDirect,api.predictPhysics(history),"independent-prediction switch restores history-conditioned Physics input");
  profile.pf.independent_hand_prediction_enabled=true;

  ctx.__BBB_POINT_INPUT__={getInputMode:()=>"compatibility"};
  const compatible=api.applyFinalPrediction(history);
  assert.equal(compatible.finalProbability.point_posterior_active,false);
  assert.equal(compatible.finalProbability.particleDiagnostics.posterior_update_method,"observed_bpt_likelihood_and_resampling");
  assert.deepEqual(compatible.finalProbability.physicsDirect,api.predictPhysics(history),"B/P/T compatibility mode must retain the prior history-conditioned Physics path");
  console.log(JSON.stringify({passed:true,point_pf_primary:true,latest_hand_only:true,independent_forecast_history:true,cumulative_rollback_switch:true,first_layer_uses_point_posterior:true,
    strict_direction_consistency:true,score_noise_ev_boost:true,compatibility_uses_legacy_pf:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
