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
  if(["B","P"].includes(point.direction)&&point.direction!==point.first_layer_ev.direction){
    assert.equal(point.direction_consistency.mismatch,true,"a direction mismatch must remain visible");
    assert.equal(point.direction_consistency.threshold_boost,.004,"mismatches require the added EV margin");
    assert(point.ev_direction.best_ev>point.finalProbability.effective_activation_ev,"a mismatch action must clear the stricter EV threshold");
  }
  const entry=point.finalProbability.entry_diagnostics;
  assert.equal(entry.first_layer_raw_direction,point.first_layer_ev.raw.direction);
  assert.equal(entry.first_layer_smoothed_direction,point.first_layer_ev.direction);
  assert.equal(entry.final_action,point.direction);
  assert.equal(entry.consistency_blocked,false,"default point mode applies the configured EV penalty instead of a hard block");
  assert.equal(point.next_hand_forecast.score_conditioned_point_posterior,true);
  assert.equal(api.noSkipTestDirection({available:true,selection_method:"post_draw_ev_argmax_v1",banker_ev:.001,player_ev:.004,score_conditioned_point_posterior:true}),"Balanced","near-equal point-mode EVs display Balanced");
  assert.equal(api.noSkipTestDirection({available:true,selection_method:"post_draw_ev_argmax_v1",banker_ev:.01,player_ev:.003,score_conditioned_point_posterior:true}),"B","outside the balance band the higher EV still selects the side");
  assert.equal(api.noSkipTestDirection(point.next_hand_forecast),Math.abs(point.next_hand_forecast.banker_ev-point.next_hand_forecast.player_ev)<=.005?"Balanced":api.selectEvDirection(point.next_hand_forecast.banker_ev,point.next_hand_forecast.player_ev).direction,"real point forecast applies its configured EV balance band");
  const mismatch=api.directionConsistency("B","P",true);
  assert.equal(mismatch.forced_skip_by_consistency,false);assert.equal(mismatch.threshold_boost,.004);assert.equal(mismatch.mismatch,true);
  api.getCardcountProfile().score_conditioned_entry.direction_consistency_mode="hard_skip";
  assert.equal(api.directionConsistency("B","P",true).forced_skip_by_consistency,true,"hard_skip profile restores the previous consistency rule");
  api.getCardcountProfile().score_conditioned_entry.direction_consistency_mode="ev_penalty";
  api.getCardcountProfile().score_conditioned_entry.enabled=false;
  const rollbackBoost=api.selectiveEvBoost(10,.8,.4,true);
  assert.equal(rollbackBoost.base_boost,.005);assert.equal(rollbackBoost.extra_boost,.004);
  assert.equal(api.noSkipTestDirection({available:true,selection_method:"post_draw_ev_argmax_v1",banker_ev:.001,player_ev:.004,score_conditioned_point_posterior:true}),"P","disabling the new score-conditioned profile restores the prior test behavior");
  api.getCardcountProfile().score_conditioned_entry.enabled=true;
  const boost=api.selectiveEvBoost(10,.8,.4,true);
  assert.equal(boost.base_boost,.003);assert.equal(boost.extra_boost,.0025);
  assert.equal(api.scoreConditionedEntryPolicy().volume_guard_target_action_rate.early,.28);
  storage.set("bgs_xgb_final_training_v5",JSON.stringify(Array.from({length:5},(_,i)=>({shoe_id:"shoe_point_runtime_test",round_index:i+1,final_p_b:.5,p_tie:.1,predicted_direction:"Skip"}))));
  const pointGuard=api.volumeGuardState(10,.5,api.decisionPolicy(10,.2),true);
  assert.equal(pointGuard.target,.28,"point-mode Volume Guard target is no longer capped below 28%");
  storage.delete("bgs_xgb_final_training_v5");
  const uiNodes=Object.fromEntries(["directionOrb","directionText","directionCode","confidence","regime","strength","modePill","roundCount","message","pointPosteriorStatus","modelWeightStatus","localCalibrationStatus"].map(id=>[id,{textContent:"",className:"",dataset:{},setAttribute(){}}]));
  ctx.document.getElementById=id=>uiNodes[id]||null;
  api.renderPrediction(point,history.length);
  assert.match(uiNodes.pointPosteriorStatus.textContent,/第一層 raw/);
  assert.match(uiNodes.pointPosteriorStatus.textContent,/EV差\/門檻/);
  assert.match(uiNodes.pointPosteriorStatus.textContent,/PF品質/);

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
  assert(compatible.finalProbability.entry_diagnostics,"B/P/T output also includes the read-only diagnostic structure");
  api.registerPrediction(history,compatible);api.settlePending("B",null,history);
  const settled=api.getTrainingRows().at(-1);
  assert(settled.entry_diagnostics&&settled.skip_diagnostics.entry_diagnostics,"entry diagnostics persist into local export snapshots");
  console.log(JSON.stringify({passed:true,point_pf_primary:true,latest_hand_only:true,independent_forecast_history:true,cumulative_rollback_switch:true,first_layer_uses_point_posterior:true,
    consistency_ev_penalty:true,score_ev_balance_band:true,diagnostics_visible_and_exported:true,compatibility_uses_legacy_pf:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
