"use strict";
/** V2.6 quality-adaptive smoothing is opt-in for one-scored-hand web only. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {earlyShoeGuard,PhysicsForecastFirstEngine,BASE}=require("./physics_forecast_first_runtime.js");
const almost=(a,b,eps=1e-10)=>assert(Math.abs(a-b)<eps,`${a} != ${b}`);
const fake={size:256,particles:Array.from({length:256},()=>[128,...Array(9).fill(32)]),
  ancestorIds:Array.from({length:256},(_,i)=>i),lastLikelihoodRelativeSE:.04};
const raw={simulated_probability:{B:.52,P:.40,T:.08},
  ev_banker:.95*.52-.40,ev_player:.40-.52,
  ev_gap:(.95*.52-.40)-(.40-.52),
  ev_gap_deviation:.05,mc_se_gap:.012,
  posterior_sd_gap:.04,particle_ess:240,posterior_update_ess:230,
  particle_support:256,simulation_exhausted:false};
const legacy=earlyShoeGuard(raw,fake,1);
const normal=earlyShoeGuard(raw,fake,1,{independentFinalScore:false});
assert.deepEqual(legacy,normal,"legacy flag=false MUST preserve old shoe-stage calculation");
const single=earlyShoeGuard(raw,fake,1,{independentFinalScore:true});
assert.equal(single.diagnostics.mode,"independent_final_score_quality");
assert(single.diagnostics.blend>legacy.diagnostics.blend,
  "quality-adaptive single score should no longer be pinned to the 1/20-shoe blend");
assert(single.diagnostics.blend>=.28&&single.diagnostics.blend<=.82);
assert(single.diagnostics.prior_weight>=.18);
assert(Math.abs(single.physical.simulated_probability.B-raw.simulated_probability.B)<
  Math.abs(legacy.physical.simulated_probability.B-raw.simulated_probability.B));
almost(Object.values(single.physical.simulated_probability).reduce((a,b)=>a+b,0),1);
almost(single.physical.ev_banker,.95*single.physical.simulated_probability.B-
  single.physical.simulated_probability.P);
almost(single.physical.ev_player,single.physical.simulated_probability.P-
  single.physical.simulated_probability.B);
const poor={...raw,posterior_update_ess:8,particle_ess:8,
  mc_se_gap:.17,posterior_sd_gap:.25};
const degraded={...fake,ancestorIds:Array(256).fill(0),lastLikelihoodRelativeSE:.90};
const low=earlyShoeGuard(poor,degraded,1,{independentFinalScore:true});
assert(low.diagnostics.blend<single.diagnostics.blend,
  "lower ESS/diversity & higher MC noise must increase prior smoothing");
assert(low.diagnostics.blend>=.28&&low.diagnostics.blend<=.82);
assert.equal(single.diagnostics.validated_accuracy_gain,false);
assert.strictEqual(earlyShoeGuard(raw,fake,1,{observed:true,independentFinalScore:true}).physical,raw,
  "actual-card EV is never smoothed in independent mode");
assert.strictEqual(earlyShoeGuard(raw,fake,20,{independentFinalScore:false}).physical,raw,
  "mature legacy history retains original no-op");
assert.throws(()=>new PhysicsForecastFirstEngine({independentFinalScoreMode:"yes"}),
  /invalid_independent_final_score_mode/);
const a=new PhysicsForecastFirstEngine({independentFinalScoreMode:true});
const b=new PhysicsForecastFirstEngine();
const scored=["P"],point=[{player:8,banker:4}];
const corrected=a.syncHistory(scored,point);
const unchanged=b.syncHistory(scored,point);
assert.equal(corrected.early_shoe_guard.mode,"independent_final_score_quality");
assert.equal(unchanged.early_shoe_guard.mode,undefined);
assert(corrected.early_shoe_guard.blend>unchanged.early_shoe_guard.blend);
assert.equal(corrected.physical_ev.samples,96*96);
assert.equal(corrected.point_score_observation_count,1);
assert.equal(corrected.action,"SKIP");
assert.equal(corrected.bet_allowed,false);
assert.equal(corrected.feature_weights.physics,5);
assert.equal(corrected.feature_weights.road,.1);
assert.deepEqual(corrected,a.predict(),"repeated read does not advance PF");
assert.deepEqual(corrected,new PhysicsForecastFirstEngine({independentFinalScoreMode:true})
  .syncHistory(scored,point),"identical scores & seed reproduce corrected EV");
const noPoints=new PhysicsForecastFirstEngine({independentFinalScoreMode:true}).syncHistory(["B"]);
const legacyNoPoints=new PhysicsForecastFirstEngine().syncHistory(["B"]);
assert.deepEqual(noPoints,legacyNoPoints,
  "the opt-in alone must not change standard B/P/T-only calculations");
const panel=fs.readFileSync("quant256_online_panel.js","utf8");
assert(panel.includes("independentFinalScoreMode:true"),"web panel must explicitly opt in");
assert(panel.includes("const fresh=new PhysicsForecastFirstEngine(CONFIG)"),
  "each score still creates a new PF256 estimator");
console.log("PASS V2.6: quality-adaptive one-score guard only; preserved legacy B/P/T and true-card EV, 96/256/96, EV/BET, deterministic replay and ESS/MC reliability");
