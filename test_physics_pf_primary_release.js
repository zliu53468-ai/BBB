"use strict";
/**
 * Direct Physics/PF10000 promotion contract. No change to rank inference:
 * B/P/T-only PF is a conditional hidden-shoe simulation, not observed cards.
 * Check that formal guards and local calibration remain active.
 */
const assert=require("node:assert/strict");
const profile=require("./cardcount_profile.json");
const {make,plain}=require("./test_pf_primary_runtime");

assert.equal(profile.bayes_markov_final_v1.enabled,false,"Bayesian Markov no longer owns direction");
assert.equal(profile.road_primary_v1.enabled,false,"road primary no longer owns direction");
assert.equal(profile.pf_primary_v1.enabled,false,"do not substitute PF-first with Physics-secondary test mode");
assert.equal(profile.hitrate_max_v1.enabled,true,"retain selective entry contract");
assert.equal(profile.hitrate_max_v1.markov.enabled,false,"no Markov boost after PF");
assert.equal(profile.local_calibration.enabled,true,"local calibration preserved");
assert.equal(profile.hitrate_max_v1.volume_guard_selective,true,"Volume Guard stays selective");
assert.equal(profile.hitrate_max_v1.require_first_final_consistency,true);
assert.equal(profile.hitrate_max_v1.min_positive_ev,.004);
assert.deepEqual(profile.hitrate_max_v1.entry_thresholds,{early:.012,middle:.009,late:.007});
assert.equal(profile.pf_release.particle_count,10000);

(async()=>{
  const {api,ctx}=await make(),examples=[
    [],["B"],Array(16).fill("B"),Array(24).fill("P"),
    Array.from({length:28},(_,i)=>i%2?"P":"B"),
    Array.from({length:28},(_,i)=>Math.floor(i/2)%2?"P":"B"),
    "BPPTBBTBPPTBBPPBBPTPBP".split("")
  ];
  assert.equal(api.bayesMarkovConfig(),null);
  assert.equal(api.roadPrimaryConfig(),null);
  assert.equal(api.pfPrimaryConfig(),null);
  assert(api.hitrateConfig());
  for(const h of examples){
    ctx.__testCalls=[];
    const p=api.applyFinalPrediction(h),r=p.finalProbability;
    assert.equal(r.mode,"physics_primary");
    assert.equal(r.primary_source,"direct_physics_particle500_physical_ev");
    assert.equal(r.particleDiagnostics.particle_count,10000);
    assert(r.particleDiagnostics.rebuild_from_scratch===true);
    assert(r.execution_order.includes("direct_physics"));
    assert(r.execution_order.includes("particle_filter"));
    assert(r.execution_order.includes("physical_ev"));
    assert(r.execution_order.includes("physics_candidate"));
    assert(r.execution_order.includes("final_ev_guard"));
    assert(r.execution_order.includes("volume_guard"));
    assert(!r.execution_order.includes("markov_prior"),"road/trend must not override physics");
    assert(!r.execution_order.includes("road_primary_candidate"));
    assert(!r.execution_order.includes("bayesian_markov_1_3"));
    assert.equal(p.bayes_markov,undefined);
    assert.equal(p.road_primary,undefined);
    assert(["B","P","Skip"].includes(p.direction));
    assert(p.direction==="Skip"?p.stake_multiplier===0:true);
    assert.equal(r.xgb_role,"advisory_only_no_veto_no_flip");
    assert(Number.isFinite(r.finalPB));
    assert(r.finalPB>=0&&r.finalPB<=1);
    assert(Number.isFinite(p.ev_banker)&&Number.isFinite(p.ev_player));
    const mass=1-r.p_tie;
    assert(Math.abs(p.ev_banker-(.95*mass*r.finalPB-mass*(1-r.finalPB)))<1e-10);
    assert(Math.abs(p.ev_player-(mass*(1-2*r.finalPB)))<1e-10);
    assert(ctx.__testCalls.includes("physics")&&ctx.__testCalls.includes("pf"));
    const repeated=api.applyFinalPrediction(h);
    assert.deepEqual(plain({dir:repeated.direction,pb:repeated.finalProbability.finalPB,ev:[repeated.ev_banker,repeated.ev_player]}),
      plain({dir:p.direction,pb:r.finalPB,ev:[p.ev_banker,p.ev_player]}),"repeatability");
  }
  const fallback=await make({noPF:true});
  const p=fallback.api.applyFinalPrediction(["B","P","B"]);
  assert.equal(p.finalProbability.mode,"physics_primary","PF outage stays on Physics, not road");
  assert(!p.finalProbability.execution_order.includes("markov_prior"));
  assert(p.finalProbability.execution_order.includes("final_ev_guard"));
  assert.equal(fallback.api.getCardcountProfile().local_calibration.enabled,true);
  console.log("PASS guarded Direct Physics + PF10000 primary activation, no Markov dominance, EV/Volume, calibration enabled, deterministic playback and PF fallback");
})().catch(e=>{console.error(e);process.exitCode=1;});
