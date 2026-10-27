"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const fixture=require("./validation/pf_update_legacy_fixture.json");
const context={};context.window=context;vm.createContext(context);
vm.runInContext(fs.readFileSync("particle_filter_runtime.js","utf8"),context);
const pf=context.__BGS_PARTICLE500__;
const outcomeOf=(p,b)=>b>p?"B":p>b?"P":"T";
const observations=fixture.history.map(outcome=>({outcome}));
observations[fixture.history.length-1]={outcome:"P",player_score:7,banker_score:4,card_count:5};
const old=pf.estimate(fixture.history,fixture.direct,"quality",{likelihood_draws:4,fallback_draws:4,forecast_draws:3});
const point=pf.estimatePoints(fixture.history,fixture.direct,"quality",{likelihood_draws:4,fallback_draws:4,forecast_draws:3},observations);
const pointAgain=pf.estimatePoints(fixture.history,fixture.direct,"quality",{likelihood_draws:4,fallback_draws:4,forecast_draws:3},observations);
assert.equal(point.diagnostics.version,"POINT_SCORE_CONDITIONED_PF500_V1");
assert.equal(point.diagnostics.posterior_update_method,"ten_class_score_pair_likelihood_and_conditional_hand_sampling");
assert.equal(point.diagnostics.point_observations_processed,1);
assert.equal(point.diagnostics.bpt_only_observations,fixture.history.length-1);
assert.equal(point.diagnostics.particle_count,500);assert.equal(point.diagnostics.persistent_state,false);
assert.equal(point.diagnostics.rebuild_from_scratch,true);assert(point.diagnostics.recent_ess_ratio>0&&point.diagnostics.recent_ess_ratio<=1);
const concentratedHistory=Array(28).fill("B"),concentratedObservations=concentratedHistory.map(()=>({outcome:"B",player_score:0,banker_score:9,card_count:4}));
const resampled=pf.estimatePoints(concentratedHistory,fixture.direct,"quality",{},concentratedObservations);
assert(resampled.diagnostics.resampling_count>0,"low ESS must trigger systematic resampling");
assert(resampled.diagnostics.recent_ess_ratio<.55,"quality uses the pre-resampling ESS rather than the reset weights");
assert.equal(resampled.diagnostics.post_resample_ess_ratio,1,"resampled weights have full ESS by construction");
assert.deepEqual(JSON.parse(JSON.stringify(point)),JSON.parse(JSON.stringify(pointAgain)),"same score history must replay deterministically");
assert.notEqual(JSON.stringify(point.posterior),JSON.stringify(old.posterior),"point scores must change the posterior relative to B/P/T only");
assert(point.diagnostics.point_vs_bpt_diagnostic.max_absolute_probability_delta>0);
assert.equal(point.diagnostics.point_vs_bpt_diagnostic.bpt_only_method,"legacy_13_rank_pf500_bpt_likelihood");
assert.equal(point.diagnostics.point_vs_bpt_diagnostic.direction_changed,
  (point.diagnostics.point_vs_bpt_diagnostic.bpt_only_probabilities.B-point.diagnostics.point_vs_bpt_diagnostic.bpt_only_probabilities.P)*
  (point.diagnostics.point_vs_bpt_diagnostic.point_conditioned_probabilities.B-point.diagnostics.point_vs_bpt_diagnostic.point_conditioned_probabilities.P)<0);
for(const [start,size] of [[0,3],[3,10],[13,10],[23,3]])
  assert(Math.abs(point.posterior.slice(start,start+size).reduce((a,b)=>a+b,0)-1)<1e-9,`posterior block ${start} must normalize`);
const sums=point.posterior;
assert(Math.abs(sums[39]-(.95*sums[23]-sums[24]))<1e-12,"banker EV uses 5% commission");
assert(Math.abs(sums[40]-(sums[24]-sums[23]))<1e-12,"player EV uses zero commission");
assert.equal(pf.validatePointObservation(7,4,5).valid,true);
assert.equal(pf.validatePointObservation(0,0,4).valid,false,"4-card 0-0 hand violates baccarat draw rules");
assert.equal(pf.validatePointObservation(10,0,5).valid,false);
const mixed=observations.slice();mixed[fixture.history.length-1]={outcome:"P",player_score:7,banker_score:4};
const noCardCount=pf.estimatePoints(fixture.history,fixture.direct,"quality",{forecast_draws:3},mixed);
assert.equal(noCardCount.diagnostics.total_cards_conditioning_used,false,"card count remains optional");
assert(outcomeOf(7,4)==="P");
console.log(JSON.stringify({passed:true,score_likelihood:true,conditional_point_sampling:true,ess_and_resampling:true,
  deterministic_replay:true,bpt_counterfactual:true,optional_total_cards:true,commission_ev:true}));
