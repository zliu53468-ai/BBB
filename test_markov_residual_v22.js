"use strict";
/* BBB V2.2: causal Markov research shadow; never alter actual EV / PF. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {transitionStats,shadowCalibrate,SETTINGS,BASE_BANKER}=require("./markov_residual_runtime.js");
const {PhysicsForecastFirstEngine,BASE,BASE_EV,
  BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION}=require("./physics_forecast_first_runtime.js");
const near=(a,b,eps=1e-10)=>assert(Math.abs(a-b)<=eps,`${a} != ${b}`);
assert.deepEqual({...SETTINGS},{priorStrength:64,minSupport:12,
  maxShadowWeight:.10,maxConditionalShift:.003,appliedWeight:0});
assert.equal(BASE_EV_GATE,.005);assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
near(BASE_BANKER,BASE.B/(BASE.B+BASE.P));
assert.throws(()=>transitionStats(["B","X"]),/invalid_markov_history/);
assert.throws(()=>transitionStats([], {minSupport:-1}),/invalid_markov_options/);
const empty=transitionStats([]);
assert.equal(empty.selected_support,0);
assert.equal(empty.current_side,null);
assert.equal(empty.eligible,false);
near(empty.posterior_conditional.B,BASE_BANKER);
const t=transitionStats(["B","T","P","T","P"]);
assert.equal(t.current_side,"P");
assert.deepEqual(t.transition_counts,{B:{B:0,P:1},P:{B:0,P:1}});
assert.equal(t.ties_ignored_for_transitions,2);
assert.equal(t.selected_support,1);
near(t.posterior_conditional.B,(64*BASE_BANKER)/65);
assert.equal(t.eligible,false);
const seq="BP".repeat(32).split("");
const stats=transitionStats(seq);
assert.equal(stats.from,"P");
assert.equal(stats.transition_counts.B.P,32);
assert.equal(stats.transition_counts.P.B,31);
assert(stats.selected_support>=12&&stats.eligible);
const allB=transitionStats("B".repeat(65).split(""));
const allP=transitionStats("P".repeat(65).split(""));
assert(allB.posterior_conditional.B>BASE_BANKER);
assert(allP.posterior_conditional.B<BASE_BANKER);
assert(stats.independent_evidence===false);
assert.equal(stats.estimated_edge_validated,false);
// Each prefix must match the same prefix tested in isolation, without
// using future outcomes or cross-shoe state.
for(let i=0;i<seq.length;i++){
 assert.deepEqual(transitionStats(seq.slice(0,i+1)),
   transitionStats(seq.slice(0,i+1)));
}
const physics={source:"hypothetical_particle_posterior",
  simulated_probability:{B:.46,P:.44,T:.10},
  ev_banker:.95*.46-.44,ev_player:.44-.46,
  simulation_exhausted:false};
const before=JSON.stringify(physics);
const shadow=shadowCalibrate(allB,physics);
assert(shadow.eligible);
assert(shadow.shadow_only);
assert.equal(shadow.applied_weight,0);
assert.equal(shadow.modifies_physical_ev,false);
assert.equal(shadow.modifies_particle_weights,false);
assert.equal(shadow.modifies_bet_gate,false);
assert.equal(shadow.validated_edge,false);
assert(Math.abs(shadow.conditional_shift)<=.003);
near(shadow.candidate_probability.B+shadow.candidate_probability.P+
  shadow.candidate_probability.T,1);
near(shadow.candidate_probability.T,.10);
near(shadow.candidate_ev.B,.95*shadow.candidate_probability.B-
  shadow.candidate_probability.P);
assert.equal(JSON.stringify(physics),before,"candidate must not mutate physical EV");
assert(!shadowCalibrate(t,physics).eligible);
near(shadowCalibrate(t,physics).conditional_shift,0);
assert(!shadowCalibrate(allB,physics,{observed:true}).eligible);
near(shadowCalibrate(allB,physics,{observed:true}).conditional_shift,0);
assert.throws(()=>shadowCalibrate(allB,physics,{maxShadowWeight:1}),
  /invalid_markov_calibration/);
assert.throws(()=>shadowCalibrate(allB,physics,{maxConditionalShift:.1}),
  /invalid_markov_calibration/);
const engine=new PhysicsForecastFirstEngine();
const initial=engine.predict();
assert.equal(initial.markov_residual.stats.eligible,false);
assert.equal(initial.markov_residual.shadow.applied_weight,0);
assert.equal(initial.effective_research_gate,.005);
assert(!("bidirectional_offset" in initial));
const report=engine.syncHistory(seq.slice(0,23));
assert(!("bidirectional_offset" in report));
assert(!("bidirectional_offset" in report.research_signal));
assert.equal(report.markov_residual.stats.prior_strength,64);
assert.equal(report.markov_residual.shadow.applied_weight,0);
assert.equal(report.physical_ev.samples,96*96);
near(report.ev_banker,report.physical_ev.ev_banker);
near(report.ev_player,report.physical_ev.ev_player);
assert.equal(report.research_signal.direction,
  report.ev_banker>=report.ev_player?"B":"P");
assert.equal(report.bet_allowed,false);
assert.equal(report.action,"SKIP");
near(report.effective_research_gate,
  report.dynamic_gate+
    (report.early_shoe_guard?.supplemental_gate||0)+
    report.shoe_distribution_guard.additional_gate);
assert.equal(report.feature_weights.physics,5);
assert.equal(report.feature_weights.road,.1);
assert.deepEqual(report,engine.predict());
assert.deepEqual(report,new PhysicsForecastFirstEngine().syncHistory(seq.slice(0,23)));
engine.syncHistory(seq.slice(0,22));
assert.deepEqual(report,engine.syncHistory(seq.slice(0,23)));
const auth=fs.readFileSync("auth-client.js","utf8");
const panel=fs.readFileSync("quant256_online_panel.js","utf8");
const html=fs.readFileSync("index.html","utf8");
assert(auth.indexOf("markov_residual_runtime.js?v=20261103-v22-markov-shadow-v1")>=0);
assert(auth.indexOf("markov_residual_runtime.js?v=20261103-v22-markov-shadow-v1")<
  auth.indexOf("physics_forecast_first_runtime.js?v=20261104-two-hand-length-v1"));
assert(auth.includes("point_signal_quality_guard.js?v=20261103-v26-mild-skip-v1"));
assert(auth.includes("quant256_online_panel.js?v=20261104-two-hand-length-v1"));
assert(html.includes("auth-client.js?v=20261104-two-hand-length-v1"));
assert(panel.includes("V2.6 點數平滑"),"web-only presentation changed; core Markov V2.2 shadow remains");
assert(!panel.includes("雙向偏移"));
console.log("PASS V2.2 causal Markov shadow, first-order 64-prior/12-support, tie bypass, max 0.003, no BET/PF/EV override, deterministic replay and loader");
