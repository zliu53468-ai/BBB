"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const {evaluate,WINDOWS,VERSION}=require("./shoe_distribution_guard.js");
const {PhysicsForecastFirstEngine,summarizeHypotheses,
 BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION}=
 require("./physics_forecast_first_runtime.js");
const {BlindPhysicsEstimator}=require("./quant256_online_runtime.js");
const near=(a,b,eps=1e-10)=>assert(Math.abs(a-b)<eps,`${a} != ${b}`);
assert.equal(VERSION,"BBB_V2_DYNAMIC_SHOE_DISTRIBUTION_GUARD_V1");
assert.deepEqual(WINDOWS,[8,16,24]);
assert.throws(()=>evaluate(["B","X"]),/invalid_distribution_history/);
assert.throws(()=>evaluate(Array(501).fill("B")),/invalid_distribution_history/);
const empty=evaluate([]);
assert.equal(empty.current_stage,"insufficient_data");
assert.equal(empty.additional_gate,0);
assert.equal(empty.predicted_next_direction,null);
const balanced=evaluate("BP".repeat(35).split(""));
assert.equal(balanced.current_stage,"no_strong_change_evidence");
assert.equal(balanced.additional_gate,0);
assert(balanced.windows.every(w=>w.eligible&&!w.significant));
const reversed=evaluate(("B".repeat(24)+"P".repeat(24)).split(""));
assert.equal(reversed.current_stage,"distribution_shift");
assert(reversed.strongest_window===16||reversed.strongest_window===24);
assert(reversed.change_probability>.8);
assert(reversed.additional_gate>0&&reversed.additional_gate<=.004);
assert.equal(reversed.predicted_next_direction,null,
 "detecting historical imbalance must never predict a future flip");
assert(reversed.windows.some(w=>w.significant&&w.difference<0));
const inverted=evaluate(("P".repeat(24)+"B".repeat(24)).split(""));
assert.equal(inverted.current_stage,"distribution_shift");
assert(inverted.windows.some(w=>w.significant&&w.difference>0));
assert(reversed.additional_gate<=.004);
const ties=evaluate(("B".repeat(24)+"T".repeat(6)+"P".repeat(24)).split(""));
near(ties.additional_gate,reversed.additional_gate);
assert.equal(ties.non_tie_rounds,reversed.non_tie_rounds);
const real=evaluate(("B".repeat(24)+"P".repeat(24)).split(""),{observed:true});
assert.equal(real.additional_gate,0);
assert.equal(real.observed_card_bypass,true);
const seq=("BP".repeat(12)+"B".repeat(12)+"P".repeat(12)).split("");
const prior=evaluate(seq.slice(0,24));
assert.deepEqual(prior,evaluate(seq.slice(0,24)),
 "the monitor must be deterministic and causal");
assert.notDeepEqual(prior.windows,evaluate(seq).windows,
 "later observations should affect later diagnostics only");
assert.equal(new BlindPhysicsEstimator().size,256);
assert.equal(new PhysicsForecastFirstEngine().simulationsPerParticle,160);
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
const engine=new PhysicsForecastFirstEngine();
const run=evaluate(["B","P"]);
const first=engine.syncHistory(["B","P"]);
assert(first.shoe_distribution_guard);
assert.equal(first.shoe_distribution_guard.additional_gate,0);
assert.equal(first.research_signal.distribution_guard.current_stage,"insufficient_data");
assert.equal(first.research_signal.direction,
  first.ev_banker>=first.ev_player?"B":"P");
assert.equal(first.action,"SKIP"); // B/P/T alone is never real card-count EV
const one=engine.syncHistory("BP".repeat(11).split(""));
const original=summarizeHypotheses(engine.quant.physics,[],{samplesPerParticle:160});
near(one.ev_banker,original.ev_banker);
near(one.ev_player,original.ev_player);
near(one.ev_gap_deviation,original.ev_gap_deviation);
assert.equal(one.physical_ev.samples,40960);
near(one.effective_research_gate,
  one.dynamic_gate+(one.early_shoe_guard?.supplemental_gate||0)+
  one.shoe_distribution_guard.additional_gate);
assert.deepEqual(one,engine.predict(),"read-only forecast cannot change");
const final=engine.syncHistory("B".repeat(24).concat("P".repeat(24)).split(""));
const base=summarizeHypotheses(engine.quant.physics,[],{samplesPerParticle:160});
near(final.ev_banker,base.ev_banker);
near(final.ev_player,base.ev_player);
assert.equal(final.shoe_distribution_guard.current_stage,"distribution_shift");
assert(final.shoe_distribution_guard.additional_gate>0);
assert.equal(final.research_signal.direction,
  final.ev_banker>=final.ev_player?"B":"P",
  "statistical regime monitor cannot flip the original physical direction");
assert.equal(final.feature_weights.physics,5);
assert.equal(final.feature_weights.road,.1);
assert.equal(final.action,"SKIP");
assert.deepEqual(final,engine.predict(),"same input must be cached");
const copy=new PhysicsForecastFirstEngine().syncHistory(
 "B".repeat(24).concat("P".repeat(24)).split(""));
assert.deepEqual(copy,final,"same history replay must match after any phase shifts");
const auth=fs.readFileSync("auth-client.js","utf8");
const page=fs.readFileSync("index.html","utf8");
const panel=fs.readFileSync("quant256_online_panel.js","utf8");
assert(auth.indexOf("shoe_distribution_guard.js?v=20261103-dynamic-shoe-guard-v1")>=0);
assert(auth.indexOf("shoe_distribution_guard.js?v=20261103-dynamic-shoe-guard-v1")<
  auth.indexOf("physics_forecast_first_runtime.js?v=20261103-v22-markov-shadow-v1"));
assert(auth.includes("quant256_online_runtime.js?v=20261103-v21-rb-move-v1"));
assert(auth.includes("quant256_online_panel.js?v=20261103-v22-markov-shadow-v1"));
assert(page.includes("auth-client.js?v=20261103-v22-markov-shadow-v1"));
assert(page.includes("viewport-fit.js?v=20261102-mobile-stable-v2"));
assert(panel.includes("分布監測"));
assert(!page.includes("side-bet-card"));
console.log("PASS V2 dynamic distribution monitor: 8/16/24 causal Bayes shift, unchanged physical EV formula/256-384-160, no direction flip, replay, auth and mobile UI");
