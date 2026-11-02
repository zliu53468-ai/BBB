"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {BlindPhysicsEstimator}=require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine,summarizeHypotheses,
  BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION,weightedContext}=
  require("./physics_forecast_first_runtime.js");
const src=fs.readFileSync("./physics_forecast_first_runtime.js","utf8");
assert(src.includes("function hashComposition(shoes)"));
assert(!src.includes("function hashHistory("));
assert(src.includes("const stableSeed=hashComposition(validParticles);"));
assert(src.includes("seed:hashComposition(counts)^0xb15f13"));
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
assert.equal(new PhysicsForecastFirstEngine().simulationsPerParticle,128);
// Same virtual remaining shoe MUST return the same simulated physical EV,
// independent of an arbitrary history label text supplied in the legacy API.
const blind=new BlindPhysicsEstimator({particles:24,seed:0xBBB256});
for(const outcome of ["B","P","T","B","P","B","P"])blind.observe(outcome);
const a=summarizeHypotheses(blind,["B","B","B"],{samplesPerParticle:64});
const b=summarizeHypotheses(blind,["P","P","P"],{samplesPerParticle:64});
const c=summarizeHypotheses(blind,["T"],{samplesPerParticle:64});
assert.deepEqual(a,b,"same remaining counts => same MC EV regardless of history");
assert.deepEqual(a,c,"ties in synthetic history may not reseed EV");
assert.equal(a.samples,24*64);
assert.equal(a.source,"hypothetical_particle_posterior");
const engine=new PhysicsForecastFirstEngine();
assert.equal(engine.predict().base_ev_gate,.005);
const report=engine.syncHistory(["B","P","T","B","B","P"]);
assert.equal(report.feature_weights.physics,5);
assert.equal(report.feature_weights.road,.1);
assert.equal(report.research_signal.source,"simulated_bpt_only");
assert.equal(report.action,"SKIP","B/P/T alone cannot prove real card edge");
assert.equal(report.observed_card_evidence,false);
assert(src.includes('const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";'));
assert.deepEqual(report,engine.predict(),"read-only predict must be repeatable");
console.log("PASS V2 seed isolation: same composition => same EV; original V2 gates, 5x/0.1x, 64 samples retained");
