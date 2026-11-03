"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const {BlindPhysicsEstimator,OnlineLinUCB256}=require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine,summarizeHypotheses,
 BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION,VERSION}=require("./physics_forecast_first_runtime.js");
const runtime=fs.readFileSync("./quant256_online_runtime.js","utf8");
const physics=fs.readFileSync("./physics_forecast_first_runtime.js","utf8");
const panel=fs.readFileSync("./quant256_online_panel.js","utf8");
const auth=fs.readFileSync("./auth-client.js","utf8");
const page=fs.readFileSync("./index.html","utf8");

assert.equal(VERSION,"BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE");
assert.equal(BASE_EV_GATE,.005);assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
assert.equal(new BlindPhysicsEstimator().size,96);
assert.equal(new OnlineLinUCB256().physics.size,96);
assert(runtime.includes("const attempts=256,next=new Array(this.size),updated=new Array(this.size);"));
assert(runtime.includes("this.lastESS<this.size*.5"));
assert(runtime.includes("const pHat=likelihoodMass/attempts,priorWeight=this.weights[i];"));
assert(runtime.includes("updated[i]=chosen?priorWeight*pHat:0"),
  "posterior must still use the same weight times likelihood");
assert(runtime.includes("this.rng()*likelihoodMass<mass"));
assert(physics.includes("function summarizeHypotheses(physics,_history,{samplesPerParticle=96}={}){"));
assert(panel.includes("lossGate:.035,simulationsPerParticle:96,"));
const engine=new PhysicsForecastFirstEngine();
assert.equal(engine.simulationsPerParticle,96);
assert.equal(engine.quant.physics.size,96);
assert.equal(engine.quant.lambda,.82);assert.equal(engine.quant.alpha,.15);
assert(physics.includes('const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";'));
assert(physics.includes("physics_weight:5"));
assert(physics.includes("road_weight:.1"));
const report=engine.syncHistory(["B"]);
assert.equal(report.physical_ev.samples,9216);
assert.equal(report.physical_ev.particle_support,96);
assert.equal(report.research_signal.source,"simulated_bpt_only");
assert.equal(report.observed_card_evidence,false);
assert.equal(report.action,"SKIP","B/P/T cannot establish a betting edge");
assert(Number.isFinite(report.physical_ev.mc_se_gap));
assert(Number.isFinite(report.physical_ev.posterior_sd_gap));
assert.deepEqual(report,engine.predict(),"repeat predict must not mutate the model");
const seq=["B","T","P","B","P","B","T","P","B"];
const a=new PhysicsForecastFirstEngine().syncHistory(seq);
const b=new PhysicsForecastFirstEngine().syncHistory(seq);
assert.deepEqual(a,b,"deterministic same-history replay");
assert.equal(a.physical_ev.samples,9216);
assert.equal(new PhysicsForecastFirstEngine().predict().physical_ev.samples,0,
  "empty history continues to show baseline");
assert.equal(summarizeHypotheses(engine.quant.physics,[]).samples,9216,
  "direct physical estimator default must match model default");

assert(auth.includes("quant256_online_runtime.js?v=20261104-pf96-l256-m96-v1"));
assert(auth.includes("physics_forecast_first_runtime.js?v=20261104-pf96-l256-m96-v1"));
assert(auth.includes("shoe_distribution_guard.js?v=20261103-dynamic-shoe-guard-v1"));
assert(auth.includes("quant256_online_panel.js?v=20261104-pf96-l256-m96-v1"));
assert(page.includes("auth-client.js?v=20261104-pf96-l256-m96-v1"));
assert(panel.includes('const KEY="bgs256d_frozen_6x15_forward_v18"'),
  "existing local-history storage key must stay stable");
console.log("PASS V2 96 particles x 256 likelihood proposals x 96 EV samples; original gates, replay, browser loader");
