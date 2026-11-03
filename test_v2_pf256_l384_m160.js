"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const {BlindPhysicsEstimator,OnlineLinUCB256}=require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine,summarizeHypotheses,
 BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION}=require("./physics_forecast_first_runtime.js");
const N=256,L=384,M=160;
const q=fs.readFileSync("./quant256_online_runtime.js","utf8");
const p=fs.readFileSync("./physics_forecast_first_runtime.js","utf8");
const panel=fs.readFileSync("./quant256_online_panel.js","utf8");
const auth=fs.readFileSync("./auth-client.js","utf8");
const page=fs.readFileSync("./index.html","utf8");
assert.equal(new BlindPhysicsEstimator().size,N);
assert.equal(new BlindPhysicsEstimator({particles:512}).size,N,
  "hard cap must actually be 256");
assert.equal(new BlindPhysicsEstimator({particles:96}).size,96,
  "lower explicit particle counts remain supported");
assert.equal(new OnlineLinUCB256().particleCount,N);
assert.equal(new OnlineLinUCB256().physics.size,N);
assert.equal(new PhysicsForecastFirstEngine().simulationsPerParticle,M);
assert(q.includes("const attempts=384,next=new Array(this.size),updated=new Array(this.size);"),
  "384 legal hand proposals must actually be performed per particle");
assert(q.includes("this.lastESS<this.size*.5"),"ESS cutoff must remain 0.5N");
assert(q.includes("resampleConditionalTransition(priorParticles[index],this.rng,outcome)"),"particle rejuvenation must use the exact prior deck and observed outcome");
assert(q.includes("const pHat=likelihoodMass/attempts,priorWeight=this.weights[i];"),
  "weighted Rao-Blackwell likelihood formula unchanged in particle count");
assert(q.includes("updated[i]=chosen?priorWeight*pHat:0"),
  "particle posterior weighting formula unchanged");
assert(q.includes("this.rng()*likelihoodMass<mass"),
  "conditional weighted reservoir sampling required");
assert(p.includes("function summarizeHypotheses(physics,_history,{samplesPerParticle=160}={}){"));
assert(panel.includes("lossGate:.035,simulationsPerParticle:160,"));
assert(panel.includes("quantOptions:{lambda:.82,alpha:.15,particles:256}"));
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
assert(p.includes('const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";'),
 "physical EV authority cannot change");
const engine=new PhysicsForecastFirstEngine();
assert.equal(engine.quant.lambda,.82);
assert.equal(engine.quant.alpha,.15);
const r=engine.syncHistory(["B"]);
assert.equal(r.physical_ev.samples,N*M,"exactly N*M physical EV deals");
assert.equal(r.physical_ev.particle_support,N);
assert.equal(engine.quant.physics.size,N);
assert.equal(engine.quant.physics.weights.length,N);
assert.equal(engine.quant.physics.ancestorIds.length,N);
assert(Number.isFinite(engine.quant.physics.lastLikelihoodRelativeSE));
assert(r.physical_ev.particle_ess<=N+1e-9);
assert.equal(r.action,"SKIP","B/P/T-only is not observed-card positive EV");
assert.equal(r.early_shoe_guard.active,true);
assert.equal(r.shoe_distribution_guard.current_stage,"insufficient_data");
assert.equal(r.feature_weights.physics,5);
assert.equal(r.feature_weights.road,.1);
assert.deepEqual(r,engine.predict());
assert.equal(summarizeHypotheses(engine.quant.physics,[]).samples,N*M);
const seq=["B","P","T","B","P","B","T"];
const left=new PhysicsForecastFirstEngine().syncHistory(seq);
const right=new PhysicsForecastFirstEngine().syncHistory(seq);
assert.deepEqual(left,right,"full V2 replay stays deterministic");
assert.equal(left.physical_ev.samples,N*M);
assert.equal(left.action,"SKIP");
assert.equal(left.research_signal.direction,
 left.ev_banker>=left.ev_player?"B":"P");
assert(auth.includes("quant256_online_runtime.js?v=20261103-v21-rb-move-v1"),"quant model remains unchanged");
assert(auth.includes("physics_forecast_first_runtime.js?v=20261103-v22-markov-shadow-v1"),"physics loader must pick up Markov research shadow");
assert(auth.includes("quant256_online_panel.js?v=20261103-v22-markov-shadow-v1"));
assert(page.includes("auth-client.js?v=20261103-v22-markov-shadow-v1"));
assert(auth.indexOf("shoe_distribution_guard.js?v=20261103-dynamic-shoe-guard-v1")<
 auth.indexOf("physics_forecast_first_runtime.js?v=20261103-v22-markov-shadow-v1"),
 "must load unchanged distribution module before the V2 engine");
assert(page.includes("viewport-fit.js?v=20261102-mobile-stable-v2"));
assert(!page.includes("side-bet-card"));
assert(panel.includes('const KEY="bgs256d_frozen_6x15_forward_v18"'),
 "prior member history key must stay intact");
console.log("PASS release PF256 L384 EV160: hard cap, true 98,304 likelihood + 40,960 EV samples, V2 gates and weights unchanged, deterministic replay, login, mobile and history");
