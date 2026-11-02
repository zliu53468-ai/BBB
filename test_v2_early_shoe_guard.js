"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {BlindPhysicsEstimator,OnlineLinUCB256}=
  require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine,earlyShoeGuard,summarizeHypotheses,
  BASE,BASE_EV,BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION,VERSION}=
  require("./physics_forecast_first_runtime.js");
const near=(a,b,eps=1e-10)=>assert(Math.abs(a-b)<eps,
  `expected ${a} near ${b}`);
assert.equal(VERSION,"BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE");
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
const blind=new BlindPhysicsEstimator();
assert.equal(blind.size,256);
const model=new OnlineLinUCB256();
assert.equal(model.particleCount,256);
assert(fs.readFileSync("quant256_online_runtime.js","utf8")
  .includes("const attempts=384,next=new Array(this.size),updated=new Array(this.size);"));
assert.equal(new PhysicsForecastFirstEngine().simulationsPerParticle,160);

// A controlled test mixture proves Bayesian-style shrinkage moves *toward*
// the exact fresh-shoe baseline and scales down unverified early EV noise.
const fake={particles:Array.from({length:96},()=>[128,...Array(9).fill(32)]),
  size:96,ancestorIds:Array.from({length:96},(_,i)=>i),
  lastLikelihoodRelativeSE:.15};
const raw={
  simulated_probability:{B:.52,P:.40,T:.08},
  ev_banker:.95*.52-.40,ev_player:.40-.52,
  ev_gap:(.95*.52-.40)-(.40-.52),
  ev_gap_deviation:0.13,
  mc_se_gap:.02,particle_ess:96,posterior_update_ess:96,
  particle_support:96,simulation_exhausted:false
};
const a=earlyShoeGuard(raw,fake,1),b=earlyShoeGuard(raw,fake,10);
assert(a.diagnostics.active&&a.diagnostics.prior_weight>0);
assert(a.diagnostics.blend>0&&a.diagnostics.blend<b.diagnostics.blend);
assert(b.diagnostics.blend<1);
assert(a.gateAddition>0&&a.gateAddition>b.gateAddition);
assert(Math.abs(a.physical.simulated_probability.B-BASE.B)<
  Math.abs(raw.simulated_probability.B-BASE.B));
near(a.physical.simulated_probability.B+
  a.physical.simulated_probability.P+a.physical.simulated_probability.T,1);
near(a.physical.ev_banker,.95*a.physical.simulated_probability.B-
  a.physical.simulated_probability.P);
near(a.physical.ev_player,a.physical.simulated_probability.P-
  a.physical.simulated_probability.B);
assert.strictEqual(earlyShoeGuard(raw,fake,20).physical,raw,
  "round 20 must return original V2 EV object");
assert.strictEqual(earlyShoeGuard(raw,fake,21).physical,raw,
  "mature shoe cannot be shrunk");
assert.strictEqual(earlyShoeGuard(raw,fake,1,{observed:true}).physical,raw,
  "real observed ranks must bypass early virtual guard");
assert.strictEqual(earlyShoeGuard({...raw,simulation_exhausted:true},fake,5)
  .diagnostics,null,"unsupported physical evidence cannot produce a signal");

const seq="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const run=()=>{
 const engine=new PhysicsForecastFirstEngine(),rows=[];
 for(let i=0;i<seq.length;i++){
   const report=engine.syncHistory(seq.slice(0,i+1)),round=i+1;
   assert.equal(engine.quant.physics.size,256);
   assert.equal(report.physical_ev.samples,40960);
   assert(["SKIP","SIGNAL"].includes(report.research_signal.action));
   assert.equal(report.action,"SKIP","B/P/T-only remains non-bettable");
   assert.equal(report.feature_weights.physics,5);
   assert.equal(report.feature_weights.road,.1);
   assert([0,.005,.035].includes(report.dynamic_gate));
   assert.equal(engine.quant.physics.weights.length,256);
   near(engine.quant.physics.weights.reduce((a,b)=>a+b,0),1);
   const snap=engine.quant.physics.snapshot();
   assert(Number.isFinite(snap.likelihood_relative_se));
   assert(snap.particle_ancestor_diversity>0&&snap.particle_ancestor_diversity<=1);
   if(round<20){
     assert(report.early_shoe_guard?.active);
     assert(report.early_shoe_guard.blend<1);
     assert(report.early_shoe_guard.supplemental_gate>=0);
     assert.equal(report.research_signal.early_shoe_guard.active,true);
   }else{
     assert.equal(report.early_shoe_guard,null);
     const raw=summarizeHypotheses(engine.quant.physics,[],
       {samplesPerParticle:160});
     near(report.ev_banker,raw.ev_banker);
     near(report.ev_player,raw.ev_player);
     near(report.ev_gap_deviation,raw.ev_gap_deviation);
     assert.deepEqual(report.physical_ev,raw,
       "no latent EV shrink or extra uncertainty at/after round 20");
   }
   assert.deepEqual(report,engine.predict(),"read-only result must remain stable");
   rows.push(report);
 }
 return rows;
};
const outputs=run();
assert.deepEqual(outputs,run(),"fixed seed must preserve full history replay");
const auth=fs.readFileSync("auth-client.js","utf8");
const page=fs.readFileSync("index.html","utf8");
assert(auth.includes("quant256_online_runtime.js?v=20261103-pf256-l384-m160-v1"),
  "unchanged particle runtime must keep its cache version");
assert(auth.includes("shoe_distribution_guard.js?v=20261103-dynamic-shoe-guard-v1"),
  "new distribution guard must load before the physics module");
assert(auth.includes("physics_forecast_first_runtime.js?v=20261103-pf256-l384-m160-v1"),
  "upgraded physics module must get new cache version");
assert(page.includes("auth-client.js?v=20261103-pf256-l384-m160-v1"));
assert(page.includes("viewport-fit.js?v=20261102-mobile-stable-v2"),
  "mobile layout fix must be loaded without changing prediction logic");
assert(!page.includes("side-bet-card"),"removed side-bet panel must stay removed");
console.log("PASS Early-Shoe V2 B/P/T guard (1-19), exact mature physical EV after 20, observed-rank bypass, deterministic replay, 256/384/160 and loader");
