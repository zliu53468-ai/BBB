"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {BlindPhysicsEstimator,OnlineLinUCB256}=require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine,summarizeHypotheses,INITIAL,
  BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION,VERSION}=
  require("./physics_forecast_first_runtime.js");
const near=(a,b,eps=1e-10)=>assert(Math.abs(a-b)<eps,`${a} not equal ${b}`);
const source=fs.readFileSync("./physics_forecast_first_runtime.js","utf8");
const online=fs.readFileSync("./quant256_online_runtime.js","utf8");
assert.equal(VERSION,"BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE");
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
assert(source.includes('const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";'));
assert(source.includes("physics_weight:5"));
assert(source.includes("road_weight:.1"));
assert.equal(new BlindPhysicsEstimator().size,256);
assert.equal(new OnlineLinUCB256().particleCount,256);
assert.equal(new PhysicsForecastFirstEngine().simulationsPerParticle,160);
assert(online.includes("this.lastESS<this.size*.5"),"ESS trigger must be enabled");

// A fixed seed and the same result history must give identical posterior,
// physical EV, research gate, and ensemble diagnostics across replays.
const seq="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const get=()=>{
  const engine=new PhysicsForecastFirstEngine();
  const reports=[];
  for(let i=0;i<seq.length;i++){
    const report=engine.syncHistory(seq.slice(0,i+1));
    assert.equal(engine.quant.physics.size,256);
    assert.equal(report.physical_ev.samples,256*160);
    assert.equal(report.feature_weights.physics,5);
    assert.equal(report.feature_weights.road,.1);
    assert(["SKIP","SIGNAL"].includes(report.research_signal.action));
    const weights=engine.quant.physics.weights;
    near(weights.reduce((a,b)=>a+b,0),1);
    const counts=engine.quant.physics.particles;
    for(const p of counts){
      p.forEach((count,k)=>assert(Number.isInteger(count)&&count>=0&&count<=INITIAL[k]));
      const remaining=p.reduce((a,b)=>a+b,0);
      assert(remaining>=416-6*(i+1)&&remaining<=416-4*(i+1));
    }
    assert(report.physical_ev.particle_ess>0);
    assert(report.physical_ev.particle_ess<=256+1e-9);
    assert(Number.isFinite(report.physical_ev.mc_se_gap));
    assert(Number.isFinite(report.physical_ev.posterior_sd_gap));
    assert(report.physical_ev.se_gap>=report.physical_ev.mc_se_gap);
    reports.push({
      evB:report.ev_banker,evP:report.ev_player,
      direction:report.research_signal.direction,
      signal:report.research_signal.action,
      ess:report.physical_ev.particle_ess,
      weights:weights.slice(),
      counts:counts.map(a=>a.slice())
    });
  }
  return reports;
};
assert.deepEqual(get(),get(),"deterministic complete V2 replay must be stable");

// Weighted MC must genuinely differ from an equally weighted mixture when
// latent compositions differ; it must not just report 'weights' decoratively.
const mixed=new BlindPhysicsEstimator({particles:8});
mixed.particles=Array.from({length:8},(_,i)=>i<4?INITIAL.slice():
  [0,0,0,0,0,0,0,0,16,0]);
mixed.weights=[.20,.20,.20,.20,.05,.05,.05,.05];
const weighted=summarizeHypotheses(mixed,["B"],{samplesPerParticle:64});
assert(weighted.particle_ess<8);
mixed.weights=Array(8).fill(.125);
const equal=summarizeHypotheses(mixed,["B"],{samplesPerParticle:64});
assert(weighted.simulated_probability.B>equal.simulated_probability.B+.05);
assert(weighted.simulated_probability.T<equal.simulated_probability.T-.05);

// Force an ESS collapse and verify legal systematic resampling, nonnegative
// card counts, fresh independent array references, and uniform new weights.
const depleted=new BlindPhysicsEstimator({particles:8,seed:777});
depleted.weights=[.99,...Array(7).fill(.01/7)];
depleted.observe("B");
assert(depleted.lastResampled);
assert(depleted.lastESS<4);
near(depleted.weights.reduce((a,b)=>a+b,0),1);
depleted.weights.forEach(w=>near(w,1/8));
assert.equal(new Set(depleted.particles).size,8,"resampled arrays must not alias");
depleted.particles.forEach(p=>assert(p.reduce((a,b)=>a+b,0)>=410));

const noSupport=new BlindPhysicsEstimator({particles:8});
noSupport.particles=noSupport.particles.map(()=>Array(10).fill(0));
noSupport.observe("T");
assert(noSupport.posteriorUnavailable);
assert.equal(summarizeHypotheses(noSupport,[],{}).simulation_exhausted,true);

console.log("PASS 256-particle likelihood weighting, ESS resampling, weighted physical EV, uncertainty, V2 invariant gates, deterministic replay");
