"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const read=n=>fs.readFileSync(n,"utf8");
async function runtime(mode){
  const data=new Map();const ctx={console,BBB_BLIND_UPGRADE_MODE:mode,
    document:{getElementById:()=>null},localStorage:{getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,String(v)),removeItem:k=>data.delete(k)},
    fetch:async n=>({ok:true,json:async()=>JSON.parse(read(n))})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const n of ["app256forward.js","app256continuation.js","blind_upgrade_runtime.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"])
    vm.runInContext(read(n),ctx,{filename:n});
  await ctx.__BGS_FINAL56__.loadModels();return ctx;
}
(async()=>{
  const ctx=await runtime("guarded"),u=ctx.__BBB_BLIND_UPGRADE__,api=ctx.__BGS_FINAL56__;
  assert.equal(u.structure("BBBBBBBBBB".split("")).entropy,0);
  assert.equal(u.structure("BPBPBPBPBP".split("")).entropy,0,"regular alternation is predictable, not maximum chaos");
  assert(u.structure("BBPPBPBPPB".split("")).entropy>.7);
  assert.equal(u.structure("TTTTTTTTTT".split("")).supported,false);
  assert.equal(u.structure("BBPPTTTTTT".split("")).entropy,null);
  const d={particle_count:500,probability_std:.04,probability_mc_se:.003,pre_resample_ess_ratio:.8,ancestor_ratio:.7,conditional_failures:0,forecast_unsupported_particles:0};
  assert(u.thresholds(15,.05,.02,d).ev<.05);
  assert.equal(u.thresholds(16,.05,.02,d).ev,.05);
  assert.equal(u.thresholds(50,.05,.02,d).ev,.05);
  assert(u.thresholds(51,.05,.02,null).ev>.05);
  assert(Math.abs(u.thresholds(51,.05,.02,d).ev-.05)<1e-12);
  for(const bad of [{},{...d,probability_std:NaN},{...d,ancestor_ratio:.01},{...d,pre_resample_ess_ratio:.1},{...d,forecast_unsupported_particles:1},{...d,probability_std:-1}])
    assert.equal(u.consensus(bad),false);
  assert(u.thresholds(5,0,0,d).ev>0,"no zero-EV bypass");
  const cases=["B","T","BBBBBBBBBBBB","BPBPBPBPBPBP","BPPBTBBPPTBBPPBTBP", "BPT".repeat(18),"BPPBBPBPPBBPBT".repeat(5)];
  const results=[];
  for(const h of cases){
    const seq=h.split(""),start=Date.now(),p=api.applyFinalPrediction(seq),r=p.finalProbability,pd=r.particleDiagnostics;
    assert.equal(r.error,"");assert.equal(pd.conditional_mode,"rules_conditioned");
    assert.equal(pd.particle_count,500);assert.equal(pd.conditional_failures,0);
    assert.equal(pd.forecast_draws,64);assert(Number.isFinite(pd.probability_std));
    assert(pd.posterior_expected_consumed_cards>=4*seq.length-1e-8&&pd.posterior_expected_consumed_cards<=6*seq.length+1e-8);
    assert.equal(pd.history_fingerprint,h);assert.equal(pd.persistent_state,false);
    for(const [i,n] of [[0,3],[3,10],[13,10],[23,3]])assert(Math.abs(r.physics.slice(i,i+n).reduce((a,b)=>a+b,0)-1)<1e-8);
    assert.equal(r.blind_upgrade.threshold.stage,seq.length<=15?"early":seq.length>50?"late":"middle");
    assert(r.primary_candidate===p.direction||p.direction==="Skip");
    const again=api.applyFinalPrediction(seq);assert.equal(again.finalProbability.finalPB,r.finalPB,"deterministic replay");
    assert.deepEqual(again.finalProbability.particleDiagnostics,pd);
    api.registerPrediction(seq,p);api.settlePending("B",{outcome:"B"},seq);
    assert.equal(JSON.stringify(api.getTrainingRows().at(-1).blind_upgrade),JSON.stringify(r.blind_upgrade));
    results.push({rounds:h.length,direction:p.direction,ms:Date.now()-start,sd:pd.probability_std,ess:pd.pre_resample_ess_ratio,ancestors:pd.ancestor_ratio});
  }
  const base=await runtime("baseline");
  assert.equal(base.__BBB_BLIND_UPGRADE__.enabled,false);
  assert.equal(base.__BBB_BLIND_UPGRADE__.thresholds(5,.05,.02,d).ev,.05);
  // Legacy source with no adapter must exactly match the rollback profile.
  const old=await runtime("baseline");
  delete old.__BBB_BLIND_UPGRADE__;
  for(const file of ["particle_filter_runtime.js","final_probability_runtime.js"])vm.runInContext(read(file),old);
  await old.__BGS_FINAL56__.loadModels();
  const seq="BPPBTBBPPTBBPPBTBP".split("");
  const a=base.__BGS_FINAL56__.applyFinalPrediction(seq),b=old.__BGS_FINAL56__.applyFinalPrediction(seq);
  assert.equal(a.direction,b.direction);assert.equal(a.finalProbability.finalPB,b.finalProbability.finalPB);
  const requested=await runtime("requested_prior");
  const ex=requested.__BGS_FINAL56__.applyFinalPrediction(["T"]).finalProbability;
  assert.equal(ex.particleDiagnostics.conditional_mode,"requested_heuristic");
  const histogram=ex.particleDiagnostics.consumed_by_outcome.T,total=histogram.reduce((a,b)=>a+b,0);
  assert(Math.abs(histogram[0]/total-.1)<.08&&Math.abs(histogram[2]/total-.5)<.1);
  console.log(JSON.stringify({ok:true,checks:["depth boundaries","entropy vs alternation","low-SD is insufficient","finite legal depletion","no flip","normalized probabilities","deterministic stateless replay","legacy rollback parity","requested prior experimental"],results}));
})().catch(e=>{console.error(e);process.exitCode=1;});
