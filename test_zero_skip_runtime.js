#!/usr/bin/env node
"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm");
function setup(withParticle=true,modelOverride=null){
  const store=new Map();
  const box={console,localStorage:{
    getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    document:{getElementById:()=>null},
    fetch:async url=>({ok:true,json:async()=>url==="final_probability_model.json"&&modelOverride
      ?modelOverride:JSON.parse(fs.readFileSync(url,"utf8"))})};
  box.window=box;vm.createContext(box);
  for(const file of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"]){
    if(!withParticle&&file==="particle_filter_runtime.js")continue;
    vm.runInContext(fs.readFileSync(file,"utf8"),box,{filename:file});
  }
  return {api:box.__BGS_FINAL56__,store};
}
(async()=>{
  const {api,store}=setup();
  await api.loadModels();
  assert.equal(api.zeroSkipExperimentEnabled(),true);
  const cases=[1,2,3,5,8,11,16,24,37,42,51,62,70];
  let forced=0,normalSkipped=0;
  for(const len of cases){
    const seq=Array.from({length:len},(_,j)=>["B","P","T","P","B","B","T"][j%7]);
    // No history change during repeated predictions.
    const before=api.applyFinalPrediction(seq);
    const r=before.finalProbability;
    assert.equal(r.mode,"physics_primary");
    assert.equal(r.zero_skip_experiment,true);
    assert(["B","P"].includes(before.direction),"zero-Skip must give an outcome at round "+len);
    assert.equal(r.flipped_from_physics,false);
    assert.equal(before.direction,r.primary_candidate);
    assert.equal(r.particle_count,500);
    assert.equal(r.physics.length,48);
    assert.equal(r.extended.length,57);
    assert.equal(r.execution_order.join(","),"direct_physics,particle_filter,physical_ev,physics_candidate,physics_ema,frozen_core,xgb_aux_filter,final_ev_guard,volume_guard");
    if(r.zero_skip_forced){
      forced++;
      assert.equal(r.standard_direction,"Skip");
      assert.equal(before.entry_tier,"experimental_forced");
      assert.equal(before.stake_multiplier,0);
      assert.equal(before.confidence,0);
    }
    api.setZeroSkipExperiment(false);
    const normal=api.applyFinalPrediction(seq);
    if(normal.direction==="Skip")normalSkipped++;
    assert.equal(normal.finalProbability.zero_skip_experiment,false);
    assert.equal(normal.finalProbability.zero_skip_forced,false);
    assert.equal(normal.finalProbability.primary_candidate,normal.finalProbability.standard_primary_candidate);
    api.setZeroSkipExperiment(true);
    assert.equal(api.applyFinalPrediction(seq).direction,before.direction,"experimental prediction must be deterministic");
  }
  assert(forced>0,"must exercise guard override cases");
  assert(normalSkipped>0,"normal mode should still abstain on at least one sample");
  api.setZeroSkipExperiment(false);
  assert.equal(store.get("bgs_zero_skip_experiment_v1"),"0");
  assert.equal(api.zeroSkipExperimentEnabled(),false);
  api.setZeroSkipExperiment(true);
  assert.equal(api.zeroSkipExperimentEnabled(),true);
  const disabledModel=JSON.parse(fs.readFileSync("final_probability_model.json","utf8"));
  disabledModel.decision_policy.zero_skip_experiment.enabled=false;
  const disabled=setup(true,disabledModel);await disabled.api.loadModels();
  assert.equal(disabled.api.zeroSkipExperimentEnabled(),false,"JSON kill switch must work without localStorage override");
  const failed=setup(false);await failed.api.loadModels();
  const fallback=failed.api.applyFinalPrediction(["B","P","T"]);
  assert.equal(fallback.direction,"Skip","missing Particle500 must fail-safe to Skip, never use Core to fabricate an action");
  assert.equal(fallback.finalProbability.mode,"physics_unavailable");
  console.log(JSON.stringify({ok:true,cases:cases.length,forced,normal_skipped:normalSkipped,
    guarantees:["model_ready_zero_skip","physics_direction_no_flip","legacy_guard_preserved","deterministic","model_unavailable_safe_skip","toggle_reversible","no_model_weight_changes"]}));
})().catch(error=>{console.error(error);process.exit(1);});
