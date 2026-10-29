"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const current=JSON.parse(fs.readFileSync("cardcount_profile.json")),baseline=JSON.parse(fs.readFileSync("validation/cardcount_moderate_v1_baseline.json"));
// This test isolates the inherited smoothing profile; selective entry is tested separately.
current.hitrate_max_v1.enabled=false;
if(current.pf_weight_smoothing_v1)current.pf_weight_smoothing_v1.enabled=false; // inherited profile regression; new tuning has its own tests
const M=require("./browser_calibration.js");
function make(profile){
  const store=new Map(),ctx={console,localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},document:{getElementById:()=>null},fetch:async file=>({ok:true,json:async()=>file==="cardcount_profile.json"?profile:JSON.parse(fs.readFileSync(file,"utf8"))})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const file of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(file,"utf8"),ctx,{filename:file});
  return {api:ctx.__BGS_FINAL56__};
}
function memory(){const map=new Map();return {getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k)};}
(async()=>{
  const a=make(current),b=make(baseline);await Promise.all([a.api.loadModels(),b.api.loadModels()]);
  let comparisons=0,maxEarlyDelta=0;
  for(let r=1;r<=20;r++){
    for(const key of Object.keys(baseline.anchors))assert(Math.abs(a.api.profileValue(key,r)-b.api.profileValue(key,r))<1e-14,"early interpolation changed: "+key);
    for(const noise of [.1,.55,.8,.95])for(const uncertainty of [.1,.7,.95])assert(Math.abs(a.api.dynamicEmaAlpha(r,noise,{},null,{physics_uncertainty:uncertainty})-b.api.dynamicEmaAlpha(r,noise,{},null,{physics_uncertainty:uncertainty}))<1e-14);
    assert.equal(M.atRound(M.settings(current.local_calibration),r).prior_strength,100);
    assert.equal(M.atRound(M.settings(current.local_calibration),r).residual_decay,1);
    assert.equal(M.atRound(M.settings(current.local_calibration),r).validation_window,32);
  }
  for(const pattern of ["BBBBPP","BP","BPTBP"]){
    for(let r=1;r<=20;r++){
      const h=Array.from({length:r-1},(_,i)=>pattern[i%pattern.length]);
      const x=a.api.applyFinalPrediction(h),y=b.api.applyFinalPrediction(h);
      assert.equal(x.direction,y.direction,"early formal direction changed");
      const delta=Math.abs(x.next_hand_forecast.directional.B-y.next_hand_forecast.directional.B);maxEarlyDelta=Math.max(maxEarlyDelta,delta);
      assert(delta<1e-12,"early displayed probability changed");
      assert.equal(x.finalProbability.particleDiagnostics.forecast_draws,3);
      assert(Math.abs(x.finalProbability.finalPB-y.finalProbability.finalPB)<1e-12);
      assert.equal(JSON.stringify(x.finalProbability.physicsDirect),JSON.stringify(y.finalProbability.physicsDirect));comparisons++;
    }
  }
  for(const r of [24,45,50,60,70]){
    assert(a.api.dynamicEmaAlpha(r,.4,{},null,{physics_uncertainty:.3})>b.api.dynamicEmaAlpha(r,.4,{},null,{physics_uncertainty:.3}));
    const h=Array.from({length:r-1},(_,i)=>"BBPPT"[i%5]),p=a.api.applyFinalPrediction(h),d=p.finalProbability.particleDiagnostics;
    assert.equal(d.forecast_draws,r<48?4:5);assert(d.blend_weight<=.16);assert.equal(d.particle_count,10000);assert.equal(d.persistent_state,false);
    assert.equal(p.finalProbability.physics.length,48);assert.equal(p.finalProbability.extended.length,57);
    assert(p.direction==="Skip"||p.direction===p.finalProbability.primary_candidate);
    if(p.direction!=="Skip")assert((p.direction==="B"?p.ev_banker:p.ev_player)>0);
  }
  const client=M.createClient(memory()),h=[],f={available:true,direction:"B",directional:{B:.55,P:.45},probabilities:{B:.495,P:.405,T:.1},uncertainty:.2};
  for(let i=0;i<70;i++){
    const t=client.prepare(h,f,"s","m");h.push(i<40?"B":"P");client.commit(t,h.at(-1),h,current.local_calibration);
    const out=client.apply(f,h,"s","m",current.local_calibration);
    assert(Math.abs(out.status.delta)<=.005+1e-12);assert(out.forecast.directional.B>.5);
    if(out.status.applied)assert(out.status.ready&&out.status.validation_count>=16&&out.status.brier_gain>=.00005);
  }
  console.log(JSON.stringify({passed:true,early_model_comparisons:comparisons,max_early_probability_delta:maxEarlyDelta,checks:["early 1-20 locked","noise/uncertainty gates","middle/late faster EMA","PF 4/5 forecasts bounded","48D/57D","recent prequential calibration remains gated"]}));
})().catch(e=>{console.error(e);process.exitCode=1;});
