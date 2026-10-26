"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const profile=JSON.parse(fs.readFileSync('cardcount_profile.json','utf8'));
async function make(enabled){
 const config=structuredClone(profile);config.adaptive_smoothing.enabled=enabled;
 const store=new Map([['bgs_xgb_final_shoe_id_v5','smooth-test']]),ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async f=>({ok:true,json:async()=>f==='cardcount_profile.json'?config:JSON.parse(fs.readFileSync(f,'utf8'))})};ctx.window=ctx;vm.createContext(ctx);
 for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(f,'utf8'),ctx,{filename:f});
 await ctx.__BGS_FINAL56__.loadModels();return {api:ctx.__BGS_FINAL56__,store,config};
}
(async()=>{
 const a=await make(true),b=await make(false),alpha=a.api.adaptiveSmoothingAlpha;
 for(const round of [1,10,20,21,35,46,70])for(const p of [.46,.49,.5,.51,.54])for(const noise of [.1,.55,.7,.95])for(const u of [.2,.6,.75,.95]){
  const base=a.api.dynamicEmaAlpha(round,noise,{},null,{physics_uncertainty:u});
  const actual=alpha(base,p,.5,round,noise,u);assert(actual>=base&&actual<=.84);
  if(round<=20||noise>=.7||u>=.75||Math.abs(p-.5)<=.015)assert.equal(actual,base);
  const value=actual*p+(1-actual)*.5;assert(value>=Math.min(p,.5)&&value<=Math.max(p,.5));
 }
 const slow=.72*.54+.28*.46,fast=alpha(.72,.54,.46,60,.2,.2)*.54+(1-alpha(.72,.54,.46,60,.2,.2))*.46;
 assert(fast>slow&&fast<=.54,'clear reversal must reduce lag without extrapolation');
 for(const val of [null,NaN,Infinity])assert.equal(alpha(.6,.54,.46,46,val,.2),.6);
 const c=a.config.adaptive_smoothing;c.max_alpha_boost=1;assert.equal(alpha(.6,.54,.46,46,.2,.2),.6);c.max_alpha_boost=.10;
 let cases=0;
 for(const n of [1,19,20,35,45,60])for(const prior of [.46,.54]){
  for(const runtime of [a,b])runtime.store.set('bgs_xgb_final_training_v5',JSON.stringify([{shoe_id:'smooth-test',round_index:n,physics_smoothed_p_b:prior,predicted_direction:'Skip',final_p_b:prior,p_tie:.1}]));
  const h=Array.from({length:n},(_,i)=>'BBPPT'[i%5]),x=a.api.applyFinalPrediction(h),y=b.api.applyFinalPrediction(h),r=x.finalProbability,t=y.finalProbability;
  assert.equal(JSON.stringify(r.physics),JSON.stringify(t.physics));assert.equal(JSON.stringify(r.extended),JSON.stringify(t.extended));assert.equal(r.clippedPB,t.clippedPB);
  assert(r.smoothingAlpha>=t.smoothingAlpha);assert(Math.abs(r.finalPB-.5)<=Math.max(Math.abs(r.clippedPB-.5),Math.abs(prior-.5))+1e-12);
  if(n+1<=20)assert.equal(r.finalPB,t.finalPB);
  assert.equal(x.forecast_direction,x.ev_direction.direction);if(x.direction!=='Skip'){assert.equal(x.direction,x.ev_direction.direction);assert(x.ev_direction.best_ev>0);}cases++;
 }
 a.store.set('bgs_xgb_final_training_v5',JSON.stringify([{shoe_id:'smooth-test',round_index:2,physics_smoothed_p_b:null}]));
 assert.equal(a.api.applyDynamicSmoothing(.54,3,.2,{low:.4,high:.6}).value,.54,'null history must not become zero');
 a.store.set('bgs_xgb_final_training_v5',JSON.stringify([{shoe_id:'other-shoe',round_index:2,physics_smoothed_p_b:.1}]));
 assert.equal(a.api.applyDynamicSmoothing(.54,3,.2,{low:.4,high:.6}).alpha,1,'shoe reset');
 console.log(JSON.stringify({passed:true,runtime_cases:cases,early_unchanged:true,high_noise_unchanged:true,clear_signal_lag_reduced:true,feature_vectors_unchanged:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
