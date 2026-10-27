'use strict';
const fs=require('node:fs'),vm=require('node:vm'),cp=require('node:child_process'),assert=require('node:assert/strict');
const profile=require('./cardcount_profile.json');
async function make(enabled,baseline=false){
 const config=baseline?JSON.parse(cp.execFileSync('git',['show','bd14399:cardcount_profile.json'],{encoding:'utf8'})):structuredClone(profile);
 if(!baseline)config.winrate_priority_v1.enabled=enabled;
 const map=new Map([['bgs_xgb_final_shoe_id_v5','unified-test']]);
 const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k)},fetch:async f=>({ok:true,json:async()=>f==='cardcount_profile.json'?config:JSON.parse(fs.readFileSync(f,'utf8'))})};ctx.window=ctx;vm.createContext(ctx);
 for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(baseline&&f==='final_probability_runtime.js'?cp.execFileSync('git',['show','bd14399:'+f],{encoding:'utf8'}):fs.readFileSync(f,'utf8'),ctx,{filename:f});
 await ctx.__BGS_FINAL56__.loadModels();return {api:ctx.__BGS_FINAL56__,map};
}
const close=(a,b)=>assert(Math.abs(a-b)<1e-12,`${a} != ${b}`);
(async()=>{
 const on=await make(true),off=await make(false),old=await make(false,true);
 assert.equal(off.api.firstSmoothingConfig(),null);assert.equal(off.api.selectiveEntryPolicy(),null);
 assert(on.api.directionConsistency('B','P').forced_skip_by_consistency);
 assert(!on.api.directionConsistency('Balanced','P').forced_skip_by_consistency);
 for(const [r,b] of [[40,.005],[41,.0035],[50,.0035],[51,.0025]]){close(on.api.selectiveEvBoost(r,.55,.6).total_boost,b);close(on.api.selectiveEvBoost(r,.56,.6).total_boost,b+.004);}
 // Unified configuration uses the existing causal journal and replay machinery.
 const physics=gap=>{const p=Array(48).fill(0),q=(2+gap/.9)/3.95;p[0]=p[3]=p[13]=1;p[23]=.9*q;p[24]=.9*(1-q);p[25]=.1;p[42]=.2;return p;};
 const history=[],snapshots=[];
 for(let i=0;i<12;i++){
  const f=on.api.firstLayerEv(physics(i%2?-.025:.05),i+1,.9,.2,history);snapshots.push(f);
  const ticket=on.api.prepareFirstSmoothingTicket(history,{first_layer_ev:f});
  assert.equal(JSON.stringify(on.api.firstLayerEv(physics(i%2?-.025:.05),i+1,.9,.2,history)),JSON.stringify(f));
  history.push('BPT'[i%3]);assert(on.api.commitFirstSmoothingTicket(ticket,history.at(-1),history));
 }
 on.map.set('bgs256d_frozen_6x15_forward_v18',JSON.stringify({history:history.slice(0,-1)}));on.api.undoRuntimeHistory();
 assert.equal(JSON.stringify(on.api.firstLayerEv(physics(-.025),12,.9,.2,history.slice(0,-1)).smoothed),JSON.stringify(snapshots.at(-1).smoothed));
 on.map.set('bgs_xgb_final_shoe_id_v5','next-shoe');assert(on.api.firstLayerEv(physics(.03),1,.9,.2,[]).smoothed.initialized);
 const clean=on.api.firstSmoothingStep(.05,-.02,30,.2,.9,.2),noisy=on.api.firstSmoothingStep(.05,-.02,30,.99,.9,.2);close(noisy.alpha_effective,clean.alpha_base*.55);
 let compared=0;
 for(const t of require('./validation/selective_entry_v2_baseline.json').cases){
  for(const x of [on,off,old]){x.map.clear();x.map.set('bgs_xgb_final_shoe_id_v5','unified-test');if(t.previous!==null)x.map.set('bgs_xgb_final_training_v5',JSON.stringify(Array.from({length:8},(_,i)=>({shoe_id:'unified-test',round_index:Math.max(1,t.history.length-8+i),physics_smoothed_p_b:t.previous,final_p_b:.52,p_tie:.1,predicted_direction:'Skip',activation_ev:0}))));}
  const h=t.history.split(''),a=on.api.applyFinalPrediction(h),b=off.api.applyFinalPrediction(h),c=old.api.applyFinalPrediction(h);
  for(const k of ['direction','ev_banker','ev_player','confidence'])assert.equal(b[k],c[k],`PR65 rollback ${k}`);
  for(const k of ['finalPB','smoothedPB','effective_activation_ev','effective_confidence_band','volume_guard_active'])assert.equal(b.finalProbability[k],c.finalProbability[k],`PR65 rollback ${k}`);
  assert.equal(JSON.stringify(b.first_layer_ev),JSON.stringify(c.first_layer_ev));
  for(const k of ['finalPB','smoothedPB'])close(a.finalProbability[k],c.finalProbability[k]);
  close(a.ev_banker,c.ev_banker);close(a.ev_player,c.ev_player);
  assert(a.finalProbability.effective_activation_ev>=c.finalProbability.effective_activation_ev);
  assert.equal(a.first_layer_ev.direction_consistency.first_layer_smoothed,a.first_layer_ev.smoothed.direction);
  assert.deepEqual(a.settlement_audit.raw,a.first_layer_ev.raw);assert.deepEqual(a.settlement_audit.smoothed,a.first_layer_ev.smoothed);
  if(a.direction_consistency.forced_skip_by_consistency){assert.equal(a.direction,'Skip');assert.equal(a.finalProbability.volume_guard_active,false);}
  assert.equal(off.api.prepareCalibrationTicket(h,b).model,old.api.prepareCalibrationTicket(h,c).model);
  compared++;
 }
 assert(!off.map.has('bgs_first_layer_gap_smoothing_v1'));
 console.log(JSON.stringify({passed:true,pr65_rollback_cases:compared,probability_and_ev_unchanged:true,unified_switch_precedes_legacy:true}));
})().catch(e=>{console.error(e);process.exitCode=1});
