'use strict';
const assert=require('node:assert/strict');
const {PhysicsForecastFirstEngine,deal}=require('./physics_forecast_first_runtime.js');
const quant=require('./quant256_online_runtime.js'),guard=require('./point_signal_quality_guard.js');
const deplete=require('./deplete_mc_runtime.js');
const ledger=require('./validation/deplete_validation.json');
assert.equal(deplete.CONFIG.defaultMode,'shadow');
let counts={B:0,P:0,SKIP:0};
for(let player=0;player<10;player++)for(let banker=0;banker<10;banker++){
 const engine=new PhysicsForecastFirstEngine({independentFinalScoreMode:true,exactSingleScore:true});
 const report=engine.syncHistory([banker>player?'B':player>banker?'P':'T'],[{player,banker}]);
 const before=JSON.stringify(report),shadow=deplete.attachShadow(report,{player,banker});
 assert.equal(shadow.deplete_mc.status,'complete');
 const {deplete_mc,...originalFields}=shadow;
 assert.deepEqual(originalFields,report,'shadow must preserve EVERY original report field');
 assert.equal(JSON.stringify(report),before,'auxiliary computation cannot mutate original objects');
 assert.deepEqual(guard.decide(shadow),guard.decide(report));
 assert.equal(deplete_mc.affects_display,false);assert.equal(deplete_mc.observed_card_evidence,false);
 assert.equal(deplete_mc.completed_shoes,256);assert.equal(deplete_mc.samples,32768);
 assert.equal(deplete_mc.independent_observation,false);
 assert(Math.abs(Object.values(deplete_mc.probability).reduce((a,b)=>a+b)-1)<1e-10);
 counts[guard.decide(shadow).direction]++;
}
assert.deepEqual(counts,{B:36,P:24,SKIP:40});
const once=deplete.estimate(5,3);deplete.estimate(2,8);assert.deepEqual(deplete.estimate(5,3),once);
const engine=new PhysicsForecastFirstEngine({independentFinalScoreMode:true,exactSingleScore:true});
const base=engine.syncHistory(['P'],[{player:5,banker:3}]);
const saved=quant.conditionalHandProposal;
try{
 quant.conditionalHandProposal=()=>({weight:0,hand:null});
 const failed=deplete.attachShadow(base,{player:5,banker:3});
 assert.equal(failed.deplete_mc.status,'unavailable');
 assert.equal(failed.ev_banker,base.ev_banker);assert.equal(failed.ev_player,base.ev_player);
 assert.deepEqual(guard.decide(failed),guard.decide(base));
}finally{quant.conditionalHandProposal=saved;}
// Compare supplementary draw rules to the existing tested dealer on paired uniforms.
let state=0x9b9773e9;const rng=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
for(let i=0;i<10000;i++){
 const counts=[124,32,32,32,31,32,32,31,32,32],u=Array.from({length:6},rng);let k=0;
 assert.equal(deplete.dealWithUniforms(counts,u),deal(counts,()=>u[k++]).outcome);
}
assert.equal(ledger.scenarios,1000000);assert.equal(ledger.seed,20261105);
assert(ledger.max_conditional_bin_standardized_error<6);
assert.equal(ledger.promotion_eligible,false);
console.log('PASS independent deplete shadow: all 100 exact primary reports unchanged, 10,000 dealer comparisons, deterministic reset, bounded-failure fallback and million-case evidence');
