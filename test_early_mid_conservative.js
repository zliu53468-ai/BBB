'use strict';
const assert=require('assert/strict'),P=require('./physics_priority_runtime'),c={early_mid_conservative_v2:{...P.CONSERVATIVE_DEFAULTS,enabled:true}};
for(let n=1;n<=30;n++){const v=P.limits(n,c);assert(v.pf_max<=.08&&v.alpha_max<=.4&&v.residual_max_delta<=.015);assert.equal(v.calibration_gain,0);assert.equal(v.point_weight_max,0);assert(v.probability_band>=.0275);}
for(const key of ['probability_band','noise_max','uncertainty_max','ess_min','quality_min','alpha','alpha_max','pf_max','residual_max_delta','calibration_gain']){
 const a=P.limits(30,c)[key],b=P.limits(31,c)[key],d=P.limits(45,c)[key],e=P.limits(46,c)[key],maxStep=key==='calibration_gain'?.100001:.05;assert(Math.abs(a-b)<maxStep,key+'30/31 transition');assert(Math.abs(d-e)<maxStep,key+'45/46 transition');
}
const input=n=>({history:Array(n-1).fill('P'),probabilities:{B:.54,P:.36,T:.10},noise:.56,uncertainty:.56,ess_ratio:.72,particle_quality:.68});
assert.equal(P.infer(input(25),[],c).direction,'Skip');assert(P.infer(input(25),[],c).reasons.includes('noise_high'));
assert(P.infer(input(25),[],c).alpha<=.40);
const records=Array.from({length:29},(_,i)=>({round:i+1,direction:'Skip',actual:'P',raw_q:.6,smoothed_q:.6}));assert.equal(P.infer(input(40),records,c).calibration.delta,0);
assert(P.infer(input(30),[...records,{round:30,actual:'P',raw_q:.6,smoothed_q:.6}],c).calibration.delta===0);
const old=P.infer(input(40));assert.deepEqual(P.infer(input(40),[],{early_mid_conservative_v2:{enabled:false}}),old);
assert.equal(P.infer(input(40),[],c).route_weight,0);assert.equal(P.infer(input(40),[],c).xgb_influence,0);
console.log('Early/middle: strict caps, no early calibration/points, smooth boundaries, high-noise Skip, rollback passed');
