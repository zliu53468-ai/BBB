'use strict';
const assert=require('node:assert/strict'),P=require('./physics_priority_runtime');
const config=require('./cardcount_profile.json').physics_priority_v1;
const input=history=>({history,probabilities:{B:.56,P:.34,T:.10},noise:.30,uncertainty:.30,ess_ratio:.9,particle_quality:.9});
const balanced='BP'.repeat(25).split(''),oneSided=Array(50).fill('B');
const off={...config,ratio_imbalance_guard:{enabled:false}};
const base=P.infer(input(oneSided),[],off),guarded=P.infer(input(oneSided),[],config);
assert.equal(base.direction,'B');assert.equal(guarded.direction,'Skip');assert(guarded.reasons.includes('ratio_imbalance'));
assert.equal(guarded.smoothed_q,base.smoothed_q);assert.equal(guarded.ev_banker,base.ev_banker);
assert.equal(P.infer(input(balanced),[],config).probability_band,P.infer(input(balanced),[],off).probability_band);
const rising=[4,3,2,1,0].map(players=>P.imbalanceGate([...Array(16-players).fill('B'),...Array(players).fill('P')],config.ratio_imbalance_guard).band_boost);
assert(rising.every((v,i)=>!i||v>=rising[i-1]));
assert.equal(P.imbalanceGate(Array(16).fill('T'),config.ratio_imbalance_guard).forced_skip,false);
assert.equal(P.imbalanceGate(Array(11).fill('B'),config.ratio_imbalance_guard).forced_skip,false);
for(let n=1;n<=30;n++){const p=P.infer(input(Array(n-1).fill('B')),[],config);assert(p.probability_band>=.0275);assert(p.alpha<=.4);assert.equal(p.calibration.delta,0);}
assert.equal(P.infer(input(balanced),[],config).route_weight,0);
console.log('PASS: imbalance only tightens/never flips; ties/minimum samples; early caps; flag rollback.');
