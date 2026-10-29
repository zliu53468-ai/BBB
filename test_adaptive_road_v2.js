'use strict';
const assert=require('assert/strict'),R=require('./road_primary_runtime'),{make}=require('./test_pf_primary_runtime');
const c={...require('./cardcount_profile.json').road_primary_v1,order:3,five_level_v1:{enabled:false}};
assert.equal(c.order,3);
const a=R.estimate(Array(30).fill('B'),c);assert.equal(a.order,3);assert.equal(a.backoff.length,1);
let seed=12345,seen=new Set();
for(let k=0;k<300;k++){const h=Array.from({length:70},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return ['B','P','T'][seed%3];});const r=R.estimate(h,c);if(r.available){seen.add(r.order);assert.equal(r.backoff.at(-1).order,r.order);assert(r.backoff.slice(0,-1).every(x=>!x.eligible));}}
assert(seen.has(1));
const two=R.estimate(Array.from({length:50},(_,i)=>i%2?'P':'B'),{...c,min_observations:24});assert.equal(two.order,2);assert.equal(two.backoff[0].eligible,false);
assert.equal(R.estimate(['B','P'],c).available,false);assert.deepEqual(R.estimate(Array(30).fill('B'),c),a);
(async()=>{const {api}=await make({road:true});assert.equal(api.profileValue('decision_alpha',1),.38);assert.equal(api.profileValue('decision_alpha',70),.56);assert(Math.abs(api.profileValue('decision_alpha',25)-.41)<1e-12);
const f=q=>({available:true,directional:{B:q,P:1-q},direction:q>.5?'B':'P'});
assert.equal(api.displayDirection(f(.5024)),'Skip');assert.equal(api.displayDirection(f(.5025)),'B');assert.equal(api.displayDirection(f(.4975)),'P');
const off=await make();assert.equal(off.api.roadPrimaryConfig(),null);assert.equal(off.api.profileValue('decision_alpha',1),.42);
console.log('PASS adaptive 3/2/1, causal rebuild, EMA anchors, 0.5pp and rollback');})().catch(e=>{console.error(e);process.exitCode=1;});
