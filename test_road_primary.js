'use strict';const assert=require('assert/strict'),R=require('./road_primary_runtime');
const v=q=>{const a=Array(48).fill(.1);a[23]=.9*q;a[24]=.9*(1-q);a[25]=.1;a[42]=.2;return a;};
const roadB=R.estimate(Array(30).fill('B')),roadP=R.estimate(Array(30).fill('P'));
assert(roadB.available&&roadP.available);assert.equal(roadB.direction,'B');assert.equal(roadP.direction,'P');
for(const [road,physical]of[[roadB,v(.2)],[roadP,v(.8)]]){const before=physical.slice(),r=R.combine(road,physical);assert.equal(Math.sign(r.physics[23]-r.physics[24]),road.direction==='B'?1:-1);assert(Math.abs(r.audit.delta)<=.00400000001);assert(r.audit.physics_weight_effective<=.15);assert(Math.abs(r.physics[25]-.1)<1e-12);assert(Math.abs(r.physics[39]-(.95*r.physics[23]-r.physics[24]))<1e-12);assert(Math.abs(r.physics[40]-(r.physics[24]-r.physics[23]))<1e-12);assert.deepEqual(physical,before);}
const short=R.estimate(['B','P']);assert(!short.available);assert.deepEqual(R.combine(short,v(.4)).physics,v(.4));
const noisy=v(.2);noisy[42]=.9;assert.equal(R.combine(roadB,noisy).audit.physics_weight_effective,0);R.estimate(Array(70).fill('P'));assert.deepEqual(R.estimate(Array(30).fill('B')),roadB);assert.throws(()=>R.estimate(['X']));assert.throws(()=>R.estimate(['B'],{prior:{B:1,P:1,T:1}}));
console.log('PASS road causal context/rebuild, B/P ownership, no physical flip, cap, fallback, uncertainty, EV/tie, no vector mutation');
