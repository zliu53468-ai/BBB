'use strict';
const assert=require('assert/strict'),R=require('./road_primary_runtime'),old=require('./validation/road_primary_pr81.cjs'),{make,plain}=require('./test_pf_primary_runtime');
const c=require('./cardcount_profile.json').road_primary_v1,off={...c,five_level_v1:{...c.five_level_v1,enabled:false}};
const v=Array(48).fill(.1);v[23]=.6;v[24]=.3;v[25]=.1;v[42]=0;
const rel=r=>R.reliability({...v,length:48},null,0,c);
let previous=-1;
for(let i=0;i<=1000;i++){const r=i/1000,w=v.slice();w[42]=1-r;const d=R.reliability(w,{particle_quality:1,recent_ess_ratio:1},0,c);assert(d.reliability<=r+1e-12);assert(d.reliability>=previous-1e-12);assert(d.five_level.nearest_level>=1&&d.five_level.nearest_level<=5);previous=d.reliability;}
for(const x of [.25,.5,.75]){const values=[x-1e-8,x+1e-8].map(r=>{const w=v.slice();w[42]=1-r;return R.reliability(w,{particle_quality:1,recent_ess_ratio:1},0,c).reliability;});assert(values[1]-values[0]<1e-6);}
for(let order=1;order<=5;order++){const r=R.estimate(Array(12+order).fill('B'),c);assert.equal(r.order,order);assert.equal(r.observations,12);assert(r.backoff.slice(0,-1).every(x=>x.observations<12));}
assert.equal(R.estimate(['B','P'],c).available,false);assert.equal(R.reliability(v,null,0,c).reliability,0);
for(const n of [0,13,15,17,30,70]){const h=Array(n).fill('P');assert.deepEqual(R.estimate(h,off),old.estimate(h,{...c,order:3}));assert.deepEqual(R.combine(R.estimate(h,off),v,off,{particle_quality:.7,recent_ess_ratio:.8},.2),old.combine(old.estimate(h,{...c,order:3}),v,c,{particle_quality:.7,recent_ess_ratio:.8},.2));}
const a=R.estimate(Array(30).fill('B'),c);R.estimate(Array(40).fill('P'),c);assert.deepEqual(R.estimate(Array(30).fill('B'),c),a);
(async()=>{const on=await make({road:true,sync:true,five:true}),rollback=await make({road:true,sync:true}),base=await make({road:true,sync:true});base.ctx.__BBB_ROAD_PRIMARY__=old;for(const x of [on,rollback,base])x.store.set('bgs_xgb_final_shoe_id_v5','five-level-test');
for(const n of [15,30]){const h=Array(n).fill('B');assert.deepEqual(plain(rollback.api.applyFinalPrediction(h)),plain(base.api.applyFinalPrediction(h)));}
const h=Array(30).fill('B'),p=on.api.applyFinalPrediction(h),d=p.road_primary.adaptive;assert.equal(p.road_primary.road.order,5);assert(d.five_level);assert.equal(d.reliability,d.five_level.mapped_reliability);assert.equal(d.ev_shrunk_again,false);const q=p.finalProbability.finalPB,mass=1-p.finalProbability.p_tie;assert(Math.abs(p.ev_banker-(.95*mass*q-mass*(1-q)))<1e-12);assert(d.ev_threshold_uplift>=0&&d.ev_threshold_uplift<=.004);on.api.registerPrediction(h,p);on.api.settlePending('T',null,h);const row=JSON.parse(on.api.exportTrainingData()).rows.at(-1);assert.equal(row.road_primary.road.order,5);assert(row.road_primary.adaptive.five_level);
console.log('PASS order 5/4/3/2/1 support, unchanged minimum, continuous conservative five levels, causality, PR81 rollback, runtime EV and exported diagnostics');})().catch(e=>{console.error(e);process.exitCode=1;});
