'use strict';
const assert=require('assert/strict'),fs=require('fs'),R=require('./road_primary_runtime'),old=require('./validation/road_primary_pr78.cjs'),{make,plain}=require('./test_pf_primary_runtime');
const c=require('./cardcount_profile.json').road_primary_v1,off={...c,adaptive_bayesian_v1:{...c.adaptive_bayesian_v1,enabled:false}};
assert(!c.early_bayesian_v1);assert(!fs.readFileSync('road_primary_runtime.js','utf8').includes('end_round'));
let seed=42;
for(let length=0;length<=80;length++){const h=Array.from({length},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return ['B','P','T'][seed%3];});assert.deepEqual(R.estimate(h,off),old.estimate(h,c));}
for(const side of ['B','P']){let previous=Infinity;for(const n of [15,20,21,30,45,46,50,70]){const h=Array(n).fill(side),r=R.estimate(h,c),base=old.estimate(h,c);assert.equal(r.order,base.order);assert.deepEqual(r.backoff,base.backoff);assert(Math.abs(r.directional_pb-.5)<Math.abs(base.directional_pb-.5));assert(r.bayesian.effective_prior_strength>24);assert(r.prior_strength<=previous);previous=r.prior_strength;assert(Math.abs(r.bayesian.data_weight+r.bayesian.prior_weight-1)<1e-12);assert.equal(r.bayesian.selected_order,r.order);assert(!('phase' in r.bayesian));}}
const x=R.estimate(Array(20).fill('B'),{...c,window:20}),y=R.estimate([...Array(40).fill('P'),...Array(20).fill('B')],{...c,window:20});assert.equal(x.directional_pb,y.directional_pb);assert.equal(x.prior_strength,y.prior_strength);assert.equal(x.order,y.order);
const h='BPPBTBPBPBBPTB'.split(''),before=R.estimate(h,c);R.estimate([...h,'B'],c);R.estimate(Array(50).fill('P'),c);assert.deepEqual(R.estimate(h,c),before);
assert.equal(R.estimate(Array(8).fill('B'),c).available,false);
(async()=>{const on=await make({road:true}),rollback=await make({road:true}),baseline=await make({road:true});for(const a of [on,rollback,baseline])a.store.set('bgs_xgb_final_shoe_id_v5','context-bayes-test');for(const a of [rollback,baseline])a.api.getCardcountProfile().road_primary_v1.adaptive_bayesian_v1.enabled=false;baseline.ctx.__BBB_ROAD_PRIMARY__=old;
for(const n of [15,30,60])assert.deepEqual(plain(rollback.api.applyFinalPrediction(Array(n).fill('B'))),plain(baseline.api.applyFinalPrediction(Array(n).fill('B'))));
const seq=Array(50).fill('B'),p=on.api.applyFinalPrediction(seq);assert(p.road_primary.road.bayesian.enabled);on.api.registerPrediction(seq,p);on.api.settlePending('T',null,seq);assert(JSON.parse(on.api.exportTrainingData()).rows.at(-1).road_primary.road.bayesian.enabled);
assert.equal(on.api.profileValue('decision_alpha',1),.38);assert.equal(on.api.profileValue('decision_alpha',70),.56);assert.equal(on.api.getCardcountProfile().display_gap,.005);
console.log('PASS whole-shoe context Bayes, no stage gate, support-only order selection, monotone shrinkage, PR78 rollback, causal isolation, runtime export and unchanged EMA/gap');})().catch(e=>{console.error(e);process.exitCode=1;});
