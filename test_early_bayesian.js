'use strict';
const assert=require('assert/strict'),R=require('./road_primary_runtime'),old=require('./validation/road_primary_pr78.cjs'),{make,plain}=require('./test_pf_primary_runtime');
const c=require('./cardcount_profile.json').road_primary_v1,off={...c,early_bayesian_v1:{...c.early_bayesian_v1,enabled:false}};
let seed=42;
for(let length=0;length<=80;length++){const h=Array.from({length},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return ['B','P','T'][seed%3];});assert.deepEqual(R.estimate(h,off),old.estimate(h,c));if(length>=20)assert.deepEqual(R.estimate(h,c),old.estimate(h,c));}
for(const side of ['B','P']){const h=Array(18).fill(side),r=R.estimate(h,c),base=old.estimate(h,c);assert(r.available&&base.available);assert(Math.abs(r.directional_pb-.5)<Math.abs(base.directional_pb-.5));assert(r.bayesian.effective_prior_strength>24);assert(Math.abs(r.bayesian.data_weight+r.bayesian.prior_weight-1)<1e-12);assert.equal(r.bayesian.prediction_round,19);assert(r.backoff.every(x=>x.minimum>=1));}
const early=R.estimate(Array(8).fill('B'),c);assert(early.available);assert.equal(early.order,1);assert.equal(old.estimate(Array(8).fill('B'),c).available,false);
const h='BPPBTBPBPBBPTB'.split(''),before=R.estimate(h,c);R.estimate([...h,'B'],c);assert.deepEqual(R.estimate(h,c),before);R.estimate(Array(50).fill('P'),c);assert.deepEqual(R.estimate(h,c),before);
const invalid={...c,early_bayesian_v1:{...c.early_bayesian_v1,extra_prior_strength:NaN}};assert(Number.isFinite(R.estimate(Array(18).fill('B'),invalid).directional_pb));
(async()=>{const on=await make({road:true}),baseline=await make({road:true});baseline.ctx.__BBB_ROAD_PRIMARY__=old;on.store.set('bgs_xgb_final_shoe_id_v5','bayes-parity');baseline.store.set('bgs_xgb_final_shoe_id_v5','bayes-parity');
for(const n of [20,30,46]){const seq=Array(n).fill('B');assert.deepEqual(plain(on.api.applyFinalPrediction(seq)),plain(baseline.api.applyFinalPrediction(seq)));}
const seq=Array(18).fill('B'),p=on.api.applyFinalPrediction(seq);assert(p.road_primary.road.bayesian.enabled);on.api.registerPrediction(seq,p);on.api.settlePending('T',null,seq);assert(JSON.parse(on.api.exportTrainingData()).rows.at(-1).road_primary.road.bayesian.enabled);
console.log('PASS early shrinkage, support-based backoff, exact disabled rollback, mid/late candidate + runtime parity, causal rebuild and export');})().catch(e=>{console.error(e);process.exitCode=1;});
