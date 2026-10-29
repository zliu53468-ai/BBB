'use strict';
const assert=require('assert/strict'),R=require('./road_primary_runtime'),old=require('./validation/road_primary_pr80.cjs'),{make,plain}=require('./test_pf_primary_runtime');
const c=require('./cardcount_profile.json').road_primary_v1;
const vector=q=>{const v=Array(48).fill(.1);v[23]=.9*q;v[24]=.9*(1-q);v[25]=.1;v[42]=.2;return v;};
const good={particle_quality:.9,recent_ess_ratio:.9},bad={particle_quality:.3,recent_ess_ratio:.4};
const road=R.estimate(Array(30).fill('B'),c),v=vector(.8),copy=v.slice();
const a=R.combine(road,v,c,good,.1),b=R.combine(road,v,c,bad,.7);assert(a.audit.adaptive.reliability>b.audit.adaptive.reliability);assert(Math.abs(b.audit.delta)<=Math.abs(a.audit.delta));assert.deepEqual(v,copy);
for(const r of [a,b]){assert.equal(Math.sign(r.physics[23]-r.physics[24]),1);assert.equal(r.physics[25],.1);assert(Math.abs(r.physics[39]-(.95*r.physics[23]-r.physics[24]))<1e-12);assert.equal(r.audit.adaptive.probability_shrink_count,1);}
const missing=R.combine(null,v,c);assert.equal(missing.audit.adaptive.reliability,0);assert(Math.abs(missing.physics[23]-missing.physics[24])<1e-12);
const fallback=R.combine(null,v,c,good,.1);assert(Math.abs(fallback.physics[23]/.9-.5)<Math.abs(.8-.5));
const off={...c,adaptive_pipeline_v1:{...c.adaptive_pipeline_v1,enabled:false}};for(const h of [[],Array(18).fill('P'),Array(50).fill('B')])assert.deepEqual(R.combine(R.estimate(h,c),v,off),old.combine(old.estimate(h,c),v,c));
(async()=>{const on=await make({road:true,sync:true}),rollback=await make({road:true}),baseline=await make({road:true});baseline.ctx.__BBB_ROAD_PRIMARY__=old;for(const x of [on,rollback,baseline])x.store.set('bgs_xgb_final_shoe_id_v5','pipeline-test');
for(const n of [15,40]){const h=Array(n).fill('B');assert.deepEqual(plain(rollback.api.applyFinalPrediction(h)),plain(baseline.api.applyFinalPrediction(h)));}
const h=Array(30).fill('B'),p=on.api.applyFinalPrediction(h),d=p.road_primary.adaptive;assert(d);assert(d.ev_threshold_uplift>=0&&d.ev_threshold_uplift<=.004);assert.equal(d.ev_shrunk_again,false);assert.equal(d.ev_banker,p.ev_banker);const mass=1-p.finalProbability.p_tie,q=p.finalProbability.finalPB;assert(Math.abs(p.ev_banker-(.95*mass*q-mass*(1-q)))<1e-12);assert(Math.abs(p.ev_player-(mass*(1-q)-mass*q))<1e-12);
const low=on.api.dynamicEmaAlpha(40,.2,{},null,{physics_uncertainty:.2},.1),hi=on.api.dynamicEmaAlpha(40,.2,{},null,{physics_uncertainty:.2},.9);assert(low<hi);assert(hi<=rollback.api.dynamicEmaAlpha(40,.2,{},null,{physics_uncertainty:.2}));
on.api.registerPrediction(h,p);on.api.settlePending('T',null,h);assert(JSON.parse(on.api.exportTrainingData()).rows.at(-1).road_primary.adaptive);const next=on.api.applyFinalPrediction([...h,'T']);assert(next.road_primary.adaptive.ema_applied);assert(next.road_primary.adaptive.ema_alpha<1);
console.log('PASS shared reliability, bounded correction/fallback, missing-data conservatism, same EV/tie, adaptive EMA, snapshot, PR80 disabled rollback');})().catch(e=>{console.error(e);process.exitCode=1;});
