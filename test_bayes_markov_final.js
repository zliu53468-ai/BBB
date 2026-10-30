'use strict';
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm'),E=require('./bayes_markov_runtime'),{make,plain}=require('./test_pf_primary_runtime');
const cfg=require('./cardcount_profile.json').bayes_markov_final_v1,physics=Array(48).fill(0);physics[23]=.46;physics[24]=.44;physics[25]=.10;physics[42]=.3;
const estimate=h=>E.estimate(h,.5,physics,cfg);
assert.equal(estimate([]).selected_order,0);assert.equal(estimate(['B','P']).selected_order,0);assert.throws(()=>estimate(['X']),/invalid_history/);
const h='BPTBBPPTBPBPBBBBPPPTBPBPTBBPBPPBBPTBPBPP'.split(''),a=estimate(h);assert.deepEqual(estimate(h),a);
for(const row of a.orders){let b=0,p=0,t=0;for(let i=row.order;i<h.length;i++)if(h.slice(i-row.order,i).join('')===row.context){b+=h[i]==='B';p+=h[i]==='P';t+=h[i]==='T';}assert.equal(row.banker,b);assert.equal(row.player,p);assert.equal(row.ties,t);assert(row.bayesian_pb>0&&row.bayesian_pb<1);}
assert.equal(estimate(Array(30).fill('B')).selected_order,3);assert.equal(estimate(Array(30).fill('P')).selected_order,3);assert(estimate(Array(30).fill('B')).raw_pb>.5);assert(estimate(Array(30).fill('P')).raw_pb<.5);
const supported=E.estimate(Array(10).fill('B'),.5,physics,{...cfg,min_support:{1:4,2:20,3:30}});assert.equal(supported.selected_order,1);
const two=E.estimate(Array(10).fill('B'),.5,physics,{...cfg,min_support:{1:4,2:5,3:30}});assert.equal(two.selected_order,2);
(async()=>{
const on=await make({bayes:true,road:true,firstOrder:true,dualLive:true,bias:true}),hi=await make({bayes:true,xgb:20}),lo=await make({bayes:true,xgb:-20});
for(const app of [on,hi,lo])app.store.set('bgs_xgb_final_shoe_id_v5','bayes-test');
on.ctx.__BGS_PARTICLE500__.estimate=()=>{throw Error('legacy PF executed');};on.ctx.BBBVNext.physicsCore=()=>{throw Error('old fusion executed');};
assert.equal(on.api.roadPrimaryConfig(),null);assert.equal(on.api.pfPrimaryConfig(),null);assert.equal(on.api.relativeBaselineConfig(),null);assert.equal(on.api.experimentalSettings(),null);
const signature=p=>plain({p:p.finalProbability.finalPB,d:p.direction,b:p.ev_banker,pv:p.ev_player});
for(const seq of [[],h,Array(50).fill('B'),Array(50).fill('P'),Array(70).fill('T')]){
const p=on.api.applyFinalPrediction(seq),r=p.finalProbability;assert.equal(r.mode,'bayes_markov_final');assert.equal(r.xgb_influence,0);assert.equal(r.particleDiagnostics,null);assert.equal(r.physics.length,48);assert.equal(r.extended.length,57);assert(p.next_hand_forecast.available);assert.deepEqual(signature(hi.api.applyFinalPrediction(seq)),signature(lo.api.applyFinalPrediction(seq)));assert(Math.abs(p.ev_banker-(1-r.p_tie)*(.95*r.finalPB-(1-r.finalPB)))<1e-12);assert(Math.abs(p.ev_player-(1-r.p_tie)*(1-2*r.finalPB))<1e-12);assert.deepEqual(signature(on.api.applyFinalPrediction(seq)),signature(p));
}
const before=on.api.applyFinalPrediction(h);on.api.applyFinalPrediction([...h,'P']);assert.deepEqual(signature(on.api.applyFinalPrediction(h)),signature(before));
on.api.registerPrediction(h,before);on.api.settlePending('T',null,h);assert(JSON.parse(on.api.exportTrainingData()).rows.at(-1).bayes_markov.causal);
const module=on.ctx.__BBB_BAYES_MARKOV__;on.ctx.__BBB_BAYES_MARKOV__=null;assert.equal(on.api.applyFinalPrediction(h).direction,'Skip');on.ctx.__BBB_BAYES_MARKOV__=module;
const off=await make({road:true,firstOrder:true,dualLive:true,bias:true}),old=await make({road:true,firstOrder:true,dualLive:true,bias:true});for(const app of [off,old])app.store.set('bgs_xgb_final_shoe_id_v5','rollback');vm.runInContext(fs.readFileSync('validation/pre_bayes_markov_runtime.js','utf8'),old.ctx);await old.ctx.__BGS_FINAL56__.loadModels();for(const seq of [[],h])assert.deepEqual(plain(off.api.applyFinalPrediction(seq)),plain(old.ctx.__BGS_FINAL56__.applyFinalPrediction(seq)));
console.log('PASS exact context counts, Bayesian priors, 1/2/3 selection, whole-shoe causality/undo, normalized tie-aware EV, XGB/PF/fusion isolation, 48D/57D unchanged, snapshot, missing module Skip and exact PR94 disabled rollback');
})().catch(e=>{console.error(e);process.exitCode=1;});
