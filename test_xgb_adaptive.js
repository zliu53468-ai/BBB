'use strict';
const assert=require('assert/strict'),fs=require('fs'),crypto=require('crypto'),X=require('./xgb_adaptive_runtime'),{make,plain}=require('./test_pf_primary_runtime');
const c=require('./cardcount_profile.json').bayes_markov_final_v1.xgb_adaptive_v1;
function rows(h,base=.4,xgb=.8){return h.map((a,i)=>({shoe_id:'s',pf_primary_policy_tag:'p',round_index:i+1,history_fingerprint:h.slice(0,i).join(''),actual_outcome:a,bayes_markov:{xgb_adaptive:{model_tag:'m',base_pb:base,xgb_pb:xgb}}}));}
const h=Array(24).fill('B'),r=rows(h),mix=(base=.4,xgb=.8,rs=r,seq=h,noise=.3,cc=c)=>X.mix(base,xgb,rs,seq,'s','m','p',noise,cc);
assert.equal(mix(.4,.8,r.slice(0,11)).audit.weight,0);assert(mix().audit.weight>0);assert(mix().audit.weight<=.25);assert(Math.abs(mix().audit.delta)<=.0100000001);assert.equal(mix().value,.41000000000000003);const mirror=X.mix(.6,.2,rows(Array(24).fill('P'),.6,.2),Array(24).fill('P'),'s','m','p',.3,c);assert(mirror.audit.delta<0);assert(mirror.audit.weight>0);assert.equal(mix(.4,.8,r,h,1).audit.weight,0);assert.equal(mix(.4,null).value,.4);assert.equal(mix(.4,.8,r,h,.3,{...c,enabled:false}).value,.4);
assert.equal(mix(.4,.8,rows(h,.8,.4)).audit.weight,0);assert.equal(mix(.4,.8,rows(Array(24).fill('T')),Array(24).fill('T')).audit.samples,0);assert.equal(mix(.4,.8,[...r,...r]).audit.samples,24);assert.equal(mix(.4,.8,r,h.slice(0,8)).audit.samples,8);
for(const changes of [{shoe_id:'other'},{pf_primary_policy_tag:'other'},{history_fingerprint:'other'},{actual_outcome:'P'},{bayes_markov:{xgb_adaptive:{model_tag:'other',base_pb:.4,xgb_pb:.8}}}])assert.equal(mix(.4,.8,r.map(x=>({...x,...changes}))).audit.samples,0);
assert.deepEqual(mix(),mix());assert.equal(mix(.4,.8,r,h,.3,{...c,max_weight:9,max_delta:1}).audit.max_weight,.25);assert.equal(mix(.4,.8,r,h,.3,{...c,max_delta:1}).audit.max_delta,.01);
const source=fs.readFileSync('final_probability_runtime.js','utf8').replace(/\r\n/g,'\n'),manifest=require('./validation/xgb_adaptive_unchanged_helpers.json'),next={applyProbabilityBounds:'dynamicEmaAlpha',applyDynamicSmoothing:'dynamicBand',dynamicBand:'decisionPolicy',decisionPolicy:'softConfidence',softConfidence:'volumeGuardState',volumeGuardState:'directPhysicsProgressWeight'};
for(const [name,hash] of Object.entries(manifest)){const a=source.indexOf('function '+name+'('),b=source.indexOf('function '+next[name]+'(',a);assert.equal(crypto.createHash('sha256').update(source.slice(a,b)).digest('hex'),hash,name+' changed');}
(async()=>{
 const app=await make({bayes:true,adaptive:true}),off=await make({bayes:true}),hi=await make({bayes:true,adaptive:true,xgb:20}),lo=await make({bayes:true,adaptive:true,xgb:-20});
 const seq='BPBPPBBPTBPBPPBBPBPBBPPTBPBP'.split('');for(const a of [app,off,hi,lo])a.store.set('bgs_xgb_final_shoe_id_v5','adaptive-s');
 const p=app.api.applyFinalPrediction(seq),q=off.api.applyFinalPrediction(seq),a=p.bayes_markov.xgb_adaptive;
 assert(Number.isFinite(a.xgb_pb));assert.equal(a.samples,0);assert.equal(a.weight,0);assert.equal(p.finalProbability.finalPB,q.finalProbability.finalPB);assert.equal(p.ev_banker,q.ev_banker);assert.equal(p.direction,q.direction);
 assert(hi.api.applyFinalPrediction(seq).bayes_markov.xgb_adaptive.xgb_pb>.99);assert(lo.api.applyFinalPrediction(seq).bayes_markov.xgb_adaptive.xgb_pb<.01);
 app.api.registerPrediction(seq,p);app.api.settlePending('B',null,seq);const policyTag=JSON.parse(app.api.exportTrainingData()).rows.at(-1).pf_primary_policy_tag;
 const key=[...app.store.keys()].find(k=>k.includes('training')&&Array.isArray(JSON.parse(app.store.get(k)||'null'))&&JSON.parse(app.store.get(k)||'[]').some(r=>r.schema_version));assert(key);
 // Explicit synthetic unit fixture: proves runtime gating wiring, not a backtest.
 const mock=rows(seq,.4,.8).map(row=>({...row,shoe_id:'adaptive-s',pf_primary_policy_tag:policyTag,bayes_markov:{xgb_adaptive:{...row.bayes_markov.xgb_adaptive,model_tag:a.model_tag}}}));
 // Favor the actual XGB stream in the fixture while keeping history-prefix validity.
 for(const row of mock){const y=row.actual_outcome==='B'?1:0;row.bayes_markov.xgb_adaptive.base_pb=1-y;row.bayes_markov.xgb_adaptive.xgb_pb=y;}
 // Use the runtime's actual training key (export contract) via a real prediction.

 app.store.set(key,JSON.stringify(mock));const mixed=app.api.applyFinalPrediction(seq),aa=mixed.bayes_markov.xgb_adaptive;assert(aa.samples>=12);assert(aa.weight>0);assert(Math.abs(aa.delta)<=.010000001);assert.equal(mixed.finalProbability.xgb_influence,aa.weight);assert(Math.abs(mixed.ev_banker-(1-mixed.finalProbability.p_tie)*(.95*mixed.finalProbability.finalPB-(1-mixed.finalProbability.finalPB)))<1e-12);
 assert.deepEqual(plain(app.api.applyFinalPrediction(seq)),plain(mixed));assert(app.api.applyFinalPrediction(seq.slice(0,6)).bayes_markov.xgb_adaptive.samples<=6);
 app.api.registerPrediction(seq,mixed);app.api.settlePending('T',null,seq);assert(JSON.parse(app.api.exportTrainingData()).rows.at(-1).bayes_markov.xgb_adaptive.model_tag);
 app.store.set('bgs_xgb_final_shoe_id_v5','other-shoe');assert.equal(app.api.applyFinalPrediction(seq).bayes_markov.xgb_adaptive.samples,0);
 const saved=app.ctx.__BBB_XGB_ADAPTIVE__;app.ctx.__BBB_XGB_ADAPTIVE__=null;assert.equal(app.api.applyFinalPrediction(seq).finalProbability.finalPB,off.api.applyFinalPrediction(seq).finalProbability.finalPB);app.ctx.__BBB_XGB_ADAPTIVE__=saved;
 console.log('PASS paired prequential Brier, sample/window/noise gates, hard weight/delta caps, ties/dedup/undo/shoe/model isolation, live XGB execution, raw-only mixing, unchanged clip/EMA/EV/VG source hashes, export and disabled fallback');
})().catch(e=>{console.error(e);process.exitCode=1;});
