"use strict";
const assert=require('node:assert/strict'),S=require('./tools/no_skip_parameter_search.js'),C=require('./tools/no_skip_calibration_candidate.js'),V=require('./tools/no_skip_validation.js');
const f={available:true,probabilities:{B:.468,P:.432,T:.1},directional:{B:.52,P:.48},uncertainty:.2};
const rows=[];for(let s=0;s<15;s++)for(let r=1;r<=4;r++)rows.push({shoe_id:'s'+s,round_index:r,outcome:r%2?'B':'P',current:'B',probabilities:f.probabilities,probe:{direct:f,pf:f,raw:f,markov_weight:0}});
const model={type:'bounded_monotone_platt',slope:1,intercept:0,max_delta:.005};
assert.throws(()=>S.validateCalibrator({...model,slope:-1}),/Invalid/);assert.throws(()=>S.validateCalibrator({...model,max_delta:.1}),/Invalid/);
const trained={candidates:[model,{...model,max_delta:.01}],training_data_kind:'synthetic_only_not_real_holdout',training_seed:1,training_input_sha256:'train',training_cache_sha256:'cache',runtime_sha256:{runtime:'frozen'},unused_previous_holdout_shoes:['s12','s13','s14']};
const candidate=C.choose({rows},trained,{enabled:false});
const changed=rows.map(r=>({...r,outcome:['s12','s13','s14'].includes(r.shoe_id)?'P':r.outcome}));assert.deepEqual(C.choose({rows:changed},trained,{enabled:false}).selected,candidate.selected);
const cache={rows,seed:2,input_sha256:'fresh',data_kind:'synthetic_only_not_real_holdout',hashes:{runtime:'frozen'}};
const report=C.evaluate(cache,candidate,{enabled:false});assert(!report.gate.production_promotion_allowed);assert.equal(report.stages.all.candidate.coverage,1);
assert.throws(()=>C.evaluate({...cache,input_sha256:'train'},candidate,{enabled:false}),/reused/);assert.throws(()=>C.evaluate({...cache,seed:1},candidate,{enabled:false}),/Fresh/);
const strong={...candidate.selected,calibrator:{...model,intercept:.08,max_delta:.01}},pred=S.runPolicy(rows,strong,candidate.prior,{enabled:false});for(const p of pred){const q=p.probabilities.B/(p.probabilities.B+p.probabilities.P);assert(Math.abs(q-.52)<=.010000000001);}
const exact=rows.map(r=>({...r,candidate:'B'}));assert.deepEqual(V.pairedBootstrap(exact,exact,100).hit_change95,[0,0]);assert.throws(()=>V.pairedBootstrap(exact,[...exact].reverse(),100),/align/);
console.log('Bounded calibration contracts passed: monotonicity, 1pp cap, no old holdout selection, fresh seed/input, paired evidence and no auto-promotion');
