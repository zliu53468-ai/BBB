"use strict";
const assert=require('node:assert/strict'),S=require('./tools/no_skip_parameter_search.js');
function forecast(b){return {available:true,probabilities:{B:.9*b,P:.9*(1-b),T:.1},directional:{B:b,P:1-b},uncertainty:.3};}
const rows=[];for(let s=0;s<15;s++)for(let r=1;r<=4;r++)rows.push({shoe_id:'shoe-'+s,round_index:r,outcome:r%2?'B':'P',current:'B',probabilities:forecast(.52).probabilities,probe:{direct:forecast(.52),pf:forecast(.52),raw:forecast(.52),markov_prior:null,markov_weight:0}});
const split=S.splitShoes(rows);assert.equal(split.train.ids.length,9);assert.equal(split.selection.ids.length,3);assert.equal(split.holdout.ids.length,3);assert.equal(split.train.ids.at(-1),'shoe-8');assert.equal(split.holdout.ids[0],'shoe-12');
for(const c of S.grid()){assert(c.markov_weight+c.prior_weight<=.12+1e-12);assert.equal(c.architecture_eligible,c.source!=='direct');}
const prior=S.fitPrior(split.train.rows),config={id:'fixture',source:'pf',alpha:'off',prior_weight:0};
const base=S.runPolicy(rows,config,prior,{enabled:false});assert(base.every(r=>['B','P'].includes(r.candidate)));assert.equal(base[0].candidate,base[4].candidate,'shoe reset failed');
const a=S.search(rows,'TEST_FIXTURE_ONLY',{enabled:false});
const changed=rows.map(r=>({...r,outcome:split.holdout.ids.includes(r.shoe_id)?'P':r.outcome}));const b=S.search(changed,'TEST_FIXTURE_ONLY',{enabled:false});
assert.deepEqual(a.selected_config,b.selected_config,'holdout labels changed model selection');assert.deepEqual(a.train_only_prior,b.train_only_prior,'holdout leaked into prior');assert.deepEqual(a.selection_scores,b.selection_scores,'holdout leaked into ranking');
assert.equal(a.gate.production_promotion_allowed,false);assert(a.gate.failures.some(s=>s.includes('real')));assert(a.selected_config.architecture_eligible);
// Prefix forecasts must not change when future labels change.
const sequence=[{...rows[0],probe:{...rows[0].probe,pf:forecast(.7)}},{...rows[1],probe:{...rows[1].probe,pf:forecast(.3)}}];
const c={...config,alpha:'responsive'},x=S.runPolicy(sequence,c,prior,{enabled:false}),y=S.runPolicy([...sequence,{...rows[2],outcome:'T'}],c,prior,{enabled:false});assert.deepEqual(x,y.slice(0,2));
console.log('Search protocol passed: whole-shoe split, one prior budget, architecture eligibility, prefix causality, untouched-holdout selection and no auto-promotion');

(async()=>{
 const V=require('./tools/no_skip_validation.js'),fs=require('node:fs'),probes=[];
 const input={data_kind:'TEST_FIXTURE_ONLY',rows:[...Array(2)].flatMap((_,s)=>['B','T','P'].map((outcome,i)=>({shoe_id:'fixture-'+s,round_index:i+1,outcome,banker_score:9,player_score:0})))};
 const audit=await V.audit(input,{onForecast:p=>{assert(!('outcome' in p));assert(!('banker_score' in p));probes.push(p);}});
 assert.equal(probes.length,6);
 const cached=audit.rows.map((r,i)=>({...r,probe:probes[i]})),cfg={id:'parity',source:'markov8',alpha:'off',prior_weight:0};
 const actual=S.runPolicy(cached,cfg,{B:.4586,P:.4462,T:.0952},JSON.parse(fs.readFileSync('cardcount_profile.json')).local_calibration);
 assert.deepEqual(actual.map(r=>r.candidate),audit.rows.map(r=>['B','P'].includes(r.current)?r.current:'B'));
 console.log('Actual-runtime probe hook and current-policy zero-skip parity passed');
})().catch(e=>{console.error(e);process.exitCode=1});
