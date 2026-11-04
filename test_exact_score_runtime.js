'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const exact=require('./exact_score_runtime.js');
const guard=require('./point_signal_quality_guard.js');
const table=require('./exact_score_table.js');
const summary=require('./validation/exact_score_million_summary.json');
let mass=0;
for(let p=0;p<10;p++)for(let b=0;b<10;b++){
 const a=new exact.ExactScoreEngine(),r=a.syncHistory([b>p?'B':p>b?'P':'T'],[{player:p,banker:b}]);
 assert.equal(r.exact_enumeration.complete,true);assert.equal(r.action,'SKIP');assert.equal(r.bet_allowed,false);
 assert.equal(r.physical_ev.samples,0);assert.equal(r.physical_ev.mc_se_gap,0);
 assert(r.ev_banker<0&&r.ev_player<0);assert.equal(guard.decide(r).direction,'SKIP');
 assert.equal(a.history.length,1);assert.equal(a.scoreRows.length,1);
 const q=r.physical_ev.model_probability;assert(Math.abs(q.B+q.P+q.T-1)<1e-10);
 mass+=r.exact_enumeration.evidence_probability;
 const lengths=r.physical_ev.draw_length_diagnostics;
 for(const n of [4,5,6]){
  assert(Math.abs(Object.values(lengths.joint_probability[n]).reduce((a,b)=>a+b)-lengths.current_length_probability[n])<1e-10);
  assert(Math.abs([4,5,6].reduce((a,k)=>a+lengths.joint_probability[k][n],0)-lengths.next_length_probability[n])<1e-10);
 }
 r.physical_ev.simulated_probability.B=0;
 assert(exact.evaluate(p,b).physical_ev.simulated_probability.B>0,'report mutation cannot poison shared table');
 assert.throws(()=>a.syncHistory(['B','P'],[{player:0,banker:1},{player:1,banker:0}]),/one_resolved_hand/);
}
assert(Math.abs(mass-1)<1e-10);
const exhausted=exact.evaluate(5,3);exhausted.physical_ev.simulation_exhausted=true;
assert.equal(guard.decide(exhausted).reason,"insufficient_quality_diagnostics");
for(const args of [[-1,0],[10,0],[1.5,3],['5',3],[null,0],[0,NaN]])assert.throws(()=>exact.evaluate(...args));
const raw=JSON.parse(JSON.stringify(table));raw.rows[0][3]=.9;
assert.throws(()=>vm.runInNewContext(fs.readFileSync('exact_score_runtime.js','utf8'),{window:{BBBExactScoreTable:raw}}),/invalid_exact/);
assert.equal(summary.independent_scenarios,1000000);assert.equal(summary.dealt_hands,2000000);
assert.equal(summary.training_samples,0);assert(summary.max_bin_standardized_error<6);
assert(summary.paired_comparison.ci95[0]<0&&summary.paired_comparison.ci95[1]>0,'million run must not claim significant improvement');
assert.deepEqual(summary.exact_v4_display_counts,{B:0,P:0,SKIP:100});
const auth=fs.readFileSync('auth-client.js','utf8');
assert(auth.indexOf('exact_score_table.js')<auth.indexOf('exact_score_runtime.js'));
assert(auth.indexOf('exact_score_runtime.js')<auth.indexOf('quant256_online_panel.js'));
console.log('PASS exact enumeration: all 100 inputs, normalization, no history, fail-closed corrupt data, UI loader, million holdout evidence');
