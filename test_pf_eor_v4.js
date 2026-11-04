'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const core=require('./pf_eor_v4_runtime.js');
const {PhysicalPointV4,QualityFilterV4,CONFIG}=core;
const sample=new PhysicalPointV4(),a=sample.analyze(5,3);
assert.equal(a.point_observation_independent,true);
assert.equal(a.history_length,1);
assert.equal(a.model_posterior.sampled_compositions,CONFIG.particles);
assert.equal(CONFIG.particles,96);
assert.equal(CONFIG.simulationsPerParticle,160);
assert.equal(a.physical_ev.samples,96*160);
assert.equal(a.deplete_mc.mode,'shadow');
assert.equal(a.deplete_mc.affects_display,false);
assert.equal(a.deplete_blended_into_primary,false);
assert.equal(a.deplete_mc.samples,32768);
assert.equal(a.eor.effects_by_point.length,10);
assert(a.eor.effects_by_point.every(Number.isFinite));
assert(a.eor.max_abs_ev_gap_delta>=0);
assert(a.eor.simulated_paired_scenarios>5000);
assert.equal(a.markov_used_for_direction,false);
assert.equal(a.linucb_used_for_direction,false);
assert.equal(a.context256_used_for_direction,false);
assert.equal(a.road_used_for_direction,false);
assert.deepEqual(a.simulated_probability,a.physical_ev.simulated_probability);
assert(Math.abs(Object.values(a.simulated_probability).reduce((x,y)=>x+y,0)-1)<1e-10);
assert(Math.abs(a.ev_banker-(.95*a.simulated_probability.B-a.simulated_probability.P))<1e-12);
assert(Math.abs(a.ev_player-(a.simulated_probability.P-a.simulated_probability.B))<1e-12);
assert.equal(a.adaptive_score_smoothing.mode,'adaptive_score_evidence');
assert.equal(a.early_shoe_guard,undefined);
assert.equal(a.adaptive_score_smoothing.history_progress_used,false);
assert.equal(a.adaptive_score_smoothing.probabilities_forced_balanced,false);
assert(a.adaptive_score_smoothing.blend>=.24&&a.adaptive_score_smoothing.blend<=.82);
assert.equal(a.v4.candidate,a.v4.relative_ev_signal>=0?'B':'P');
assert(Math.abs(a.v4.relative_ev_signal-
  (a.ev_banker-a.ev_player-a.v4.baseline_ev_gap))<1e-12);
assert(['B','P','SKIP'].includes(a.v4.direction));
const b=sample.analyze(2,8),again=sample.analyze(5,3);
assert.deepEqual(again,a,'相同雙方總點必須獨立、可重現，不受其他輸入影響');
assert.notDeepEqual(b.simulated_probability,a.simulated_probability);
assert.throws(()=>sample.analyze(10,3));assert.throws(()=>sample.analyze(5.5,2));
const gate=new QualityFilterV4();
const quality={quality:.98,ess_ratio:.98,ancestor_diversity:.99,
 mc_gap_se:.006,posterior_gap_spread:.02};
const eor={max_abs_ev_gap_delta:.005};
const good={point_observation_independent:true,ev_banker:.007,ev_player:0,
 quality,eor,physical_ev:{simulation_exhausted:false},
 deplete_mc:{status:'complete',ev_banker:.006,ev_player:0,paired_mc_gap_se:.006}};
const passed=gate.decide(good);assert.equal(passed.direction,'B');
const small=gate.decide({...good,ev_banker:.001});
assert.equal(small.direction,'SKIP');assert.equal(small.reason,'weak_ev_gap');
const bad=gate.decide({...good,deplete_mc:{...good.deplete_mc,
 ev_banker:-.03,ev_player:0,paired_mc_gap_se:.0001}});
assert.equal(bad.direction,'SKIP');
assert.equal(bad.reason,'deplete_disagreement');
const baseline=1.95*.458597-2*.446247;
const neutral=gate.decide({...good,ev_banker:baseline,ev_player:0,
 deplete_mc:{...good.deplete_mc,ev_banker:baseline,ev_player:0}});
assert.equal(neutral.direction,'SKIP','八副牌基準 EV 差不能被當成偏莊訊號');
const missing=gate.decide({...good,deplete_mc:{status:'unavailable'}});
assert.equal(missing.direction,'SKIP');
const balanced={B:0,P:0,SKIP:0};
for(let player=0;player<=9;player++)for(let banker=0;banker<=9;banker++){
 const r=sample.analyze(player,banker);
 balanced[r.v4.direction]++;
 assert.equal(r.adaptive_score_smoothing.mode,'adaptive_score_evidence');
 assert.equal(r.v4.candidate,r.v4.relative_ev_signal>=0?'B':'P');
}
assert(balanced.B>=10&&balanced.P>=10,'莊與閒都必須有可達的品質通過路徑');
const accepted=balanced.B+balanced.P;
assert(accepted>0&&balanced.B/accepted>.35&&balanced.B/accepted<.65,
 '100 種等權總點輸入的方向健全性檢查應大致平衡；非實戰勝率');
console.log('V4 score-grid diagnostic',balanced);
const auth=fs.readFileSync('auth-client.js','utf8');
const active=auth.match(/const scripts=\[[\s\S]*?\];/);
assert(active);
for(const name of ['quant256_online_runtime.js','physics_forecast_first_runtime.js',
 'deplete_mc_runtime.js','pf_eor_v4_runtime.js','pf_eor_v4_panel.js'])
 assert(active[0].includes(name));
for(const badName of ['quant256_online_panel.js','final_probability_runtime.js',
 'bayes_markov_runtime.js','vnext/','point_transition_panel.js','point_signal_quality_guard.js'])
 assert(!active[0].includes(badName),'legacy decision script accidentally loaded: '+badName);
assert(active[0].indexOf('deplete_mc_runtime.js')<active[0].indexOf('pf_eor_v4_runtime.js'));
assert(active[0].indexOf('pf_eor_v4_runtime.js')<active[0].indexOf('pf_eor_v4_panel.js'));
const panel=fs.readFileSync('pf_eor_v4_panel.js','utf8');
assert(panel.includes('engine.analyze(pair.player,pair.banker)'));
assert(panel.includes('transitionStats?.(audit.map(outcome))'));
console.log('PASS adaptive score-only PF96/M160, quality smoothing, baseline-relative EV, bidirectional grid, Deplete shadow, audit and loader');
