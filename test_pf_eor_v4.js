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
assert.equal(a.deplete_mc.status,'complete');
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

// Raw PF B/P/T and commission-adjusted EV may NOT be shrunk towards a baseline.
assert.equal(a.early_shoe_guard,undefined);
assert.equal(a.adaptive_score_smoothing,undefined);
assert.equal(a.quality.mode,'adaptive_confidence_only');
assert.equal(a.quality.ev_modified,false);
assert.equal(a.quality.probability_shrinkage,0);
assert.deepEqual(a.original_pf_ev,{B:a.ev_banker,P:a.ev_player});
assert.equal(a.v4.candidate,a.ev_banker>=a.ev_player?'B':'P');
assert.equal(a.v4.decision_metric,'raw_ev_gap');
assert(Math.abs(a.v4.ev_difference-(a.ev_banker-a.ev_player))<1e-12);
assert(Math.abs(a.v4.estimator_gap_se-
  Math.hypot(a.quality.mc_gap_se,a.quality.posterior_mean_se_gap))<1e-12);
assert(Math.abs(a.v4.threshold-
  Math.max(CONFIG.minEvGap,CONFIG.confidenceZ*a.v4.estimator_gap_se))<1e-12);

// Separate the per-particle MC sampling dispersion from posterior dispersion.
const diag=a.physical_ev.ev_uncertainty_diagnostics;
assert.equal(diag.calibrated_confidence_interval,false);
assert.equal(diag.estimated_uncertainty_only,true);
assert(diag.posterior_gap_dispersion_sd_debiased>=0);
assert(diag.posterior_gap_dispersion_sd_debiased<=diag.posterior_gap_dispersion_sd+1e-12);
assert(Math.abs(a.physical_ev.pf_estimator_se_gap-
  Math.hypot(a.physical_ev.mc_se_gap,a.physical_ev.posterior_mean_se_gap))<1e-12);
assert(['B','P','SKIP'].includes(a.v4.direction));

const b=sample.analyze(2,8),again=sample.analyze(5,3);
assert.deepEqual(again,a,'相同雙方總點必須獨立且可重現');
assert.notDeepEqual(b.simulated_probability,a.simulated_probability);
assert.throws(()=>sample.analyze(10,3));
assert.throws(()=>sample.analyze(5.5,2));

// Use synthetic decisions to check both directions with identical quality.
const gate=new QualityFilterV4(),mc=.006,posteriorMean=.002;
const quality={mode:'adaptive_confidence_only',quality:.98,ess_ratio:.98,
 ancestor_diversity:.99,mc_gap_se:mc,posterior_gap_spread:.02,
 posterior_mean_se_gap:posteriorMean,estimator_gap_se:Math.hypot(mc,posteriorMean)};
const eor={max_abs_ev_gap_delta:.005};
const good={point_observation_independent:true,ev_banker:.030,ev_player:0,
 quality,eor,physical_ev:{simulation_exhausted:false},
 deplete_mc:{status:'complete',ev_banker:.029,ev_player:0,paired_mc_gap_se:.003}};
const passed=gate.decide(good);
assert.equal(passed.direction,'B');
assert.equal(passed.reason,'quality_pass');
const player={...good,ev_banker:0,ev_player:.030,
 deplete_mc:{...good.deplete_mc,ev_banker:0,ev_player:.029}};
const passedPlayer=gate.decide(player);
assert.equal(passedPlayer.direction,'P');
assert.equal(passedPlayer.reason,'quality_pass');
const small=gate.decide({...good,ev_banker:.001});
assert.equal(small.direction,'SKIP');
assert.equal(small.reason,'weak_ev_gap');
const noisy={...quality,mc_gap_se:.025,posterior_mean_se_gap:0,
 estimator_gap_se:.025};
assert.equal(gate.decide({...good,quality:noisy}).reason,'weak_ev_gap');
const fragile=gate.decide({...good,eor:{max_abs_ev_gap_delta:.050}});
assert.equal(fragile.direction,'SKIP');
assert.equal(fragile.reason,'eor_fragile');
const bad=gate.decide({...good,deplete_mc:{...good.deplete_mc,
 ev_banker:-.05,ev_player:0,paired_mc_gap_se:.0001}});
assert.equal(bad.direction,'SKIP');
assert.equal(bad.reason,'deplete_disagreement');
const neutral=gate.decide({...good,ev_banker:0,ev_player:0,
 deplete_mc:{...good.deplete_mc,ev_banker:0,ev_player:0}});
assert.equal(neutral.direction,'SKIP');
const invalid=gate.decide({...good,quality:{...quality,estimator_gap_se:.0001}});
assert.equal(invalid.reason,'diagnostics_unavailable');
assert.equal(gate.decide({...good,deplete_mc:{status:'unavailable'}}).direction,'SKIP');

// A 10x10 functional grid checks accessibility; NOT a fabricated 50/50 quota.
const grid={B:0,P:0,SKIP:0};
for(let player=0;player<=9;player++)for(let banker=0;banker<=9;banker++){
 const r=sample.analyze(player,banker);
 grid[r.v4.direction]++;
 assert.equal(r.v4.candidate,r.ev_banker>=r.ev_player?'B':'P');
 assert.equal(r.quality.probability_shrinkage,0);
 assert.deepEqual(r.original_pf_ev,{B:r.ev_banker,P:r.ev_player});
 assert.equal(r.deplete_mc.status,'complete');
}
assert(grid.B>0&&grid.P>0&&grid.SKIP>0,
 'Both B/P directions must be reachable without artificial 50/50 targeting');
console.log('V4 100-input score-grid diagnostic (not real-world frequency)',grid);

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
const runtime=fs.readFileSync('pf_eor_v4_runtime.js','utf8');
assert(panel.includes('engine.analyze(pair.player,pair.banker)'));
assert(panel.includes('transitionStats?.(audit.map(outcome))'));
assert(!runtime.includes('physics.adaptiveScoreSmoother('));
assert(runtime.includes('const physical=raw,quality=rawQualityDiagnostics(raw,pf)'));
console.log('PASS raw physical EV, deconvolved uncertainty, adaptive confidence gate, EOR/Deplete, independent point-input V4');
