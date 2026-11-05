'use strict';
// Pinned A vs B diagnostics. These are not real-world win-rate tests.
const fs=require('fs'),path=require('path'),vm=require('vm'),{performance}=require('perf_hooks');
const root=path.resolve(__dirname,'..'),core=require(root+'/pf_eor_v4_runtime.js');
const quant=require(root+'/quant256_online_runtime.js');
const A={...core.CONFIG,simulationsPerParticle:160,eorParticleCount:16};
const B={...core.CONFIG},C={...core.CONFIG,noiseGuardMultiplier:.80};
function stats(values){const s=values.slice().sort((a,b)=>a-b);return {mean:s.reduce((a,b)=>a+b,0)/s.length,p50:s[Math.floor(s.length*.5)],p95:s[Math.min(s.length-1,Math.floor(s.length*.95))],max:s.at(-1)};}
const rows=[],modelA=new core.PhysicalPointV4(A),modelB=new core.PhysicalPointV4(B),filterC=new core.QualityFilterV4(C);
for(let p=0;p<10;p++)for(let b=0;b<10;b++){
 let t=performance.now();const a=modelA.analyze(p,b,{seed:0}),aMs=performance.now()-t;
 t=performance.now();const v=modelB.analyze(p,b,{seed:0}),bMs=performance.now()-t;
 for(const r of [a,v]){
  if(![r.ev_banker,r.ev_player,r.v4.estimator_gap_se,r.eor.max_abs_ev_gap_delta].every(Number.isFinite))throw Error('nonfinite diagnostic');
  if(r.v4.candidate!==(r.ev_banker-r.ev_player>=core.BASE_GAP?'B':'P'))throw Error('EV direction changed');
 }
 rows.push({player:p,banker:b,A:a,B:v,C:filterC.decide(v),aMs,bMs});
}
function summary(key){const counts={B:0,P:0,SKIP:0},reasons={};let opposite=0;
 for(const row of rows){const r=key==='A'?row.A:row.B,d=key==='C'?row.C:r.v4;counts[d.direction]++;reasons[d.reason]=(reasons[d.reason]||0)+1;
  if((r.ev_banker-r.ev_player-core.BASE_GAP)*(r.deplete_mc.ev_banker-r.deplete_mc.ev_player-core.BASE_GAP)<0)opposite++;}
 return {grid_counts:counts,skip_reasons:reasons,pf_deplete_raw_opposite_grid_count:opposite,
  mean_mc_gap_se:stats(rows.map(r=>(key==='A'?r.A:r.B).quality.mc_gap_se)).mean,
  mean_estimator_gap_se:stats(rows.map(r=>(key==='A'?r.A:r.B).quality.estimator_gap_se)).mean,
  latency_ms:stats(rows.map(r=>key==='A'?r.aMs:r.bMs))};}
// Keep estimator, input and particle ensemble fixed; vary only the diagnostic
// EOR paired-draw stream in this test sandbox. Never optimize a production seed.
const source=fs.readFileSync(root+'/pf_eor_v4_runtime.js','utf8'),stability=[];
for(const [p,b] of [[0,0],[5,3],[2,8],[9,9],[8,1],[4,6],[1,4],[7,2]]){
 const pf=new quant.BlindPhysicsEstimator({particles:96,exactSingleScore:true,seed:0xBBB256});
 pf.observe(b>p?'B':p>b?'P':'T',{player:p,banker:b});
 const sets={A:[],B:[]};
 for(const salt of [0,101,997,65537]){
  const ctx={module:{exports:{}},require:n=>require(path.resolve(root,n))};
  vm.runInNewContext(source.replace('0x45d3e77^','(0x45d3e77^'+salt+')^'),ctx);
  for(const [name,count] of [['A',16],['B',24]])sets[name].push(ctx.module.exports.computeEOR(pf,{player:p,banker:b},{particles:count,samples:96}).effects_by_point);
 }
 const dispersion=name=>Array.from({length:10},(_,i)=>{const a=sets[name].map(s=>s[i]),m=a.reduce((x,y)=>x+y,0)/a.length;return a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1);}).reduce((a,b)=>a+b,0)/10;
 stability.push({player:p,banker:b,A_mean_variance:dispersion('A'),B_mean_variance:dispersion('B')});
}
const result={baseline_commit:'4f0ea49fe04925f775813b968a14db26abf40983',decision_rule:'baseline_centered_EV_for_all_config_variants',kind:'functional_and_numerical_grid_not_real_holdout',
 real_holdout_status:'INSUFFICIENT_REAL_HOLDOUT_DATA',A_config:A,B_config:B,C_experiment_config:C,
 A:summary('A'),B:summary('B'),C:summary('C'),
 eor_stability:{method:'8 fixed pairs, 4 diagnostic RNG streams, same posterior particles; limited diagnostic, not a confidence guarantee',rows:stability,
 A_mean_variance:stats(stability.map(r=>r.A_mean_variance)).mean,B_mean_variance:stats(stability.map(r=>r.B_mean_variance)).mean},
 decision:'Use 0.65 in production as a small observation reduction; no real holdout evidence of higher hit rate. Retain 0.80 as comparison only',
 rows:rows.map(r=>({player:r.player,banker:r.banker,A:r.A.v4,B:r.B.v4,C:r.C,aMs:r.aMs,bMs:r.bMs}))};
fs.mkdirSync(root+'/validation',{recursive:true});fs.writeFileSync(root+'/validation/v41_comparison.json',JSON.stringify(result,null,2));
console.log(JSON.stringify({...result,rows:undefined},null,2));
