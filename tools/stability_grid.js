'use strict';
const fs=require('fs'),{performance}=require('perf_hooks');
const {StablePhysicalPointV4}=require('../stability_runtime.js');
const m=new StablePhysicalPointV4(),rows=[],counts={B:0,P:0,SKIP:0},reasons={};
for(let p=0;p<10;p++)for(let b=0;b<10;b++){
 const t=performance.now(),r=m.analyze(p,b,{seed:0}),ms=performance.now()-t;
 counts[r.v4.direction]++;for(const reason of r.v4.reasons)reasons[reason]=(reasons[reason]||0)+1;
 if(![r.ev_banker,r.ev_player,r.v4.estimator_gap_se,r.exact_reference.deviation].every(Number.isFinite))throw Error('invalid output');
 rows.push({player:p,banker:b,direction:r.v4.direction,candidate:r.v4.candidate,reasons:r.v4.reasons,
  batches:r.sampling_stability.batches,agreement:r.sampling_stability.agreement,
  estimatorSE:r.v4.estimator_gap_se,reference:r.exact_reference,ms});
}
const result={scope:'Complete input coverage with fixed seed, not real win rate or real-world signal frequency',
 real_holdout:'INSUFFICIENT_REAL_HOLDOUT_DATA',counts,reasons,
 batch3:rows.filter(r=>r.batches===3).length,batch5:rows.filter(r=>r.batches===5).length,
 meanSE:rows.reduce((s,r)=>s+r.estimatorSE,0)/100,meanMs:rows.reduce((s,r)=>s+r.ms,0)/100,maxMs:Math.max(...rows.map(r=>r.ms)),rows};
fs.writeFileSync('validation/stability_grid.json',JSON.stringify(result,null,2));console.log(JSON.stringify({...result,rows:undefined},null,2));
