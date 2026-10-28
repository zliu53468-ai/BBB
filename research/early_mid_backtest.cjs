'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict'),{Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const ROOT=path.resolve(__dirname,'..'),REF='116ffc97b62f2519659d5ceb5ef5d717da0b8c7b',hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
if(!isMainThread){(async()=>{
 const {runtime,settle}=require('./priority_runtime.cjs'),main=await runtime({legacy:true}),before=await runtime({revision:REF}),after=await runtime();
 for(const shoe of workerData.shoes){for(const r of [main,before,after]){r.data.clear();r.data.set('bgs_xgb_final_shoe_id_v5',shoe.shoe_id);}const h=[],rows=[];
  for(const row of shoe.rows){const m=main.api.applyFinalPrediction(h);before.win.__labPosterior=main.win.__labPosterior;const a=before.api.applyFinalPrediction(h);after.win.__labPosterior=main.win.__labPosterior;const b=after.api.applyFinalPrediction(h),audit=b.physics_priority;
   if(h.length<30){assert(audit.alpha<=.40);assert(b.finalProbability.particleDiagnostics.blend_weight<=.08+1e-12);assert(Math.abs(audit.residual.applied_delta)<=.015+1e-12);assert.equal(audit.calibration.delta,0);assert.equal(audit.point_weight,0);}
   rows.push({round:h.length+1,actual:row.outcome,main:m.direction,before:a.direction,after:b.direction,q_before:a.finalProbability.finalPB,q_after:b.finalProbability.finalPB,reasons:audit.reasons,band:audit.probability_band,alpha:audit.alpha,pf_weight:b.finalProbability.particleDiagnostics?.blend_weight,residual:audit.residual,calibration:audit.calibration});
   for(const [r,p]of [[main,m],[before,a],[after,b]])settle(r,h,p,row.outcome,{outcome:row.outcome});h.push(row.outcome);
  }parentPort.postMessage({shoe_id:shoe.shoe_id,rows});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});}
else{(async()=>{
 const input=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);fs.mkdirSync(out,{recursive:true});
 const shoes=fs.readFileSync(input,'utf8').trim().split('\n').map(JSON.parse);assert.equal(shoes.length,500);
 const protocol={version:'EARLY_MID_CONSERVATIVE_V2',before_ref:REF,main_ref:'c6f4a0afbf9fc9de3806ae662319471daadfb405',seed:2026102921,shoes:500,maximum_hands:70,primary_horizon:70,secondary_horizons:[50,60],synthetic:true,no_points_or_latent_cards_input:true,input_sha256:hash(input),candidate:JSON.parse(fs.readFileSync(path.join(ROOT,'cardcount_profile.json'))).physics_priority_v1,
  sources:Object.fromEntries(['final_probability_runtime.js','physics_priority_runtime.js','cardcount_profile.json','particle_filter_runtime.js','physics_multitask_model.json','final_probability_model.json','app256forward.js','app256continuation.js','research/priority_runtime.cjs','research/early_mid_backtest.cjs'].map(f=>[f,hash(path.join(ROOT,f))]))};
 const lock=path.join(out,'protocol.json');if(fs.existsSync(lock)&&fs.readFileSync(lock,'utf8')!==JSON.stringify(protocol,null,2))throw Error('Locked candidate changed');if(!fs.existsSync(lock))fs.writeFileSync(lock,JSON.stringify(protocol,null,2));
 const output=path.join(out,'predictions.jsonl');if(fs.existsSync(output))throw Error('Do not overwrite test results');const stream=fs.createWriteStream(output,{flags:'wx'}),buckets=Array.from({length:12},()=>[]);shoes.forEach((s,i)=>buckets[i%12].push(s));let done=0;
 await Promise.all(buckets.map(shoes=>new Promise((resolve,reject)=>{const w=new Worker(__filename,{workerData:{shoes}});w.on('message',s=>{stream.write(JSON.stringify(s)+'\n');done++;if(done%10===0)fs.writeFileSync(path.join(out,'progress.json'),JSON.stringify({done,total:500}));if(done%100===0)console.log(done+'/500');});w.on('error',reject);w.on('exit',c=>c?reject(Error('worker '+c)):resolve());})));
 await new Promise(resolve=>stream.end(resolve));fs.writeFileSync(path.join(out,'progress.json'),JSON.stringify({done,total:500,complete:true}));console.log('Complete500 paired shoes');
})().catch(e=>{console.error(e);process.exitCode=1;});}
