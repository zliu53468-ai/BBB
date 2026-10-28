'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const ROOT=path.resolve(__dirname,'..'),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
if(!isMainThread){(async()=>{
 const {runtime,settle}=require('./priority_runtime.cjs'),old=await runtime({legacy:true}),next=await runtime();
 for(const shoe of workerData.shoes){
  for(const r of [old,next]){r.data.clear();r.data.set('bgs_xgb_final_shoe_id_v5',shoe.shoe_id);}
  const history=[],out=[];
  for(const row of shoe.rows){
   const a=old.api.applyFinalPrediction(history);next.win.__labPosterior=old.win.__labPosterior;const b=next.api.applyFinalPrediction(history);
   const obs={outcome:row.outcome}; // BUTTON-ONLY: hidden points/cards never enter predictors.
   out.push({round:row.round_index,actual:row.outcome,old:a.direction,next:b.direction,old_q:a.finalProbability.finalPB,next_q:b.finalProbability.finalPB,
    old_tie:a.finalProbability.p_tie??a.next_hand_forecast.probabilities.T,next_tie:b.next_hand_forecast.probabilities.T,
    reasons:b.physics_priority.reasons,noise:b.physics_priority.noise,uncertainty:b.physics_priority.uncertainty,ess:b.physics_priority.ess_ratio,
    alpha:b.physics_priority.alpha,band:b.physics_priority.probability_band,losses_before:b.physics_priority.protection?.losses,cooldown_until:b.physics_priority.protection?.cooldown_until});
   settle(old,history,a,row.outcome,obs);settle(next,history,b,row.outcome,obs);history.push(row.outcome);
  }
  next.api.applyFinalPrediction(history); // settle/reconcile last prediction, no scoring of extra forecast.
  parentPort.postMessage({shoe_id:shoe.shoe_id,rows:out});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});}
else{(async()=>{
 const input=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]),total=process.argv[4]?Number(process.argv[4]):null;
 fs.mkdirSync(out,{recursive:true});let shoes=fs.readFileSync(input,'utf8').trim().split('\n').map(JSON.parse);if(total)shoes=shoes.slice(0,total);
 const protocol={schema_version:1,baseline:'c6f4a0afbf9fc9de3806ae662319471daadfb405',input_sha256:hash(input),shoes:shoes.length,mode:'BPT_only_no_points_no_latent_counts',commission:.05,tie:'push, excluded from win/loss streak sequence',streak_skip_policy:'Skip does not erase a losing/winning recommendation run; time-adjacent streaks also reported separately',cut:14,
  candidate_config:JSON.parse(fs.readFileSync(path.join(ROOT,'cardcount_profile.json'))).physics_priority_v1,
  source_sha256:Object.fromEntries(['physics_priority_runtime.js','point_conditioned_pf.js','final_probability_runtime.js','cardcount_profile.json','app256forward.js','app256continuation.js','physics_multitask_model.json','final_probability_model.json','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','research/priority_runtime.cjs','research/priority_backtest.cjs'].map(f=>[f,hash(path.join(ROOT,f))]))};
 const lock=path.join(out,'protocol.json');if(fs.existsSync(lock)){const prior=JSON.parse(fs.readFileSync(lock));if(JSON.stringify(prior)!==JSON.stringify(protocol))throw Error('Locked source/config/input changed; start a new experiment');}else fs.writeFileSync(lock,JSON.stringify(protocol,null,2),{flag:'wx'});
 const file=path.join(out,'predictions.jsonl'),done=new Set();if(fs.existsSync(file)){let content=fs.readFileSync(file,'utf8');const end=content.lastIndexOf('\n');if(end!==content.length-1){content=content.slice(0,end+1);fs.writeFileSync(file,content);}for(const line of content.split('\n').filter(Boolean)){const s=JSON.parse(line),expected=shoes.find(x=>x.shoe_id===s.shoe_id);if(!expected||s.rows.length!==expected.rows.length||s.rows.some((r,i)=>r.actual!==expected.rows[i].outcome||r.round!==i+1))throw Error('Bad checkpoint');if(done.has(s.shoe_id))throw Error('Duplicate shoe');done.add(s.shoe_id);}}
 const buckets=Array.from({length:12},()=>[]);shoes.filter(s=>!done.has(s.shoe_id)).forEach((s,i)=>buckets[i%12].push(s));const stream=fs.createWriteStream(file,{flags:'a'}),status={phase:'running',total:shoes.length,done:done.size,started_at:new Date().toISOString()};
 const progress=()=>fs.writeFileSync(path.join(out,'progress.json'),JSON.stringify({...status,updated_at:new Date().toISOString()}));progress();
 await Promise.all(buckets.map(shoes=>new Promise((resolve,reject)=>{const w=new Worker(__filename,{workerData:{shoes}});w.on('message',m=>{stream.write(JSON.stringify(m)+'\n');status.done++;if(status.done%10===0)progress();if(status.done%100===0)console.log(JSON.stringify(status));});w.on('error',reject);w.on('exit',c=>c?reject(Error('Worker '+c)):resolve());})));
 await new Promise(resolve=>stream.end(resolve));status.phase='complete';progress();console.log(JSON.stringify(status));
})().catch(e=>{console.error(e);process.exitCode=1;});}
