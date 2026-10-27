#!/usr/bin/env node
"use strict";
// Offline, frozen search space. No production writes, model retraining or holdout selection.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {Worker,isMainThread,parentPort,workerData}=require('node:worker_threads');
const V=require('./no_skip_validation.js'),M=require('../browser_calibration.js');
const ROOT=path.resolve(__dirname,'..'),hash=v=>crypto.createHash('sha256').update(v).digest('hex');
const ALPHAS={off:[1,1,1],responsive:[.48,.58,.65],fast:[.65,.75,.84]};
const stage=r=>r<=20?0:r<=45?1:2;
function splitShoes(rows){
 const ids=[...new Set(rows.map(r=>r.shoe_id))];if(ids.length<15)throw Error('At least 15 complete shoes for chronological 60/20/20 protocol');
 const a=Math.floor(ids.length*.60),b=Math.floor(ids.length*.80),groups={train:new Set(ids.slice(0,a)),selection:new Set(ids.slice(a,b)),holdout:new Set(ids.slice(b))};
 return Object.fromEntries(Object.entries(groups).map(([k,ids])=>[k,{ids:[...ids],rows:rows.filter(r=>ids.has(r.shoe_id))}]));
}
function fitPrior(rows){const counts={B:0,P:0,T:0};for(const r of rows)counts[r.outcome]++;return Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v/rows.length]));}
function grid(){
 const configs=[];for(const source of ['direct','pf','markov4','markov8','markov12'])for(const alpha of Object.keys(ALPHAS))for(const priorWeight of [0,.04]){
  const w=source.startsWith('markov')?Number(source.slice(6))/100:0;
  if(w+priorWeight>.12+1e-12||source==='direct'&&priorWeight)continue;
  configs.push({id:`${source}_${alpha}_prior${priorWeight}`,source,alpha,prior_weight:priorWeight,markov_weight:w,architecture_eligible:source!=='direct'});
 }return configs;
}
function rawForecast(row,source){
 const p=row.probe;if(source==='direct')return p.direct;if(source==='pf')return p.pf;if(source==='markov8')return p.raw;
 const w=p.markov_weight>0?Number(source.slice(6))/100:0;
 const probs={};for(const k of ['B','P','T'])probs[k]=(1-w)*p.pf.probabilities[k]+w*(p.markov_prior?.[k]??p.pf.probabilities[k]);
 const mass=probs.B+probs.P;return {...p.pf,probabilities:probs,directional:{B:probs.B/mass,P:probs.P/mass}};
}
function validateCalibrator(m){if(!m||m.type!=="bounded_monotone_platt"||![m.slope,m.intercept,m.max_delta].every(Number.isFinite)||m.slope<.5||m.slope>1.5||Math.abs(m.intercept)>.08||m.max_delta<=0||m.max_delta>.01)throw Error("Invalid bounded monotone calibrator");}
function runPolicy(rows,c,prior,calibration){
 const m=c.source?.startsWith('markov')?Number(c.source.slice(6))/100:0;
 if(!['direct','pf','markov4','markov8','markov12'].includes(c.source)||!ALPHAS[c.alpha]||!Number.isFinite(c.prior_weight)||c.prior_weight<0||m+c.prior_weight>.12+1e-12)throw Error('Invalid policy or prior budget');
 if(c.calibrator)validateCalibrator(c.calibrator);
 const map=new Map(),storage={getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k)},client=M.createClient(storage);
 let shoe=null,history=[],previous=null;const results=[];
 for(const row of rows){
  if(shoe!==row.shoe_id){shoe=row.shoe_id;history=[];previous=null;map.clear();}
  let raw=rawForecast(row,c.source);
  if(c.calibrator){const p=raw.directional.B,m=c.calibrator,logit=Math.log(Math.max(1e-9,p)/Math.max(1e-9,1-p)),q=1/(1+Math.exp(-(m.slope*logit+m.intercept))),b=Math.max(1e-9,Math.min(1-1e-9,Math.max(p-m.max_delta,Math.min(p+m.max_delta,q)))),mass=1-raw.probabilities.T;raw={...raw,directional:{B:b,P:1-b},probabilities:{B:mass*b,P:mass*(1-b),T:raw.probabilities.T}};}
  const f=client.apply(raw,history,shoe,c.id,calibration).forecast,ticket=client.prepare(history,raw,shoe,c.id);
  let b=f.probabilities.B,p=f.probabilities.P,t=f.probabilities.T;
  // One shared <=12% prior budget; the empirical prior is train-only and same-side only.
  if(c.prior_weight&&Math.sign(prior.B-prior.P)===Math.sign(b-p)){
   b=(1-c.prior_weight)*b+c.prior_weight*prior.B;p=(1-c.prior_weight)*p+c.prior_weight*prior.P;t=(1-c.prior_weight)*t+c.prior_weight*prior.T;
  }
  const current=b/(b+p),alpha=ALPHAS[c.alpha][stage(row.round_index)],smoothed=previous===null?current:alpha*current+(1-alpha)*previous;
  // Candidate uses one causal EMA on raw forecasts, never a second EMA on finalPB.
  previous=smoothed;const direction=smoothed>=.5?'B':'P';
  results.push({shoe_id:row.shoe_id,round_index:row.round_index,outcome:row.outcome,candidate:direction,probabilities:{B:(1-t)*smoothed,P:(1-t)*(1-smoothed),T:t}});
  history.push(row.outcome);client.commit(ticket,row.outcome,history,calibration);
 }return results;
}
function ranking(a,b){return (b.metrics.hit_rate_excluding_pushes-a.metrics.hit_rate_excluding_pushes)||(b.metrics.realized_ev_per_bet-a.metrics.realized_ev_per_bet)||(a.metrics.brier_bpt-b.metrics.brier_bpt)||(a.config.markov_weight+a.config.prior_weight-b.config.markov_weight-b.config.prior_weight)||a.config.id.localeCompare(b.config.id);}
function search(rows,dataKind,calibration){
 const split=splitShoes(rows),prior=fitPrior(split.train.rows),configs=grid();
 const selection=configs.map(config=>({config,metrics:V.metrics(runPolicy(split.selection.rows,config,prior,calibration),'candidate')})).sort(ranking);
 const selected=selection.find(x=>x.config.architecture_eligible).config;
 // The holdout is first evaluated only after this selection has been frozen.
 const candidate=runPolicy(split.holdout.rows,selected,prior,calibration),baseline=split.holdout.rows.map(r=>({...r,candidate:['B','P'].includes(r.current)?r.current:'B'}));
 const summary=V.metrics(candidate,'candidate'),base=V.metrics(baseline,'candidate'),ci=V.bootstrap(candidate,'candidate'),paired=V.pairedBootstrap(candidate,baseline),gate=V.assess(dataKind,summary,ci);
 if(summary.hit_rate_excluding_pushes<base.hit_rate_excluding_pushes)gate.failures.push('Selected candidate hit rate regressed against current zero-skip baseline');
 if(summary.realized_ev_per_bet<base.realized_ev_per_bet)gate.failures.push('Selected candidate EV regressed against current zero-skip baseline');
 if(summary.brier_bpt>base.brier_bpt+.0015)gate.failures.push('Selected candidate B/P/T Brier regressed by >0.0015');
 if(!paired.hit_change95||paired.hit_change95[0]<0||!paired.ev_change95||paired.ev_change95[0]<0)gate.failures.push("Paired whole-shoe evidence does not establish non-regression");
 gate.statistical_checks_passed=!gate.failures.length;
 const stages={};for(const [name,select] of Object.entries({all:()=>true,'1-20':r=>r<=20,'21-45':r=>r>=21&&r<=45,'46-70':r=>r>=46&&r<=70,'<=40':r=>r<=40,'41-50':r=>r>=41&&r<=50,'>50':r=>r>50,'50-70':r=>r>=50&&r<=70})){stages[name]={candidate:V.metrics(candidate.filter(r=>select(r.round_index)),'candidate'),baseline:V.metrics(baseline.filter(r=>select(r.round_index)),'candidate')};}
 const controls={};for(const source of ['direct','pf','markov8']){const c={id:'control_'+source,source,alpha:'off',prior_weight:0,markov_weight:source==='markov8'?.08:0};controls[source]=V.metrics(runPolicy(split.holdout.rows,c,prior,calibration),'candidate');}
 const side=prior.B>=prior.P?'B':'P';controls.train_prior_only=V.metrics(split.holdout.rows.map(r=>({...r,candidate:side,probabilities:prior})),'candidate');
 return {schema_version:1,data_kind:dataKind,protocol:'chronological whole-shoe 60% prior fit / 20% selection / 20% one-shot holdout',split:Object.fromEntries(Object.entries(split).map(([k,v])=>[k,{shoes:v.ids.length,first_shoe:v.ids[0],last_shoe:v.ids.at(-1),rows:v.rows.length}])),search_space:configs,selection_scores:selection,train_only_prior:prior,selected_config:selected,holdout:{stages,controls,whole_shoe_bootstrap:ci,paired_change:paired},gate,production_changed:false,real_model_retrained:false,note:'No runtime changes or automatic deployment. Simulation and a point hit rate above 52% cannot establish a real sustainable 52-53% hit rate.'};
}
async function collect(payload,workers,onProgress){
 const clean=V.validateRows(payload),ids=[...new Set(clean.map(r=>r.shoe_id))],lookup=new Map(ids.map((id,i)=>[id,i]));
 const parts=Array.from({length:workers},()=>[]);for(const row of clean)parts[lookup.get(row.shoe_id)%workers].push(row);
 const replies=await Promise.all(parts.filter(p=>p.length).map((rows,index)=>new Promise((resolve,reject)=>{
  const w=new Worker(__filename,{workerData:{payload:{...payload,rows},index}});w.on('message',m=>{if(m.progress)onProgress(m.index,m.progress);else resolve(m)});w.on('error',reject);w.on('exit',code=>{if(code)reject(Error('Cache worker failed '+code))});
 })));
 const hashes=JSON.stringify(replies[0].hashes);if(replies.some(r=>JSON.stringify(r.hashes)!==hashes))throw Error('Workers loaded different runtimes');
 return {data_kind:payload.data_kind||'unspecified',seed:payload.seed??null,input_sha256:hash(JSON.stringify(payload)),rows:replies.flatMap(r=>r.rows).sort((a,b)=>lookup.get(a.shoe_id)-lookup.get(b.shoe_id)||a.round_index-b.round_index),hashes:replies[0].hashes};
}
async function main(args){
 const option=k=>{const i=args.indexOf(k);return i<0?null:args[i+1]};const output=option('--output-dir');if(!output||fs.existsSync(output))throw Error('Use a new --output-dir');
 const workers=Number(option('--workers')||4);if(!Number.isInteger(workers)||workers<1||workers>8)throw Error('workers must be 1..8');
 let payload;const input=option('--input'),n=option('--synthetic');if((input!==null)===(n!==null))throw Error('Choose --input or --synthetic');
 if(input)payload=JSON.parse(fs.readFileSync(input,'utf8'));else{const seed=Number(option('--seed')||2026102802);if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('32-bit seed required');payload={data_kind:'synthetic_only_not_real_holdout',seed,rows:require('../calibrate_macro_direction.js').synthetic(Number(n),seed)};}
 const cache=await collect(payload,workers,(worker,n)=>console.log(`worker ${worker+1}: ${n} causal forecasts`));fs.mkdirSync(output,{recursive:true});
 fs.writeFileSync(path.join(output,'cache.json'),JSON.stringify(cache)+'\n',{flag:'wx'});
 if(args.includes('--cache-only')){console.log(JSON.stringify({cache:path.join(output,'cache.json'),rows:cache.rows.length,seed:cache.seed,selection_performed:false}));return 0;}
 const config=JSON.parse(fs.readFileSync(path.join(ROOT,'cardcount_profile.json'),'utf8'));
 const report=search(cache.rows,payload.data_kind||'unspecified',config.local_calibration||{});report.input_sha256=hash(JSON.stringify(payload));report.runtime_sha256=cache.hashes;report.seed=payload.seed??null;
 fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({selected:report.selected_config,split:report.split,holdout:report.holdout.stages.all,controls:report.holdout.controls,gate:report.gate},null,2));return report.gate.statistical_checks_passed?0:2;
}
module.exports={splitShoes,fitPrior,grid,runPolicy,ranking,search,validateCalibrator};
if(!isMainThread){(async()=>{const probes=[];const r=await V.audit(workerData.payload,{onForecast:p=>probes.push(p),onProgress:n=>parentPort.postMessage({index:workerData.index,progress:n})});parentPort.postMessage({rows:r.rows.map((row,i)=>({...row,probe:probes[i]})),hashes:r.runtime_sha256});})().catch(e=>{console.error(e);process.exitCode=1});}
else if(require.main===module)main(process.argv.slice(2)).then(code=>{process.exitCode=code}).catch(e=>{console.error(e);process.exitCode=1});
