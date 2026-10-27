#!/usr/bin/env node
"use strict";
// Offline audit only. Never modifies runtime, policies or model weights.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'..');
const RUNTIME=['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js','physics_multitask_model.json','final_probability_model.json','cardcount_profile.json'];
const STAGES={all:()=>true,'1-20':r=>r<=20,'21-45':r=>r>=21&&r<=45,'46-70':r=>r>=46&&r<=70,'<=40':r=>r<=40,'41-50':r=>r>=41&&r<=50,'>50':r=>r>50,'50-70':r=>r>=50&&r<=70};
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function validateRows(payload){
 if(!payload||!Array.isArray(payload.rows)||!payload.rows.length)throw Error('Full chronological B/P/T rows required');
 const clean=[],seen=new Set();let current=null,index=0;
 for(const row of payload.rows){
  if(!row||typeof row.shoe_id!=='string'||!row.shoe_id.trim())throw Error('shoe_id required');
  if(row.shoe_id!==current){if(seen.has(row.shoe_id))throw Error('Interleaved shoes');seen.add(row.shoe_id);current=row.shoe_id;index=0;}
  if(row.round_index!==++index||!['B','P','T'].includes(row.outcome))throw Error('Consecutive complete rounds from 1, including ties and skipped hands, required');
  clean.push({shoe_id:row.shoe_id,round_index:row.round_index,outcome:row.outcome});
 }
 if(payload.data_kind==='real_observations'){
  const p=payload.provenance;
  if(payload.chronological_order!=='oldest_to_newest'||!p||typeof p.source!=='string'||!p.source.trim()||p.complete_shoes!==true||p.held_out!==true||p.model_generated_outcomes!==false)throw Error('Real provenance, complete shoes and independent holdout declarations required; external source review still required');
 }
 return clean;
}
function wilson(w,n){
 if(!n)return null;const z=1.959963984540054,p=w/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,r=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [Math.max(0,c-r),Math.min(1,c+r)];
}
function metrics(rows,key){
 let wagers=0,wins=0,losses=0,pushes=0,unavailable=0,profit=0,brier=0,bptBrier=0,binaryRows=0,probRows=0,run=0,longest=0;
 const sides={B:0,P:0,Balanced:0,unavailable:0};
 for(const row of rows){
  const direction=row[key];if(sides[direction]!==undefined)sides[direction]++;else sides.unavailable++;
  if(row.probabilities){probRows++;const p=row.probabilities;bptBrier+=['B','P','T'].reduce((sum,k)=>sum+(p[k]-Number(k===row.outcome))**2,0);if(row.outcome!=='T'){brier+=(p.B/(p.B+p.P)-Number(row.outcome==='B'))**2;binaryRows++;}}
  if(!['B','P'].includes(direction)){unavailable++;continue;}
  wagers++;if(row.outcome==='T'){pushes++;continue;}
  if(direction===row.outcome){wins++;profit+=direction==='B'?.95:1;run=0;}else{losses++;profit--;run++;longest=Math.max(longest,run);}
 }
 return {rows:rows.length,wagers,wins,losses,pushes,not_bet:unavailable,coverage:rows.length?wagers/rows.length:0,hit_rate_excluding_pushes:wins+losses?wins/(wins+losses):null,wins_per_all_hands:rows.length?wins/rows.length:null,realized_ev_per_bet:wagers?profit/wagers:null,profit_units:profit,wilson95_descriptive_only:wilson(wins,wins+losses),max_loss_streak:longest,brier_non_tie:binaryRows?brier/binaryRows:null,brier_bpt:probRows?bptBrier/probRows:null,directions:sides};
}
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)|0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function bootstrap(rows,key,samples=2000,seed=880142){
 const groups=new Map();for(const row of rows){if(!groups.has(row.shoe_id))groups.set(row.shoe_id,[]);groups.get(row.shoe_id).push(row);}
 const shoes=[...groups.values()].map(g=>metrics(g,key));if(shoes.length<2)return {shoes:shoes.length,samples:0,hit95:null,ev95:null};
 const random=rng(seed),hits=[],evs=[];
 for(let i=0;i<samples;i++){let w=0,l=0,b=0,p=0;for(let j=0;j<shoes.length;j++){const s=shoes[Math.floor(random()*shoes.length)];w+=s.wins;l+=s.losses;b+=s.wagers;p+=s.profit_units;}if(w+l)hits.push(w/(w+l));if(b)evs.push(p/b);}
 const interval=a=>{if(a.length!==samples)return null;a.sort((a,b)=>a-b);const q=p=>{const x=p*(a.length-1),i=Math.floor(x);return a[i]+(x-i)*((a[i+1]??a[i])-a[i]);};return [q(.025),q(.975)];};
 return {method:'whole-shoe percentile bootstrap',shoes:shoes.length,samples,seed,hit95:interval(hits),ev95:interval(evs)};
}
function pairedBootstrap(candidate,baseline,samples=2000,seed=880143){
 if(candidate.length!==baseline.length||candidate.some((r,i)=>r.shoe_id!==baseline[i].shoe_id||r.round_index!==baseline[i].round_index||r.outcome!==baseline[i].outcome))throw Error('Paired replay rows do not align');
 const ids=[...new Set(candidate.map(r=>r.shoe_id))];
 if(ids.length<2)return {shoes:ids.length,samples:0,hit_change95:null,ev_change95:null};
 const c=ids.map(id=>metrics(candidate.filter(r=>r.shoe_id===id),'candidate')),b=ids.map(id=>metrics(baseline.filter(r=>r.shoe_id===id),'candidate')),random=rng(seed),hits=[],evs=[];
 for(let i=0;i<samples;i++){const sums=[[0,0,0,0],[0,0,0,0]];for(let j=0;j<ids.length;j++){const k=Math.floor(random()*ids.length);for(const [n,m] of [[0,c[k]],[1,b[k]]]){sums[n][0]+=m.wins;sums[n][1]+=m.losses;sums[n][2]+=m.wagers;sums[n][3]+=m.profit_units;}}const [x,y]=sums;if(x[0]+x[1]&&y[0]+y[1])hits.push(x[0]/(x[0]+x[1])-y[0]/(y[0]+y[1]));if(x[2]&&y[2])evs.push(x[3]/x[2]-y[3]/y[2]);}
 const interval=a=>{if(a.length!==samples)return null;a.sort((x,y)=>x-y);const q=p=>{const x=p*(a.length-1),i=Math.floor(x);return a[i]+(x-i)*((a[i+1]??a[i])-a[i]);};return [q(.025),q(.975)];};
 return {method:'paired whole-shoe percentile bootstrap',shoes:ids.length,samples,seed,hit_change95:interval(hits),ev_change95:interval(evs)};
}
function assess(kind,summary,ci){
 const failures=[];
 if(kind!=='real_observations')failures.push('No independent real B/P/T holdout; simulation is not deployment evidence');
 if(ci.shoes<1000)failures.push('Fewer than 1000 independent whole shoes');
 if(summary.wins+summary.losses<10000)failures.push('Fewer than 10000 resolved predictions');
 if(summary.coverage!==1)failures.push('Not 100% B/P coverage');
 if(summary.hit_rate_excluding_pushes===null||summary.hit_rate_excluding_pushes<.52)failures.push('Observed hit rate below 52%');
 if(!ci.hit95||ci.hit95[0]<.52)failures.push('Whole-shoe 95% lower hit bound below 52%');
 if(!ci.ev95||ci.ev95[0]<=0)failures.push('Positive commission-adjusted EV not established');
 return {statistical_checks_passed:!failures.length,failures,production_promotion_allowed:false,source_review_required:true,remaining_required:['Verify source, chronology, complete shoes and zero holdout tuning','Review independent temporal subperiods and stage stability','Any changed policy requires a fresh holdout; never repeatedly select against this report'],note:'Even passing an interval is evidence under sampled conditions, never a guarantee of future hit rate'};
}
async function audit(payload,{onProgress=()=>{},onForecast=null,experimental=false}={}){
 const frozen=Object.fromEntries(RUNTIME.map(n=>[n,sha(fs.readFileSync(path.join(ROOT,n)))]));
 const rows=validateRows(payload),store=new Map(),storage={getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)};
 const ctx={console,document:{getElementById:()=>null},localStorage:storage,fetch:async f=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(ROOT,String(f).split('?')[0]),'utf8'))})};ctx.window=ctx;vm.createContext(ctx);
 for(const name of RUNTIME.filter(n=>n.endsWith('.js')))vm.runInContext(fs.readFileSync(path.join(ROOT,name),'utf8'),ctx,{filename:name});
 const api=ctx.__BGS_FINAL56__;await api.loadModels();if(experimental){if(!api.getExperimentalStatus?.([]).enabled)throw Error('Approved experimental configuration unavailable');api.setNoSkipTestMode(true);}const profile=JSON.parse(fs.readFileSync(path.join(ROOT,'cardcount_profile.json'),'utf8'));
 const tag=ctx.__BBB_BROWSER_CALIBRATION__.fingerprint(api.version,...['physics_multitask_model.json','final_probability_model.json'].map(n=>JSON.parse(fs.readFileSync(path.join(ROOT,n),'utf8'))),profile.local_calibration||{});
 const client=ctx.__BBB_BROWSER_CALIBRATION__.createClient(storage),result=[];let shoe=null,history=[];
 for(const row of rows){
  if(row.shoe_id!==shoe){shoe=row.shoe_id;history=[];store.clear();store.set('bgs_xgb_final_shoe_id_v5',shoe);}
  // Predict before reading the current outcome; point/card-count columns were discarded.
  const p=api.applyFinalPrediction(history),f=p.next_hand_forecast;
  if(!f?.available||!['B','P','T'].every(k=>Number.isFinite(f.probabilities[k])&&f.probabilities[k]>=0&&f.probabilities[k]<=1)||Math.abs(Object.values(f.probabilities).reduce((a,b)=>a+b,0)-1)>1e-6)throw Error('Model unavailable or invalid probabilities; never invent a direction');
  if(onForecast){
   const pf=p.finalProbability.physics.slice(),base=p.hitrate_max?.physics_pf;
   if(base)[pf[23],pf[24],pf[25]]=[base.B,base.P,base.T];
   onForecast(JSON.parse(JSON.stringify({shoe_id:shoe,round_index:row.round_index,direct:api.nextHandForecast(p.finalProbability.physicsDirect),pf:api.nextHandForecast(pf),raw:p.raw_next_hand_forecast,markov_prior:p.hitrate_max?.markov?.probabilities,markov_weight:p.hitrate_max?.markov?.weight||0})));
  }
  const current=api.noSkipTestDirection(f);
  // Offline candidate ONLY: zero edge is explicitly reported; no fabricated >50% confidence.
  const candidate=current==='Balanced'?'B':current;
  const ticket=experimental?api.prepareDisplayCalibrationTickets(history,p.base_raw_next_hand_forecast||p.raw_next_hand_forecast):client.prepare(history,p.raw_next_hand_forecast,shoe,tag);
  result.push({shoe_id:shoe,round_index:row.round_index,outcome:row.outcome,current,candidate,zero_edge_tie_fallback:current==='Balanced',probabilities:JSON.parse(JSON.stringify(f.probabilities)),calibration_applied:p.browser_calibration?.applied===true});
  history.push(row.outcome);if(experimental)api.commitDisplayCalibrationTickets(ticket,row.outcome,history);else client.commit(ticket,row.outcome,history,profile.local_calibration||{});api.recordCompletedHand(history,{outcome:row.outcome});
  if(result.length%100===0)onProgress(result.length);
 }
 const stages={};for(const [name,select] of Object.entries(STAGES)){const selected=result.filter(r=>select(r.round_index));stages[name]={current:metrics(selected,'current'),zero_skip_candidate:metrics(selected,'candidate')};}
 for(const [name,digest] of Object.entries(frozen))if(sha(fs.readFileSync(path.join(ROOT,name)))!==digest)throw Error('Runtime/model changed during replay: '+name);
 const ci=bootstrap(result,'candidate'),gate=assess(payload.data_kind,stages.all.zero_skip_candidate,ci);
 return {schema_version:1,data_kind:payload.data_kind||'unspecified',input_sha256:sha(JSON.stringify(payload)),runtime_sha256:frozen,protocol:{experimental_requested:experimental,unvalidated_experiment:experimental,first_hand_included:true,input:'past_BPT_only',learning:'same-browser causal calibration; commit only after actual result',tie_settlement:'push',banker_commission:.05,player_commission:0,selection:'frozen current no-skip probabilities; offline exact-tie Banker fallback, never alternate to fake balance',full_retraining:false},seed:payload.seed??null,stages,whole_shoe_bootstrap:ci,gate,rows:result};
}
async function main(args){
 const option=name=>{const i=args.indexOf(name);return i>=0?args[i+1]:null;};const output=option('--output');if(!output)throw Error('--output required');
 if(fs.existsSync(output))throw Error('Use a new report path; do not overwrite past validation evidence');
 let payload;const count=option('--synthetic'),input=option('--input');if((count!==null)===(input!==null))throw Error('Choose --input REAL.json or --synthetic SHOES');
 if(count!==null){const n=Number(count),seed=Number(option('--seed')||2026102801);if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('32-bit seed required');payload={data_kind:'synthetic_only_not_real_holdout',seed,rows:require(path.join(ROOT,'calibrate_macro_direction.js')).synthetic(n,seed)};}else payload=JSON.parse(fs.readFileSync(input,'utf8'));
 const report=await audit(payload,{experimental:args.includes('--experimental'),onProgress:n=>console.log(n+' prequential hands validated')});fs.mkdirSync(path.dirname(path.resolve(output)),{recursive:true});fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({output,summary:report.stages.all,ci:report.whole_shoe_bootstrap,gate:report.gate},null,2));return report.gate.statistical_checks_passed?0:2;
}
module.exports={validateRows,wilson,metrics,bootstrap,pairedBootstrap,assess,audit};
if(require.main===module)main(process.argv.slice(2)).then(code=>{process.exitCode=code}).catch(e=>{console.error(e);process.exitCode=1});
