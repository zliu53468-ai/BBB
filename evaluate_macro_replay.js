#!/usr/bin/env node
"use strict";
/* Causal, fixed-unit replay. Input is the completed-hands JSON exported by UI.
 * node evaluate_macro_replay.js hands.json report.json [baseline-directory]
 * Never uses the current outcome/points/cards until AFTER making a prediction.
 */
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const M=require("./macro_ema.js");
const args=process.argv.slice(2);
const option=args.indexOf("--candidate-coefficients");
let candidateCoefficients=null;
if(option>=0){
  if(!args[option+1])throw new Error("--candidate-coefficients requires JSON or a calibration report file");
  const raw=args[option+1].trim();
  const values=JSON.parse(raw.startsWith("{")?raw:fs.readFileSync(raw,"utf8"));
  const candidate=values.candidate_coefficients||values;
  candidateCoefficients={};
  for(const key of ["six_card","low_score","point_diff"]){
    const v=candidate[key];
    if(typeof v!=="number"||!Number.isFinite(v)||Math.abs(v)>.020000001)
      throw new Error("Candidate coefficient "+key+" must be a number between -.02 and .02");
    candidateCoefficients[key]=v;
  }
  args.splice(option,2);
}
const [input,output,baseline]=args;
if(!input||!output){console.error("Usage: node evaluate_macro_replay.js hands.json report.json [baseline-directory]");process.exit(1);}
const payload=JSON.parse(fs.readFileSync(input,"utf8")),rows=Array.isArray(payload)?payload:payload.rows;
if(!Array.isArray(rows)||!rows.length)throw new Error("completed-hands rows required");
const shoes=new Map();
for(const row of rows){
  if(row.shoe_id===undefined||row.shoe_id===null)throw new Error("shoe_id required");
  const key=String(row.shoe_id),group=shoes.get(key)||[];
  if(row.round_index!==group.length+1)throw new Error("Each shoe needs consecutive round_index, including ties/skipped hands");
  group.push(M.normalizeHand(row));shoes.set(key,group);
}
function empty(){return {eligible_hands:0,bets:0,non_tie_bets:0,tie_pushes:0,wins:0,losses:0,unit_profit:0,skips:0};}
function record(m,action,actual){
  m.eligible_hands++;
  if(action==="Skip"){m.skips++;return;}
  m.bets++;if(actual==="T"){m.tie_pushes++;return;}
  m.non_tie_bets++;
  if(action===actual){m.wins++;m.unit_profit+=action==="B"?.95:1;}
  else{m.losses++;m.unit_profit--;}
}
function metrics(m){return {...m,unit_profit:+m.unit_profit.toFixed(10),
  hit_rate_on_bets:m.non_tie_bets?m.wins/m.non_tie_bets:null,
  realized_ev_per_bet:m.bets?m.unit_profit/m.bets:null,
  skip_rate:m.eligible_hands?m.skips/m.eligible_hands:null};}
async function runtime(directory,coefficients=null){
  const store=new Map(),ctx={console,localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},document:{getElementById:()=>null},
    fetch:async url=>({ok:true,json:async()=>{
      const bundle=JSON.parse(fs.readFileSync(path.join(directory,url),"utf8"));
      if(coefficients&&url==="final_probability_model.json"){
        bundle.decision_policy.macro_ema.coefficients={...coefficients};
        bundle.decision_policy.macro_ema.coefficient_units="probability_delta";
      }
      return bundle;
    }})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const filename of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"]){
    const file=path.join(directory,filename);if(fs.existsSync(file))vm.runInContext(fs.readFileSync(file,"utf8"),ctx,{filename});
  }
  const api=ctx.__BGS_FINAL56__;await api.loadModels();return {api,store};
}
async function replay(directory,coefficients=null){
  const {api,store}=await runtime(directory,coefficients),totals=Object.fromEntries(["all","early","middle","late"].map(k=>[k,empty()]));
  let maxMacroDelta=0,unfitted=0,flips=0;
  for(const [shoe,events] of shoes){
    store.clear();store.set("bgs_xgb_final_shoe_id_v5",shoe);const history=[];
    for(let i=0;i<events.length;i++){
      const actual=events[i],round=i+1,stage=round<=40?"early":round<=50?"middle":"late";
      if(history.length){ // Matches UI: no prediction before any history.
        const prediction=api.applyFinalPrediction(history),r=prediction.finalProbability;
        if(r.error||r.mode!=="physics_primary")throw new Error("Replay runtime error: "+r.error);
        if(prediction.direction!=="Skip"&&prediction.direction!==r.primary_candidate)flips++;
        maxMacroDelta=Math.max(maxMacroDelta,Math.abs(r.macro_adjustment?.delta||0));
        if(r.macro_adjustment?.reason==="zero_coefficients_or_signal")unfitted++;
        record(totals.all,prediction.direction,actual.outcome);record(totals[stage],prediction.direction,actual.outcome);
        api.registerPrediction(history,prediction);api.settlePending(actual.outcome,actual,history);
      }
      history.push(actual.outcome);if(api.recordCompletedHand)api.recordCompletedHand(history,actual);
    }
  }
  return {version:api.version,metrics:Object.fromEntries(Object.entries(totals).map(([k,v])=>[k,metrics(v)])),no_flip_violations:flips,max_macro_delta:maxMacroDelta,zero_coefficient_or_signal_predictions:unfitted};
}
(async()=>{
  const started=Date.now(),report={input:path.basename(input),data_kind:payload.data_kind||"user_supplied_observations",shoes:shoes.size,completed_hands:rows.length,
    definitions:{hit_rate_on_bets:"wins / non-tie bets; ties excluded",realized_ev_per_bet:"fixed-unit net profit / all bets; banker +0.95, player +1, loss -1, tie 0",skip_rate:"skips / eligible hands; first hand per shoe excluded",stages:"early 1-40, middle 41-50, late 51+",timing:"predict first, then settle/update observations",limitation:"Small/synthetic runs are mechanics checks, not evidence of profitable prediction."},
    current:await replay(__dirname)};
  if(baseline)report.baseline=await replay(path.resolve(baseline));
  if(candidateCoefficients){
    report.candidate_coefficients=candidateCoefficients;
    report.candidate=await replay(__dirname,candidateCoefficients);
    report.candidate_minus_current={};
    for(const stage of ["all","early","middle","late"]){
      report.candidate_minus_current[stage]={};
      for(const metric of ["hit_rate_on_bets","realized_ev_per_bet","skip_rate"]){
        const a=report.candidate.metrics[stage][metric],b=report.current.metrics[stage][metric];
        report.candidate_minus_current[stage][metric]=a===null||b===null?null:a-b;
      }
    }
  }
  report.elapsed_seconds=(Date.now()-started)/1000;
  fs.writeFileSync(output,JSON.stringify(report,null,2)+"\n");
  console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exit(1);});
