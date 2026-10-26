#!/usr/bin/env node
"use strict";
/*
 * V15-B SHADOW ONLY, never modifies the checked-in model or runtime constants.
 * Usage: node v15_balanced_replay.js observed_hands.json report.json
 *        node v15_balanced_replay.js --synthetic 5 /tmp/v15-smoke.json
 * All policies use separate isolated runtime + separate history state per shoe.
 * 60% train, 20% calibration and 20% one-time holdout in shoe insertion order.
 */
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const M=require("./macro_ema.js");
const args=process.argv.slice(2);
const synthetic=args[0]==="--synthetic";
const [input,output]=synthetic?[null,args[2]]:args;
if(!output||(!synthetic&&!input)){
  console.error("Usage: node v15_balanced_replay.js observed_hands.json report.json | --synthetic 5 report.json");process.exit(1);
}
const payload=synthetic
 ?{rows:require("./calibrate_macro_direction.js").synthetic(Number(args[1])),data_kind:"synthetic_mechanics_only"}
 :JSON.parse(fs.readFileSync(input,"utf8"));
const rows=Array.isArray(payload)?payload:payload.rows;
if(!Array.isArray(rows)||rows.length<5)throw Error("At least five completed hands required");
const shoes=new Map(),closed=new Set();let lastShoe=null;
for(const row of rows){
  if(row.shoe_id===undefined||row.shoe_id===null||!String(row.shoe_id).trim())throw Error("Nonempty shoe_id is mandatory");
  const id=String(row.shoe_id);
  // Exact order preserves meaningful train / calibration / future shoe holdout.
  if(id!==lastShoe){if(closed.has(id))throw Error("Interleaved shoe_id: supply chronologically grouped shoes");
    if(lastShoe!==null)closed.add(lastShoe);lastShoe=id;}
  const events=shoes.get(id)||[];
  if(row.round_index!==events.length+1)throw Error("Missing / duplicate round_index in "+id+" (ties included)");
  events.push(M.normalizeHand(row));
  shoes.set(id,events);
}
if(shoes.size<5)throw Error("At least five independent shoes required");
const ordered=[...shoes],n=ordered.length,a=Math.floor(n*.6),b=Math.floor(n*.8);
const subsets={train:ordered.slice(0,a),calibration:ordered.slice(a,b),holdout:ordered.slice(b)};
const parameters=JSON.parse(fs.readFileSync(path.join(__dirname,"v15_candidate_policies.json"),"utf8"));
if(parameters.mode!=="offline_shadow_only"||parameters.hard_limits?.auto_activate!==false)
  throw Error("V15 shadow policy config must remain non-activating");
if(!Array.isArray(parameters.candidates)||parameters.candidates.length<1||parameters.candidates.length>8)
  throw Error("Expected 1..8 explicit offline candidates");
const ids=new Set();
for(const c of parameters.candidates){
  if(typeof c.id!=="string"||!/^early_precision_[a-zA-Z0-9]+$/.test(c.id)||ids.has(c.id))
    throw Error("Invalid or duplicate candidate id");
  ids.add(c.id);
  for(const [name,cap] of [["early_band_relief",.002],["early_ev_relief",.0015]]){
    if(typeof c[name]!=="number"||!Number.isFinite(c[name])||c[name]<0||c[name]>cap)
      throw Error("Candidate relief outside offline safety bound: "+name);
  }
}
const configs=[{id:"baseline",early_band_relief:0,early_ev_relief:0},...parameters.candidates];
const stages=["all","rounds_2_5","rounds_6_10","rounds_11_20","rounds_21_40","rounds_41_50","rounds_51_plus"];
function stageFor(round){
  return round<=5?"rounds_2_5":round<=10?"rounds_6_10":round<=20?"rounds_11_20":
    round<=40?"rounds_21_40":round<=50?"rounds_41_50":"rounds_51_plus";
}
function blank(){return {eligible:0,bets:0,non_tie_bets:0,tie_pushes:0,wins:0,losses:0,skips:0,net_units:0,skip_causes:{},failed_gates:{}};}
function add(m,d){
  for(const key of ["eligible","bets","non_tie_bets","tie_pushes","wins","losses","skips","net_units"])m[key]+=d[key];
  for(const key of ["skip_causes","failed_gates"])for(const [k,v]of Object.entries(d[key]))m[key][k]=(m[key][k]||0)+v;
}
function metrics(m){
  return {...m,net_units:+m.net_units.toFixed(8),
    action_rate:m.eligible?m.bets/m.eligible:null,
    hit_rate_on_bets:m.non_tie_bets?m.wins/m.non_tie_bets:null,
    realized_ev_per_bet:m.bets?m.net_units/m.bets:null,
    realized_ev_per_opportunity:m.eligible?m.net_units/m.eligible:null,
    skip_rate:m.eligible?m.skips/m.eligible:null};
}
function diff(a,b){
  return Object.fromEntries(["action_rate","hit_rate_on_bets","realized_ev_per_bet","realized_ev_per_opportunity","skip_rate"].map(k=>
    [k,a[k]===null||b[k]===null?null:a[k]-b[k]]));
}
function seedRandom(seed){let x=seed>>>0;return()=>{x+=0x6D2B79F5;let t=x;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
function quantile(v,p){if(!v.length)return null;v.sort((a,b)=>a-b);const q=p*(v.length-1),i=Math.floor(q);return v[i]*(1-(q-i))+v[Math.min(i+1,v.length-1)]*(q-i);}
function pairedCI(base,other,seed){
  const keys=base.map(x=>x.shoe);
  if(keys.length!==other.length||keys.some((id,i)=>other[i].shoe!==id))throw Error("Paired shoe order mismatch");
  if(keys.length<2)return null;
  const rand=seedRandom(seed),history={action_rate:[],hit_rate_on_bets:[],realized_ev_per_bet:[],realized_ev_per_opportunity:[]};
  for(let b=0;b<300;b++){
    const m0=blank(),m1=blank();
    for(let k=0;k<keys.length;k++){const i=Math.floor(rand()*keys.length);add(m0,base[i].stages.all);add(m1,other[i].stages.all);}
    const d=diff(metrics(m1),metrics(m0));
    for(const key of Object.keys(history))if(d[key]!==null&&Number.isFinite(d[key]))history[key].push(d[key]);
  }
  return Object.fromEntries(Object.entries(history).map(([k,v])=>[k,{low_95:quantile(v,.025),high_95:quantile(v,.975)}]));
}
async function runtime(config){
  const store=new Map(),ctx={console,
    localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    document:{getElementById:()=>null},
    fetch:async url=>({ok:true,status:200,json:async()=>{
      const file=path.resolve(__dirname,url),model=JSON.parse(fs.readFileSync(file,"utf8"));
      if(url==="final_probability_model.json"&&config.id!=="baseline"){
        // Only in-memory modifications on first 40 rounds. No auto-promotion.
        const policy=model.decision_policy;
        policy.confidence_band.early=Math.max(policy.confidence_band.minimum,
          policy.confidence_band.early-config.early_band_relief);
        model.ev_thresholds.early=Math.max(0,model.ev_thresholds.early-config.early_ev_relief);
      }
      return model;
    }})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const name of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"]){
    vm.runInContext(fs.readFileSync(path.join(__dirname,name),"utf8"),ctx,{filename:name});
  }
  const api=ctx.__BGS_FINAL56__;await api.loadModels();
  const status=api.getModelStatus();
  if(!status.physics||!status.final56)throw Error("Physics or 57D model unavailable: "+JSON.stringify(status));
  return {api,store};
}
function tally(action,actual,diag){
  const m=blank();m.eligible++;
  if(action==="Skip"){
    m.skips++;
    const primary=diag?.primary_reason||"missing_diagnostics";
    m.skip_causes[primary]=1;
    for(const code of diag?.failed_conditions||[])m.failed_gates[code]=1;
  }else{
    if(action!=="B"&&action!=="P")throw Error("invalid action "+action);
    m.bets++;
    if(actual==="T")m.tie_pushes++;
    else{m.non_tie_bets++;if(action===actual){m.wins++;m.net_units+=action==="B"?.95:1;}
      else{m.losses++;m.net_units--;}}
  }
  return m;
}
async function replay(config,subset){
  const {api,store}=await runtime(config),perShoe=[];
  for(const [shoe,events] of subset){
    store.clear();store.set("bgs_xgb_final_shoe_id_v5",shoe);
    const history=[],byStage=Object.fromEntries(stages.map(s=>[s,blank()]));
    for(let i=0;i<events.length;i++){
      const actual=events[i],round=i+1;
      if(history.length){
        const prediction=api.applyFinalPrediction(history),r=prediction.finalProbability;
        if(r.mode!=="physics_primary"||r.error)throw Error("Prediction failed "+shoe+"/"+round+": "+r.error);
        if(!r.skip_diagnostics||r.skip_diagnostics.final_action!==prediction.direction)throw Error("missing/inconsistent V15 diagnostic");
        const evLed=r.direction_selection==="post_draw_ev_argmax_v1",lockedCandidate=evLed?r.direction_candidate:r.primary_candidate;
        if(evLed&&prediction.direction!=="Skip"&&prediction.direction!==prediction.ev_direction?.direction)throw Error("EV argmax violated: "+shoe+"/"+round);
        if(r.flipped_from_physics||prediction.direction!=="Skip"&&prediction.direction!==lockedCandidate)
          throw Error("No-flip violated: "+shoe+"/"+round);
        const m=tally(prediction.direction,actual.outcome,r.skip_diagnostics);
        add(byStage.all,m);add(byStage[stageFor(round)],m);
        api.registerPrediction(history,prediction);api.settlePending(actual.outcome,actual,history);
      }
      history.push(actual.outcome);
      api.recordCompletedHand(history,actual);
    }
    perShoe.push({shoe,stages:byStage});
  }
  const totals=Object.fromEntries(stages.map(s=>[s,blank()]));
  for(const row of perShoe)for(const stage of stages)add(totals[stage],row.stages[stage]);
  return {perShoe,metrics:Object.fromEntries(stages.map(s=>[s,metrics(totals[s])]))};
}
function summarize(entry){return entry.metrics;}
function eligibleToRecommend(baseline,candidate,delta,ci,subsetSize){
  if(subsetSize<30||baseline.all.non_tie_bets<500||candidate.all.non_tie_bets<500)return false;
  return delta.action_rate>0&&delta.hit_rate_on_bets>=0&&delta.realized_ev_per_bet>=0&&
    Number.isFinite(ci?.action_rate?.low_95)&&ci.action_rate.low_95>0&&
    Number.isFinite(ci?.hit_rate_on_bets?.low_95)&&ci.hit_rate_on_bets.low_95>=0&&
    Number.isFinite(ci?.realized_ev_per_bet?.low_95)&&ci.realized_ev_per_bet.low_95>=0;
}
(async()=>{
  const started=Date.now();
  const out={version:"V15_B_OFFLINE_ONLY",source:synthetic?"synthetic_mechanics_only":payload.data_kind||"user_supplied_unverified",
    file:synthetic?null:path.basename(input),total_shoes:shoes.size,total_hands:rows.length,
    chronological_shoe_split:Object.fromEntries(Object.entries(subsets).map(([k,v])=>[k,v.length])),
    definitions:{action_rate:"B/P bets divided by eligible hands (first round of each shoe is excluded)",
      hit_rate_on_bets:"wins divided by non-tie B/P bets; ties are pushes",
      realized_ev_per_bet:"fixed-stake realized net units / all bets, Banker win +0.95, Player win +1, loss -1, tie 0",
      stage_definitions:stages,confidence_interval:"paired 300-iteration shoe-cluster bootstrap",
      note:"No training/weights/UI/guard source edited; candidate params injected into separate in-memory model instances. Synthetic data can never establish live profitability."},
    candidates:configs.slice(1),splits:{},selection:null,promotion:{eligible:false,reason:"not_evaluated",auto_activate:false}};
  for(const [name,subset]of Object.entries(subsets)){
    if(name==="holdout")continue; // One-time holdout is intentionally untouched during selection.
    const baseline=await replay(configs[0],subset),proposals=[];
    for(const config of configs.slice(1)){
      const candidate=await replay(config,subset),delta=diff(candidate.metrics.all,baseline.metrics.all);
      proposals.push({id:config.id,metrics:summarize(candidate),delta_vs_baseline:delta,
        paired_ci95:pairedCI(baseline.perShoe,candidate.perShoe,20261011+proposals.length)});
    }
    out.splits[name]={baseline:summarize(baseline),candidates:proposals};
  }
  // Candidate choice uses TRAIN and CALIBRATION only, never the future holdout.
  const train=out.splits.train,cal=out.splits.calibration;
  const selected=cal.candidates.filter((c,i)=>{
    const t=train.candidates[i],d=c.delta_vs_baseline;
    return d.action_rate>0&&d.hit_rate_on_bets>=0&&d.realized_ev_per_bet>=0&&
      t.delta_vs_baseline.action_rate>0&&t.delta_vs_baseline.hit_rate_on_bets>=0&&t.delta_vs_baseline.realized_ev_per_bet>=0;
  }).sort((x,y)=>y.delta_vs_baseline.action_rate-x.delta_vs_baseline.action_rate)[0];
  out.selection=selected?.id||null;
  const baselineHoldout=await replay(configs[0],subsets.holdout);
  out.splits.holdout={baseline:summarize(baselineHoldout),candidate:null};
  if(selected){
    const cfg=configs.find(x=>x.id===selected.id),candidateHoldout=await replay(cfg,subsets.holdout);
    const d=diff(candidateHoldout.metrics.all,baselineHoldout.metrics.all),ci=pairedCI(baselineHoldout.perShoe,candidateHoldout.perShoe,20269999);
    out.splits.holdout.candidate={id:cfg.id,metrics:summarize(candidateHoldout),delta_vs_baseline:d,paired_ci95:ci};
    const largeEnough=eligibleToRecommend(baselineHoldout.metrics,candidateHoldout.metrics,d,ci,subsets.holdout.length);
    const realEvidence=!synthetic&&payload.data_kind==="real_observed";
    const qualified=realEvidence&&largeEnough;
    out.promotion={eligible:qualified,auto_activate:false,
      reason:!realEvidence?"source_not_verified_real_observations":
        !largeEnough?"insufficient_holdout_evidence_or_negative_lower_ci":"candidate_for_manual_review_only",
      candidate_id:qualified?cfg.id:null,requires_manual_approval:true};
  }else out.promotion={eligible:false,auto_activate:false,reason:"no_train_calibration_candidate_improved_both_hit_rate_and_ev",requires_manual_approval:true};
  out.elapsed_seconds=(Date.now()-started)/1000;
  fs.writeFileSync(output,JSON.stringify(out,null,2)+"\n");
  console.log(JSON.stringify({ok:true,source:out.source,total_shoes:out.total_shoes,eligible_train:out.splits.train.baseline.all.eligible,
    selected:out.selection,promotion:out.promotion,elapsed_seconds:out.elapsed_seconds}));
})().catch(e=>{console.error(e);process.exit(1);});
