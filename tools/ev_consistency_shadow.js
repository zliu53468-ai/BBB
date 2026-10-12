#!/usr/bin/env node
"use strict";

/*
 * Offline-only chronological shoe shadow for the V14.8 safe patch.
 * It is not referenced by index.html or any production entry point.
 *
 * Usage:
 *   node tools/ev_consistency_shadow.js --synthetic 20 /tmp/report.json
 *   node tools/ev_consistency_shadow.js observed_hands.json /tmp/report.json
 */
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm"),cp=require("node:child_process");
const ROOT=path.resolve(__dirname,"..");
const BASELINE_COMMIT="7777a321be4476d38979f33625294fdfeecc118a";
const MODES=["quality","off","fixed_0.06","fixed_0.10","fixed_0.16","uncertainty_only"];
const args=process.argv.slice(2),synthetic=args[0]==="--synthetic";
const input=synthetic?null:args[0],output=synthetic?args[2]:args[1];
if(!output||(!synthetic&&!input)){
  console.error("Usage: node tools/ev_consistency_shadow.js --synthetic N report.json | hands.json report.json");
  process.exit(2);
}

const payload=synthetic
  ?{data_kind:"synthetic_mechanics_only",rows:require(path.join(ROOT,"calibrate_macro_direction.js")).synthetic(Number(args[1]||20))}
  :JSON.parse(fs.readFileSync(input,"utf8"));
const rows=Array.isArray(payload)?payload:payload.rows;
if(!Array.isArray(rows)||!rows.length)throw Error("completed hands are required");
const shoes=new Map();let last=null;const closed=new Set();
for(const row of rows){
  const id=String(row.shoe_id??"");if(!id)throw Error("nonempty shoe_id is required");
  if(last!==id){if(closed.has(id))throw Error("shoe_id values must be chronologically grouped");if(last!==null)closed.add(last);last=id;}
  const events=shoes.get(id)||[];
  if(+row.round_index!==events.length+1)throw Error("missing/duplicate round_index in "+id);
  events.push(row);shoes.set(id,events);
}
if(shoes.size<5)throw Error("at least five complete shoes are required");
const ordered=[...shoes],holdout=ordered.slice(Math.floor(ordered.length*.8));
if(!holdout.length)throw Error("chronological holdout is empty");

const current=name=>fs.readFileSync(path.join(ROOT,name),"utf8");
const atBaseline=name=>cp.execFileSync("git",["show",`${BASELINE_COMMIT}:${name}`],{cwd:ROOT,encoding:"utf8"});
const inject=(source,evConsistency,particleMode)=>{
  const evLine=/const EV_CONSISTENCY_V1=(?:true|false);[^\n]*/;
  const particleLine=/const PARTICLE_BLEND_MODE="[^"]+";/;
  if(!evLine.test(source)||!particleLine.test(source))throw Error("runtime shadow injection anchors missing");
  return source.replace(evLine,`const EV_CONSISTENCY_V1=${evConsistency};`)
    .replace(particleLine,`const PARTICLE_BLEND_MODE="${particleMode}";`);
};
const sources={
  currentFinal:current("final_probability_runtime.js"),currentParticle:current("particle_filter_runtime.js"),currentMacro:current("macro_ema.js"),
  baselineFinal:atBaseline("final_probability_runtime.js"),baselineParticle:atBaseline("particle_filter_runtime.js"),baselineMacro:atBaseline("macro_ema.js")
};

async function runtime(config){
  const store=new Map(),quiet={log(){},warn(){},error(){}},ctx={console:quiet,
    localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    document:{getElementById:()=>null},fetch:async url=>({ok:true,status:200,json:async()=>JSON.parse(current(String(url)))})};
  ctx.window=ctx;vm.createContext(ctx);
  vm.runInContext(current("app256forward.js"),ctx,{filename:"app256forward.js"});
  vm.runInContext(current("app256continuation.js"),ctx,{filename:"app256continuation.js"});
  const particle=config.exactBaseline?sources.baselineParticle:sources.currentParticle;
  const macro=config.exactBaseline?sources.baselineMacro:sources.currentMacro;
  const final=config.exactBaseline?sources.baselineFinal:inject(sources.currentFinal,config.evConsistency,config.particleMode);
  vm.runInContext(particle,ctx,{filename:"particle_filter_runtime.js"});
  vm.runInContext(macro,ctx,{filename:"macro_ema.js"});
  vm.runInContext(final,ctx,{filename:"final_probability_runtime.js"});
  const api=ctx.__BGS_FINAL56__;await api.loadModels();
  if(!api.getModelStatus().physics||!api.getModelStatus().final56)throw Error("runtime model load failed");
  return {api,store};
}
function empty(){return {eligible:0,bets:0,non_tie_bets:0,tie_pushes:0,wins:0,losses:0,skips:0,net_units:0,brier_sum:0,brier_n:0};}
function summarize(m){return {...m,net_units:+m.net_units.toFixed(8),
  hit_rate:m.non_tie_bets?m.wins/m.non_tie_bets:null,skip_rate:m.eligible?m.skips/m.eligible:null,
  realized_ev_per_bet:m.bets?m.net_units/m.bets:null,conditional_direction_brier:m.brier_n?m.brier_sum/m.brier_n:null};}
async function replay(config){
  const {api,store}=await runtime(config),m=empty(),fingerprints=[],modeChecks={observations:0,violations:[]};
  for(const [shoe,events] of holdout){
    store.clear();store.set("bgs_xgb_final_shoe_id_v5",shoe);const history=[];
    for(let i=0;i<events.length;i++){
      const actual=String(events[i].outcome).toUpperCase();
      if(history.length){
        const prediction=api.applyFinalPrediction(history),r=prediction.finalProbability;
        if(r.mode!=="physics_primary"||r.error)throw Error(`${config.id} runtime failure ${shoe}/${i+1}: ${r.error}`);
        m.eligible++;const finalPB=+r.finalPB;
        if(actual!=="T"){m.brier_n++;m.brier_sum+=(finalPB-(actual==="B"?1:0))**2;}
        if(prediction.direction==="Skip")m.skips++;
        else{
          m.bets++;if(actual==="T")m.tie_pushes++;
          else{m.non_tie_bets++;if(prediction.direction===actual){m.wins++;m.net_units+=prediction.direction==="B"?.95:1;}else{m.losses++;m.net_units--;}}
        }
        const compact={shoe,round:i+1,direction:prediction.direction,finalPB,evB:prediction.ev_banker,evP:prediction.ev_player,
          candidate:r.primary_candidate,volume:r.volume_guard_active};
        fingerprints.push(compact);
        if(!config.exactBaseline){
          modeChecks.observations++;
          if(r.particle_blend_mode!==config.particleMode)modeChecks.violations.push(`${shoe}/${i+1}:mode`);
          if(config.particleMode==="off"&&r.execution_order.includes("particle_filter"))modeChecks.violations.push(`${shoe}/${i+1}:off_called_particle`);
          if(config.particleMode.startsWith("fixed_")){
            const want=Number(config.particleMode.slice(6));
            if(Math.abs((r.particleDiagnostics?.blend_weight??-1)-want)>1e-12)modeChecks.violations.push(`${shoe}/${i+1}:fixed_weight`);
          }
          if(config.particleMode==="uncertainty_only")for(const index of [23,24,25,39,40,41])
            if(Math.abs((r.physics?.[index]??NaN)-(r.physicsDirect?.[index]??NaN))>1e-12)modeChecks.violations.push(`${shoe}/${i+1}:direct_head_${index}`);
        }
        api.registerPrediction(history,prediction);api.settlePending(actual,events[i],history);
      }
      history.push(actual);api.recordCompletedHand(history,events[i]);
    }
  }
  return {id:config.id,metrics:summarize(m),fingerprints,mode_checks:modeChecks};
}
function numericEqual(a,b){return a===b||Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-12;}
function behavioralDiff(base,other){
  if(base.length!==other.length)return {equal:false,mismatches:["length"]};
  const mismatches=[];
  for(let i=0;i<base.length;i++)for(const key of ["shoe","round","direction","finalPB","evB","evP","candidate","volume"])
    if(typeof base[i][key]==="number"? !numericEqual(base[i][key],other[i][key]):base[i][key]!==other[i][key]){
      mismatches.push(`${base[i].shoe}/${base[i].round}:${key}`);break;
    }
  return {equal:mismatches.length===0,mismatch_count:mismatches.length,mismatches:mismatches.slice(0,20)};
}
function delta(base,candidate){
  const keys=["hit_rate","skip_rate","realized_ev_per_bet","conditional_direction_brier"];
  return Object.fromEntries(keys.map(k=>[k,candidate[k]===null||base[k]===null?null:candidate[k]-base[k]]));
}

(async()=>{
  const configs=[
    {id:"baseline_7777a321",exactBaseline:true},
    {id:"production_defaults",evConsistency:false,particleMode:"quality"},
    {id:"ev_consistency_v1",evConsistency:true,particleMode:"quality"},
    ...MODES.filter(x=>x!=="quality").map(particleMode=>({id:"particle_"+particleMode,evConsistency:false,particleMode}))
  ];
  const runs={};for(const config of configs){const result=await replay(config);runs[config.id]=result;}
  const base=runs.baseline_7777a321.metrics,evc=runs.ev_consistency_v1.metrics,evDelta=delta(base,evc);
  const gates={hit_rate:evc.hit_rate>=base.hit_rate,
    realized_ev_per_bet:evc.realized_ev_per_bet>=base.realized_ev_per_bet-.001,
    absolute_correct_bets:evc.wins>=.95*base.wins,
    skip_rate:evDelta.skip_rate>=-.05&&evDelta.skip_rate<=.02};
  const report={schema:"bbb_v14_8_safe_patch_shadow_v1",baseline_commit:BASELINE_COMMIT,
    source:payload.data_kind||"user_supplied_unverified",total_shoes:shoes.size,total_hands:rows.length,
    split:{method:"chronological shoes; latest 20% holdout only",holdout_shoes:holdout.length,holdout_ids:holdout.map(x=>x[0])},
    definitions:{hit_rate:"wins / non-tie placed bets",skip_rate:"skips / eligible hands",
      realized_ev_per_bet:"fixed-unit net / all placed bets; Banker +0.95, Player +1, loss -1, tie push 0",
      conditional_direction_brier:"Brier on non-tie outcomes using conditional Final P(B)",absolute_correct_bets:"wins"},
    production_default_equivalence:behavioralDiff(runs.baseline_7777a321.fingerprints,runs.production_defaults.fingerprints),
    ev_consistency:{baseline:base,candidate:evc,delta_vs_baseline:evDelta,gates,accepted:Object.values(gates).every(Boolean),
      recommended_flag:Object.values(gates).every(Boolean),current_checked_in_default:false},
    particle_ablation:Object.fromEntries(MODES.map(mode=>{
      const key=mode==="quality"?"production_defaults":"particle_"+mode,r=runs[key];
      return [mode,{metrics:r.metrics,delta_vs_quality:delta(runs.production_defaults.metrics,r.metrics),mode_checks:r.mode_checks}];
    })),
    warnings:["Shadow tool is never loaded by the production page.",synthetic?"Synthetic mechanics-only holdout cannot establish live predictive advantage or profitability.":"Input provenance is user supplied and not independently verified."]};
  fs.writeFileSync(output,JSON.stringify(report,null,2)+"\n");
  console.log(JSON.stringify({ok:true,holdout_shoes:holdout.length,production_default_equal:report.production_default_equivalence.equal,
    ev_consistency_accepted:report.ev_consistency.accepted,recommended_flag:report.ev_consistency.recommended_flag,
    particle_mode_violations:Object.fromEntries(MODES.map(mode=>[mode,report.particle_ablation[mode].mode_checks.violations.length]))}));
})().catch(error=>{console.error(error);process.exit(1);});
