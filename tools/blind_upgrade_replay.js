"use strict";
// Chronological B/P/T-only comparison. No fitting or automatic model promotion.
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,".."),args=process.argv.slice(2),synthetic=args[0]==="--synthetic";
const output=synthetic?args[2]:args[1];
if(!output)throw Error("Usage: node tools/blind_upgrade_replay.js hands.json report.json | --synthetic 5 report.json");
const payload=synthetic?{rows:require(path.join(root,"calibrate_macro_direction.js")).synthetic(Number(args[1]))}:JSON.parse(fs.readFileSync(args[0],"utf8"));
const rows=Array.isArray(payload)?payload:payload.rows,shoes=new Map(),closed=new Set();let last=null;
if(!Array.isArray(rows))throw Error("rows required");
for(const row of rows){
  const id=String(row.shoe_id??"");if(!id||!["B","P","T"].includes(row.outcome))throw Error("shoe_id and B/P/T required");
  if(id!==last){if(closed.has(id))throw Error("interleaved shoes");if(last!==null)closed.add(last);last=id;}
  const events=shoes.get(id)||[];
  if(row.round_index!==events.length+1)throw Error("non-contiguous chronology");
  // Deliberately strip scores, card counts, and all future fields.
  events.push({outcome:row.outcome});shoes.set(id,events);
}
if(shoes.size<5)throw Error("at least five chronological shoes required");
const holdout=[...shoes].slice(Math.floor(shoes.size*.8));
async function runtime(mode){
  const store=new Map(),ctx={console,BBB_BLIND_UPGRADE_MODE:mode,document:{getElementById:()=>null},
    localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    fetch:async n=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(root,n),"utf8"))})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const file of ["app256forward.js","app256continuation.js","blind_upgrade_runtime.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"])
    vm.runInContext(fs.readFileSync(path.join(root,file),"utf8"),ctx,{filename:file});
  await ctx.__BGS_FINAL56__.loadModels();return {store,api:ctx.__BGS_FINAL56__};
}
async function replay(mode){
  const {store,api}=await runtime(mode),m={predictions:0,bets:0,wins:0,losses:0,ties:0,skips:0,net_units:0,brier_sum:0,brier_n:0,max_losing_run:0,no_flip_violations:0};
  const stages={};let maxMs=0;
  for(const [id,events] of holdout){
    store.clear();const history=[];let losses=0;
    for(const actual of events){
      if(history.length){
        const start=Date.now(),p=api.applyFinalPrediction(history),r=p.finalProbability;maxMs=Math.max(maxMs,Date.now()-start);
        if(r.error||r.mode!=="physics_primary")throw Error(mode+" runtime error: "+r.error);
        m.predictions++;const stage=history.length<=15?"1_15":history.length>50?"51_plus":"16_50";
        const st=stages[stage]??={predictions:0,skips:0,bets:0,wins:0,losses:0};st.predictions++;
        if(p.direction!=="Skip"&&p.direction!==r.primary_candidate)m.no_flip_violations++;
        if(actual.outcome!=="T"){m.brier_n++;m.brier_sum+=(r.finalPB-Number(actual.outcome==="B"))**2;}
        if(p.direction==="Skip"){m.skips++;st.skips++;}
        else{
          m.bets++;st.bets++;
          if(actual.outcome==="T")m.ties++;
          else if(p.direction===actual.outcome){m.wins++;st.wins++;m.net_units+=p.direction==="B"?.95:1;losses=0;}
          else{m.losses++;st.losses++;m.net_units--;m.max_losing_run=Math.max(m.max_losing_run,++losses);}
        }
        api.registerPrediction(history,p);api.settlePending(actual.outcome,actual,history);
      }
      history.push(actual.outcome);api.recordCompletedHand(history,actual);
    }
    console.log(mode+": finished "+id+" ("+events.length+" hands)");
  }
  return {...m,skip_rate:m.skips/m.predictions,hit_rate:m.wins+m.losses?m.wins/(m.wins+m.losses):null,
    realized_ev_per_bet:m.bets?m.net_units/m.bets:null,brier:m.brier_n?m.brier_sum/m.brier_n:null,max_prediction_ms:maxMs,stages};
}
(async()=>{
  const baseline=await replay("baseline"),candidate=await replay("guarded");
  const report={data_kind:synthetic?"synthetic_mechanics_only":"user_supplied_unverified",total_shoes:shoes.size,holdout_shoes:holdout.length,
    split:"last 20% of chronological shoes; no parameter selection",inputs:"B/P/T only",baseline,candidate,
    delta:{skip_rate:candidate.skip_rate-baseline.skip_rate,hit_rate:candidate.hit_rate===null||baseline.hit_rate===null?null:candidate.hit_rate-baseline.hit_rate},
    performance_claim_validated:false,auto_promote:false,note:"Smoke comparison only; simulated or unverified outcomes cannot establish casino advantage or simultaneous hit-rate/coverage gains."};
  fs.writeFileSync(output,JSON.stringify(report,null,2));console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exitCode=1;});
