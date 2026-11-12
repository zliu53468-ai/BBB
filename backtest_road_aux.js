"use strict";
/**
 * Offline, chronological, shoe-grouped research display comparison.
 * Input: JSON array of {"shoe_id":"...","history":["B","P","T",...]}
 * or {"shoes":[...]}; each shoe must contain 2..72 results.
 * Only results AFTER the current history prefix are scored; ties are pushes.
 * Outputs research coverage and conditional B/P hit rate. NO EV claims.
 *
 * Usage: node backtest_road_aux.js ./real_shoe_histories.json
 */
const fs=require("node:fs");
global.self=global;
const engine=require("./soft_shoe_decision_runtime.js");
require("./soft_shoe_model_bundle.js");
const sequence=require("./soft_shoe_frozen_lstm_runtime.js");
require("./soft_shoe_frozen_model_bundle.js");
const asHistory=record=>Array.isArray(record)?record:
 typeof record==="string"?record.split(""):
 Array.isArray(record?.history)?record.history:
 typeof record?.history==="string"?record.history.split(""):null;
const rate=(a,b)=>b?(a/b):null;
function aggregate(){
 return {signals:0,ties:0,scored:0,hits:0};
}
function select(m,next,pred){
 if(!["B","P"].includes(pred))return;
 m.signals++;
 if(next==="T"){m.ties++;return;}
 m.scored++;
 if(next===pred)m.hits++;
}
function describe(m,total){
 return {...m,coverage:rate(m.signals,total),
  hit_rate_excluding_ties:rate(m.hits,m.scored)};
}
function backtest(shoes){
 if(!Array.isArray(shoes)||!shoes.length)throw Error("expected_nonempty_shoe_array");
 const soft=aggregate(),confirmed=aggregate(),counts={shoes:0,decisions:0};
 for(const record of shoes){
  const history=asHistory(record);
  if(!history||history.length<2||history.length>72||
     history.some(x=>!["B","P","T"].includes(x)))
   throw Error("each shoe must have 2..72 B/P/T hands");
  counts.shoes++;
  // Never score the first hand with zero prior data.
  for(let i=1;i<history.length;i++){
   const observedPrefix=history.slice(0,i);
   const outcome=history[i];
   const report=engine.run(observedPrefix,{},global.BBB_SOFT_SHOE_XGB_MODEL,
    sequence,global.BBB_SOFT_SHOE_LSTM_MODEL);
   const v=report.research_view||{};
   counts.decisions++;
   if(v.eligible===true&&["B","P"].includes(v.direction)){
    select(soft,outcome,v.direction);
    if(v.road_confirmed===true)select(confirmed,outcome,v.direction);
   }
   if(report.action!=="Skip"||report.bankroll_fraction!==0||
      report.validated_edge!==false)
    throw Error("unverified_formal_gate_changed");
  }
 }
 const a=describe(soft,counts.decisions),b=describe(confirmed,counts.decisions);
 return {research_only:true,estimated_only:true,validated_betting_edge:false,
  selection:"soft-only research signals versus same signals with road confirmation",
  ...counts,soft_only:a,soft_and_road_confirmed:b,
  selective_hit_rate_change_pp:
   a.hit_rate_excluding_ties===null||b.hit_rate_excluding_ties===null?
   null:100*(b.hit_rate_excluding_ties-a.hit_rate_excluding_ties),
  note:"No holdout validation supplied; no proof of accuracy improvement or positive EV."};
}
if(require.main===module){
 try{
  const input=process.argv[2];
  if(!input)throw Error("usage: node backtest_road_aux.js shoes.json");
  const data=JSON.parse(fs.readFileSync(input,"utf8"));
  process.stdout.write(JSON.stringify(backtest(Array.isArray(data)?data:data.shoes),null,2)+"\n");
 }catch(e){process.stderr.write(String(e.message||e)+"\n");process.exitCode=1;}
}
module.exports={backtest};
