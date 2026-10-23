/* Prepare causal Core + Original 6D rows; Python rebuilds candidate Direct Physics. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const [directory,input,output]=process.argv.slice(2);
if(!output)throw Error("node build_core_rows.js runtime-dir complete-hands.json output.json");
const dir=path.resolve(directory),payload=JSON.parse(fs.readFileSync(input,"utf8")),rows=payload.rows;
if(!Array.isArray(rows)||!rows.length)throw Error("Complete observations required");
const store=new Map(),ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async url=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(dir,String(url).split("?")[0]),"utf8"))})};
ctx.window=ctx;vm.createContext(ctx);
for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(path.join(dir,f),"utf8"),ctx,{filename:f});
(async()=>{
 const api=ctx.__BGS_FINAL56__,core=ctx.__BGS256_CONTINUATION_TEST__;await api.loadModels();
 let shoe=null,history=[];const seen=new Set(),out=[];
 for(const row of rows){
  const id=String(row.shoe_id??"");if(!id)throw Error("shoe_id required");
  if(id!==shoe){if(seen.has(id))throw Error("Interleaved shoes");seen.add(id);shoe=id;history=[];store.clear();store.set("bgs_xgb_final_shoe_id_v5",id);}
  if(row.round_index!==history.length+1)throw Error("Missing observation");
  // Compute the feature prefix before even reading the current label/measurements.
  let original=null;if(history.length){const prediction=core.hazardChoose(history);original=api.buildOriginal7(history,prediction);}
  const observed=ctx.__BGS_MACRO_EMA__.normalizeHand(row);
  if(original&&observed.outcome!=="T")out.push({shoe_id:shoe,history:history.join(""),...original,actual_outcome:observed.outcome});
  history.push(observed.outcome);
 }
 fs.writeFileSync(output,JSON.stringify({schema_version:1,data_kind:payload.data_kind,feature_timing:"before_current_outcome",physics_source:"must_rebuild_using_candidate_joblib",rows:out})+"\n");
 console.log(`${out.length} causal binary rows; no supplied Physics/snapshot overrides`);
})().catch(e=>{console.error(e);process.exit(1)});
