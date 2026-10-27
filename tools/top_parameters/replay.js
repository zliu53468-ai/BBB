/* Formal causal replay. Optional --local-calibration learns only after each actual result. */
"use strict";
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const [directory,input,output]=process.argv.slice(2);
if(!directory||!input||!output)throw Error('node replay.js runtime-directory complete-hands.json output.json');
const learnLocal=process.argv.includes("--local-calibration");
const dir=path.resolve(directory),payload=JSON.parse(fs.readFileSync(input,'utf8')),rows=Array.isArray(payload)?payload:payload.rows;
if(!Array.isArray(rows)||!rows.length)throw Error('Nonempty completed hands required');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const store=new Map(),ctx={console,localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},document:{getElementById:()=>null},fetch:async url=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(dir,String(url).split('?')[0]),'utf8'))})};
ctx.window=ctx;vm.createContext(ctx);
for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(path.join(dir,f),'utf8'),ctx,{filename:f});
(async()=>{
 const api=ctx.__BGS_FINAL56__,M=ctx.__BGS_MACRO_EMA__;await api.loadModels();
 const outputRows=[],seen=new Set();let shoe=null,history=[];
 for(const row of rows){
  const id=String(row.shoe_id??'');if(!id)throw Error('shoe_id required');
  if(id!==shoe){if(seen.has(id))throw Error('Interleaved shoe');seen.add(id);shoe=id;history=[];store.clear();store.set('bgs_xgb_final_shoe_id_v5',shoe);}
  if(row.round_index!==history.length+1)throw Error('Need consecutive complete hands including ties');
  // Do not reveal current outcome or optional measurements to runtime before prediction.
  let prediction=null,ticket=null;
  if(history.length){prediction=api.applyFinalPrediction(history);const r=prediction.finalProbability;
   if(r.error||r.mode!=='physics_primary'||!Number.isFinite(r.finalPB))throw Error('Runtime failed: '+JSON.stringify(r));
   if(!['B','P','Skip'].includes(prediction.direction))throw Error('Invalid direction');
   if(learnLocal)ticket=api.prepareCalibrationTicket(history,prediction);
   api.registerPrediction(history,prediction);
  }
  const observed=M.normalizeHand(row);
  if(prediction){const r=prediction.finalProbability;
   outputRows.push({shoe_id:shoe,round_index:row.round_index,outcome:observed.outcome,p_b:r.finalPB,uncalibrated_p_b:prediction.calibration_input_forecast?.directional?.B??r.finalPB,browser_calibration:prediction.browser_calibration,direction:prediction.direction,ev_direction:prediction.ev_direction,direction_selection:r.direction_selection,direction_candidate:r.direction_candidate,forecast_direction:prediction.forecast_direction,draw_report:r.draw_report,settlement_audit:r.settlement_audit,observed_hand:observed});
   api.settlePending(observed.outcome,observed,history);
  }
  history.push(observed.outcome);if(ticket)api.commitCalibrationTicket(ticket,observed.outcome,history);api.recordCompletedHand(history,observed);
 }
 const files=['app256forward.js','app256continuation.js','particle_filter_runtime.js','final_probability_runtime.js','macro_ema.js','cardcount_profile.json','physics_multitask_model.json','final_probability_model.json'];
 fs.writeFileSync(output,JSON.stringify({schema_version:1,data_kind:payload.data_kind||'unspecified',input_sha256:sha(input),runtime_sha256:Object.fromEntries(files.map(f=>[f,sha(path.join(dir,f))])),local_calibration_learning:learnLocal,lifecycle:'formal_analysis_every_eligible_hand',first_hand_excluded:true,rows:outputRows},null,2)+'\n');
 console.log(`${outputRows.length} causal predictions -> ${output}`);
})().catch(e=>{console.error(e);process.exit(1)});
