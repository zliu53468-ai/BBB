#!/usr/bin/env node
"use strict";
// Integration regression of real Frozen Core result handlers + V15 telemetry
// adapter. No production model topology, weights or Frozen Core code is altered.
const assert=require("node:assert/strict");
const storage=new Map();
const HISTORY="bgs256d_frozen_6x15_forward_v18";
let failOneHistoryWrite=false,failAllHistoryWrites=false;
global.window=global;
global.localStorage={
  getItem:k=>storage.get(k)??null,
  setItem:(k,v)=>{
    if(k===HISTORY&&(failAllHistoryWrites||failOneHistoryWrite)){
      failOneHistoryWrite=false;throw Error("QuotaExceededError: simulated");
    }
    storage.set(k,String(v));
  },
  removeItem:k=>storage.delete(k)
};
const baseHistory="BPPBTBBPPTBBPPBTBPBPPBP".split(""); // 23 completed hands
assert.equal(baseHistory.length,23);
storage.set(HISTORY,JSON.stringify({history:baseHistory,last_selected:"",selection_streak:0}));
class Element{
  constructor(id){this.id=id;this.listeners=[];this.value="";this.textContent="";this.innerHTML="";this.hidden=false;this.className="";this.dataset={};
    this.classList={add:()=>{},remove:()=>{},toggle:()=>{}};}
  setAttribute(name,value){this[name]=value;}
  addEventListener(type,callback,capture=false){this.listeners.push({type,callback,capture:capture===true});}
  cloneNode(){return new Element(this.id);}
  replaceWith(next){elements.set(this.id,next);}
  click(){
    const event={defaultPrevented:false,immediateStopped:false,preventDefault(){this.defaultPrevented=true;},
      stopImmediatePropagation(){this.immediateStopped=true;}};
    const handlers=[...this.listeners].filter(x=>x.type==="click").sort((a,b)=>Number(b.capture)-Number(a.capture));
    for(const handler of handlers){if(event.immediateStopped)break;handler.callback(event);}
  }
}
const ids=["btnB","btnP","btnT","btnStart","btnNoSkipTest","forecastGap","btnBack","btnEnd","btnExportTraining","btnExportObservations",
 "roundCount","roundPill","modePill","message","historyTrack","directionText","directionCode","confidence",
 "regime","strength","directionOrb","localCalibrationStatus","macroCardCount","macroPlayerScore","macroBankerScore"];
const elements=new Map(ids.map(id=>[id,new Element(id)]));
global.document={getElementById:id=>elements.get(id)||null,createElement:()=>new Element("temp"),
  body:{appendChild:()=>{}}};
global.fetch=async()=>({ok:false,status:404,json:async()=>({})});
global.URL={createObjectURL:()=>"",revokeObjectURL:()=>{}};
global.Blob=function(){};
require("./app256forward.js");
require("./app256continuation.js");

require("./particle_filter_runtime.js");
require("./macro_ema.js");
require("./browser_calibration.js");
require("./final_probability_runtime.js");
const el=id=>elements.get(id),stored=()=>JSON.parse(storage.get(HISTORY)).history;
assert.equal(el("roundCount").textContent,23,"Frozen Core initially loaded 23 results");
assert.deepEqual(stored(),baseHistory);
const report=(id)=>el(id).click();


for(const id of ["macroCardCount","macroPlayerScore","macroBankerScore"])elements.delete(id);
for(const outcome of ["B","P","T"]){report("btn"+outcome);assert.equal(stored().at(-1),outcome);}
assert.equal(stored().length,26);
(async()=>{await global.__BGS_FINAL56__.loadModels();report("btnStart");assert.match(el("message").textContent,/分析完成/);report("btnBack");assert.equal(stored().length,25);
report("btnNoSkipTest");assert.equal(el("btnNoSkipTest")["aria-pressed"],"true");
assert.equal(el("directionText").textContent,"未就緒");
assert.match(el("message").textContent,/模型未就緒/);
const fs=require("node:fs");global.fetch=async file=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(file,"utf8"))});
await global.__BGS_FINAL56__.loadModels();
const api=global.__BGS_FINAL56__,tie={available:true,directional:{B:.5,P:.5}};
assert.equal(api.noSkipTestDirection(tie),"B");assert.equal(api.noSkipTestDirection({available:false}),null);
assert.equal(api.noSkipTestDirection({available:true,directional:{B:.49,P:.51}}),"P");
report("btnNoSkipTest"); // Back to moderate display for threshold control checks.
const sampleForecast={available:true,direction:"B",directional:{B:.51,P:.49},probabilities:{B:.459,P:.441,T:.1}};
const changeGap=value=>{el("forecastGap").value=value;for(const h of el("forecastGap").listeners.filter(x=>x.type==="change"))h.callback();};
changeGap("0.03");api.renderPrediction({next_hand_forecast:sampleForecast,direction:"Skip"},stored().length);
assert.equal(el("directionText").textContent,"觀望");
changeGap("0.01");api.renderPrediction({next_hand_forecast:sampleForecast,direction:"Skip"},stored().length);
assert.equal(el("directionText").textContent,"莊");
changeGap("0.02");report("btnNoSkipTest");
const rowCount=api.getTrainingCount();
for(const side of ["B","P","T","P"]){
 report("btn"+side);
 const expected=api.noSkipTestDirection(api.applyFinalPrediction(stored()).next_hand_forecast);
 assert.equal(el("directionText").textContent,expected==="B"?"莊":"閒");
 assert.match(el("regime").textContent,/不列入正式下注/);
 assert.equal(storage.has("bgs_xgb_final_pending_v5"),false,"test must not register bets");
}
assert.equal(api.getTrainingCount(),rowCount,"test must not add wager training rows");
assert.equal(api.getLocalCalibrationStatus().seen,0,"independent mode does not train a local journal");
assert.equal(api.getLocalCalibrationStatus().directional_samples,0,"independent mode does not reuse directional calibration");
assert.match(el("localCalibrationStatus").textContent,/不是重訓/);
const localBefore=JSON.stringify(api.getLocalCalibrationStatus());
api.applyFinalPrediction(stored());api.applyFinalPrediction(stored());
assert.equal(JSON.stringify(api.getLocalCalibrationStatus()),localBefore,"re-analysis must not train again");
report("btnBack");assert.equal(Number(el("roundCount").textContent),stored().length);assert.equal(api.getLocalCalibrationStatus().directional_samples,0);
report("btnNoSkipTest");assert.equal(el("btnNoSkipTest")["aria-pressed"],"false");
report("btnP");assert.equal(api.getLocalCalibrationStatus().directional_samples,0,"independent mode stays uncalibrated in normal display");
assert.equal(Number(el("roundCount").textContent),stored().length);
report("btnNoSkipTest");report("btnStart");assert.equal(el("btnNoSkipTest")["aria-pressed"],"false");
report("btnNoSkipTest");report("btnEnd");assert.equal(el("btnNoSkipTest")["aria-pressed"],"false");
assert.equal(storage.has("bgs_browser_calibration_v1"),false,"end clears local calibration");
console.log("continuous test mode, no wager registration, undo, stop and unavailable model passed");
console.log("button-only UI: B/P/T, analysis and undo without score fields passed");})().catch(e=>{console.error(e);process.exitCode=1;});
