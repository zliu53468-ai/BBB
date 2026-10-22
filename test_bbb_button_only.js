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
const ids=["btnB","btnP","btnT","btnStart","btnBack","btnEnd","btnExportTraining","btnExportObservations",
 "roundCount","roundPill","modePill","message","historyTrack","directionText","directionCode","confidence",
 "regime","strength","directionOrb","macroCardCount","macroPlayerScore","macroBankerScore"];
const elements=new Map(ids.map(id=>[id,new Element(id)]));
global.document={getElementById:id=>elements.get(id)||null,createElement:()=>new Element("temp"),
  body:{appendChild:()=>{}}};
global.fetch=async()=>({ok:false,status:404,json:async()=>({})});
global.URL={createObjectURL:()=>"",revokeObjectURL:()=>{}};
global.Blob=function(){};
require("./app256forward.js");
require("./app256continuation.js");

require("./macro_ema.js");
require("./final_probability_runtime.js");
const el=id=>elements.get(id),stored=()=>JSON.parse(storage.get(HISTORY)).history;
assert.equal(el("roundCount").textContent,23,"Frozen Core initially loaded 23 results");
assert.deepEqual(stored(),baseHistory);
const report=(id)=>el(id).click();


for(const id of ["macroCardCount","macroPlayerScore","macroBankerScore"])elements.delete(id);
for(const outcome of ["B","P","T"]){report("btn"+outcome);assert.equal(stored().at(-1),outcome);}
assert.equal(stored().length,26);
(async()=>{await global.__BGS_FINAL56__.loadModels();report("btnStart");assert.match(el("message").textContent,/分析完成/);report("btnBack");assert.equal(stored().length,25);console.log("button-only UI: B/P/T, analysis and undo without score fields passed");})().catch(e=>{console.error(e);process.exitCode=1;});
