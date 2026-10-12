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
require("./particle_filter_runtime.js");
require("./macro_ema.js");
require("./final_probability_runtime.js");
const el=id=>elements.get(id),stored=()=>JSON.parse(storage.get(HISTORY)).history;
assert.equal(el("roundCount").textContent,23,"Frozen Core initially loaded 23 results");
assert.deepEqual(stored(),baseHistory);
const report=(id)=>el(id).click();

// Invalid optional points must NOT block the authoritative Banker result.
el("macroCardCount").value="6";
el("macroPlayerScore").value="9";
el("macroBankerScore").value="1";
report("btnB");
assert.equal(stored().length,24);
assert.equal(stored().at(-1),"B");
assert.equal(el("roundCount").textContent,24);
assert.match(el("message").textContent,/已記錄第 24 局.*已忽略/);
assert.equal(el("macroBankerScore").value,"","optional fields cleared");

// Valid optional points must still be retained in the observational archive.
el("macroCardCount").value="4";
el("macroPlayerScore").value="9";
el("macroBankerScore").value="0";
report("btnP");
assert.equal(stored().length,25);
assert.equal(stored().at(-1),"P");
assert.equal(el("roundCount").textContent,25);
const api=global.__BGS_FINAL56__;
let archived=JSON.parse(api.exportMacroObservations()).rows;
assert.equal(archived.at(-1).outcome,"P");
assert.equal(archived.at(-1).player_score,9);
assert.equal(archived.at(-1).banker_score,0);
report("btnT");
assert.equal(stored().length,26);
assert.equal(stored().at(-1),"T");
assert.equal(el("roundCount").textContent,26);

// Frozen Core save silently catches quota errors. UI adapter must retry/verify.
failOneHistoryWrite=true;
report("btnP");
assert.equal(stored().length,27);
assert.equal(stored().at(-1),"P");
assert.equal(el("roundCount").textContent,27);

// Persistent failure must be visible rather than falsely reporting success.
failAllHistoryWrites=true;
report("btnB");
assert.equal(stored().length,27);
assert.equal(el("roundCount").textContent,28,"Frozen Core in-memory state continued");
assert.match(el("message").textContent,/本局儲存失敗/);
failAllHistoryWrites=false;
// Resimulate browser reload (rather than accepting unsaved in-memory state).
delete require.cache[require.resolve("./app256forward.js")];
require("./app256forward.js");
assert.equal(el("roundCount").textContent,27,"reload uses persisted 27, does not pretend the failed 28th hand was saved");
report("btnB");
assert.equal(stored().length,28);
assert.equal(el("roundCount").textContent,28);
assert.equal(stored().at(-1),"B");
// Frozen Core back handler still works without prediction or architecture changes.
report("btnBack");
assert.equal(stored().length,27);
assert.equal(el("roundCount").textContent,27);
// Pressing Start should NOT add a completed-hand result: display next round only.
console.log(JSON.stringify({ok:true,checks:["invalid_optional_does_not_block","valid_observation","banker_player_tie","silent_storage_write_retry",
  "persistent_storage_failure_warns","reload_restores_last_saved_history","backwards_compatible_core"],final_round:stored().length}));
