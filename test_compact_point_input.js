"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const PHASE_KEY="bgs_score_entry_phase_v2";
class Element{
  constructor(id){
    this.id=id;this.listeners={};this.value="";this.textContent="";this.hidden=false;
    this.classList={toggle:()=>{},add:()=>{},remove:()=>{}};
  }
  addEventListener(event,fn){(this.listeners[event]||=[]).push(fn);}
  click(){for(const fn of this.listeners.click||[])fn({});}
  change(){for(const fn of this.listeners.change||[])fn({});}
}
const ids=["resultModePanel","pointModePanel","entryStageTitle","entryStageNote",
  "macroBankerScore","macroPlayerScore","pointResultPreview","btnStart",
  "btnB","btnP","btnT","btnBack","btnEnd"];
function make(initial=[],savedPhase="history",denyWrite=false){
  const els=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
  const data=new Map([[HISTORY_KEY,JSON.stringify({history:initial})]]);
  if(savedPhase==="points")data.set(PHASE_KEY,"points");
  const storage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,String(v)),
    removeItem:k=>data.delete(k)};
  let predictions=0;
  const observed=[];
  els.btnStart.addEventListener("click",()=>{if(JSON.parse(storage.getItem(HISTORY_KEY)).history.length)predictions++;});
  for(const outcome of ["B","P","T"]){
    els["btn"+outcome].addEventListener("click",()=>{
      observed.push({outcome,b:els.macroBankerScore.value,p:els.macroPlayerScore.value});
      if(denyWrite)return;
      const entry=JSON.parse(storage.getItem(HISTORY_KEY));
      entry.history.push(outcome);
      storage.setItem(HISTORY_KEY,JSON.stringify(entry));
      // The production postclick runtime clears these scores after logging.
      els.macroBankerScore.value="";els.macroPlayerScore.value="";
    });
  }
  els.btnBack.addEventListener("click",()=>{
    const entry=JSON.parse(storage.getItem(HISTORY_KEY));entry.history.pop();
    storage.setItem(HISTORY_KEY,JSON.stringify(entry));
  });
  els.btnEnd.addEventListener("click",()=>{
    storage.removeItem(HISTORY_KEY);storage.removeItem(PHASE_KEY);
  });
  const context={document:{getElementById:id=>els[id]||null},localStorage:storage,
    Promise,console};
  context.window=context;
  if(savedPhase==="points")context.__BGS_FINAL56__={loadModels:async()=>({physics:true,final56:true})};
  vm.createContext(context);vm.runInContext(fs.readFileSync("point_input_mode.js","utf8"),context);
  return {api:context.__BBB_POINT_INPUT__,els,data,storage,observed,
    get predictions(){return predictions;},
    choose(b,p){
      els.macroBankerScore.value=b;els.macroBankerScore.change();
      els.macroPlayerScore.value=p;els.macroPlayerScore.change();
    }};
}
(async()=>{
  const a=make();
  assert.equal(a.api.getPhase(),"history");
  assert.equal(a.els.resultModePanel.hidden,false);
  assert.equal(a.els.pointModePanel.hidden,true);
  assert.equal(a.api.deriveOutcome("0","0"),"T");
  assert.equal(a.api.deriveOutcome("9","0"),"B");
  assert.equal(a.api.deriveOutcome("0","9"),"P");
  for(const value of ["","-1","10","4.5","NaN"," "])
    assert.equal(a.api.deriveOutcome(value,"5"),null);
  a.els.btnStart.click(); // No historical observations -> no change
  assert.equal(a.api.getPhase(),"history");
  assert.equal(a.predictions,0);
  for(const outcome of ["B","P","T"]){
    a.els["btn"+outcome].click();
  }
  assert.equal(a.api.readHistory().join(""),"BPT");
  assert.equal(a.predictions,0,"prefilling history must not forecast repeatedly");
  a.els.btnStart.click();
  assert.equal(a.api.getPhase(),"points");
  assert.equal(a.predictions,1);
  assert.equal(a.els.resultModePanel.hidden,true);
  assert.equal(a.els.pointModePanel.hidden,false);
  assert.equal(a.data.get(PHASE_KEY),"points");
  const n=a.observed.length;
  a.els.macroBankerScore.value="7";a.els.macroBankerScore.change();
  assert.equal(a.api.readHistory().length,3,"one score must not auto-commit");
  assert.equal(a.observed.length,n);
  a.els.macroPlayerScore.value="2";a.els.macroPlayerScore.change();
  assert.equal(a.api.readHistory().join(""),"BPTB");
  assert.equal(a.predictions,2,"new prediction must be automatic");
  assert.deepEqual(a.observed.at(-1),{outcome:"B",b:"7",p:"2"});
  assert.equal(a.els.macroBankerScore.value,"");
  assert.equal(a.els.macroPlayerScore.value,"");
  a.els.macroPlayerScore.change();
  assert.equal(a.api.readHistory().length,4,"cleared score must prevent double write");
  a.choose("3","9");a.choose("4","4");
  assert.equal(a.api.readHistory().join(""),"BPTBPT");
  assert.equal(a.predictions,4);
  assert.deepEqual(a.observed.slice(-2),[{outcome:"P",b:"3",p:"9"},{outcome:"T",b:"4",p:"4"}]);
  a.els.btnBack.click();
  assert.equal(a.api.getPhase(),"points","undo must keep point mode after analysis starts");
  assert.equal(a.api.readHistory().join(""),"BPTBP");
  assert.equal(a.els.macroBankerScore.value,"");
  a.els.btnEnd.click();
  assert.equal(a.api.getPhase(),"history");
  assert.equal(a.data.has(PHASE_KEY),false);

  const failed=make(["B"],"history",true);
  failed.els.btnStart.click();
  failed.choose("6","2");
  assert.equal(failed.api.readHistory().join(""),"B","failed persistence cannot be reported as committed");
  assert.equal(failed.predictions,1,"failed persistence cannot trigger next prediction");
  assert.match(failed.els.pointResultPreview.textContent,/未確認寫入成功/);

  const forecastError=make(["B"]);
  forecastError.els.btnStart.click();
  forecastError.els.btnStart.addEventListener("click",()=>{throw Error("synthetic next-forecast failure");});
  forecastError.choose("7","4");
  assert.equal(forecastError.api.readHistory().join(""),"BB");
  assert.match(forecastError.els.pointResultPreview.textContent,/已自動記錄第 2 局/);
  assert.match(forecastError.els.pointResultPreview.textContent,/下一局分析未完成/);
  assert.equal(forecastError.els.macroBankerScore.value,"","committed scores must be cleared even if next forecast fails");

  const restored=make(["B","P"],"points");
  assert.equal(restored.api.getPhase(),"points");
  await new Promise(resolve=>setImmediate(resolve)); // await asynchronous model-restore callback
  assert.equal(restored.predictions,1,"reload should restore active analysis once");
  assert.equal(restored.api.readHistory().join(""),"BP");
  restored.choose("8","1");
  assert.equal(restored.api.readHistory().join(""),"BPB");

  const html=fs.readFileSync("index.html","utf8");
  const css=fs.readFileSync("styles.css","utf8");
  const auth=fs.readFileSync("auth-client.js","utf8");
  for(const id of ids)assert(html.includes('id="'+id+'"'),"missing element "+id);
  assert(!html.includes('id="btnRecordPoints"'),"no manual score confirmation allowed");
  assert(!html.includes('id="modePoints"'),"history-first is automatic, not a mode toggle");
  assert(css.includes("grid-template-columns:repeat(2,minmax(0,1fr))"));
  assert(auth.includes('"point_input_mode.js?v=20261027-history-auto-score-v2"'));
  assert(auth.indexOf('"point_input_mode.js?v=20261027-history-auto-score-v2"')>
         auth.indexOf('"final_probability_runtime.js?'));
  assert(!html.includes('<script src="point_input_mode.js"'),"must run after successful license verification");
  console.log(JSON.stringify({passed:true,bootstrap_history:true,start_switch:true,
    auto_win_and_next_forecast:true,duplicate_guard:true,undo_and_end:true,reload:true,
    post_license_only:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
