"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const PHASE_KEY="bgs_score_entry_phase_v2";
const INPUT_MODE_KEY="bgs_score_input_mode_v1";
class Element{
  constructor(id){this.id=id;this.listeners={};this.value="";this.textContent="";this.hidden=false;this.attributes={};
    this.classList={toggle:()=>{},add:()=>{},remove:()=>{}};}
  addEventListener(event,fn){(this.listeners[event]||=[]).push(fn);}
  setAttribute(name,value){this.attributes[name]=value;}
  click(){for(const fn of this.listeners.click||[])fn({});}
  change(){for(const fn of this.listeners.change||[])fn({});}
}
const ids=["resultModePanel","pointModePanel","entryStageTitle","entryStageNote",
  "macroCardCount","macroBankerScore","macroPlayerScore","pointResultPreview","btnStart",
  "compatModeToggle","compatModeHint","btnB","btnP","btnT","btnBack","btnEnd"];
function make(initial=[],savedPhase="history",denyWrite=false){
  const els=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
  const data=new Map([[HISTORY_KEY,JSON.stringify({history:initial})]]);
  if(savedPhase==="points")data.set(PHASE_KEY,"points");
  const storage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,String(v)),removeItem:k=>data.delete(k)};
  let predictions=0;
  const observed=[];
  els.btnStart.addEventListener("click",()=>{if(JSON.parse(storage.getItem(HISTORY_KEY)).history.length)predictions++;});
  for(const outcome of ["B","P","T"]){
    els["btn"+outcome].addEventListener("click",()=>{
      observed.push({outcome,b:els.macroBankerScore.value,p:els.macroPlayerScore.value,c:els.macroCardCount.value});
      if(denyWrite)return;
      const entry=JSON.parse(storage.getItem(HISTORY_KEY));entry.history.push(outcome);
      storage.setItem(HISTORY_KEY,JSON.stringify(entry));
      els.macroBankerScore.value="";els.macroPlayerScore.value="";els.macroCardCount.value="";
    });
  }
  els.btnBack.addEventListener("click",()=>{const entry=JSON.parse(storage.getItem(HISTORY_KEY));entry.history.pop();storage.setItem(HISTORY_KEY,JSON.stringify(entry));});
  els.btnEnd.addEventListener("click",()=>{storage.removeItem(HISTORY_KEY);storage.removeItem(PHASE_KEY);});
  const context={document:{getElementById:id=>els[id]||null},localStorage:storage,Promise,console};context.window=context;
  context.__BGS_PARTICLE500__={validatePointObservation:()=>({valid:true})};
  if(savedPhase==="points")context.__BGS_FINAL56__={loadModels:async()=>({physics:true,final56:true})};
  vm.createContext(context);vm.runInContext(fs.readFileSync("point_input_mode.js","utf8"),context);
  return {api:context.__BBB_POINT_INPUT__,els,data,storage,observed,get predictions(){return predictions;},
    choose(b,p,c=""){
      els.macroCardCount.value=c;els.macroCardCount.change();
      els.macroBankerScore.value=b;els.macroBankerScore.change();
      els.macroPlayerScore.value=p;els.macroPlayerScore.change();
    }};
}
(async()=>{
  const a=make();
  assert.equal(a.api.getPhase(),"points");
  assert.equal(a.api.getInputMode(),"points","point input is the default mode");
  assert.equal(a.els.resultModePanel.hidden,true,"B/P/T buttons are hidden by default");
  assert.equal(a.els.pointModePanel.hidden,false,"score fields are ready without a history seed step");
  assert.equal(a.api.deriveOutcome("0","0"),"T");
  assert.equal(a.api.deriveOutcome("9","0"),"B");
  assert.equal(a.api.deriveOutcome("0","9"),"P");
  for(const value of ["","-1","10","4.5","NaN"," "])assert.equal(a.api.deriveOutcome(value,"5"),null);
  a.els.btnStart.click();
  assert.equal(a.api.getPhase(),"points","empty shoe starts directly in point entry");assert.equal(a.predictions,0);
  assert.equal(a.els.pointModePanel.hidden,false);
  const fresh=make();fresh.choose("9","0","4");
  assert.equal(fresh.api.readHistory().join(""),"B");assert.equal(fresh.predictions,1,"first scored hand starts next-hand analysis");
  assert.deepEqual(fresh.observed.at(-1),{outcome:"B",b:"9",p:"0",c:"4"});
  const freshCompat=make();freshCompat.els.compatModeToggle.click();
  assert.equal(freshCompat.api.getPhase(),"points");assert.equal(freshCompat.els.resultModePanel.hidden,false);
  freshCompat.els.btnT.click();assert.equal(freshCompat.api.readHistory().join(""),"T");
  assert.equal(freshCompat.predictions,1,"compatibility mode can start a fresh shoe");
  assert.equal(a.api.readHistory().length,0,"no B/P/T historical pattern is required or imported");
  assert.equal(a.data.get(PHASE_KEY),"points");
  a.choose("9","0","4");
  assert.equal(a.api.readHistory().join(""),"B");assert.equal(a.predictions,1,"first hand is recorded and forecasts the next hand");
  assert.deepEqual(a.observed.at(-1),{outcome:"B",b:"9",p:"0",c:"4"});
  const n=a.observed.length;
  a.els.macroCardCount.value="5";a.els.macroCardCount.change();
  a.els.macroBankerScore.value="7";a.els.macroBankerScore.change();
  assert.equal(a.api.readHistory().length,1,"one score must not auto-commit");assert.equal(a.observed.length,n);
  a.els.macroPlayerScore.value="2";a.els.macroPlayerScore.change();
  assert.equal(a.api.readHistory().join(""),"BB");assert.equal(a.predictions,2,"completed point entry starts next forecast");
  assert.deepEqual(a.observed.at(-1),{outcome:"B",b:"7",p:"2",c:"5"});
  assert.equal(a.els.macroBankerScore.value,"");assert.equal(a.els.macroPlayerScore.value,"");assert.equal(a.els.macroCardCount.value,"");
  a.els.macroPlayerScore.change();assert.equal(a.api.readHistory().length,2,"cleared score prevents duplicate write");
  a.choose("3","9");a.choose("4","4");
  assert.equal(a.api.readHistory().join(""),"BBPT");assert.equal(a.predictions,4);
  assert.deepEqual(a.observed.slice(-2),[
    {outcome:"P",b:"3",p:"9",c:""},{outcome:"T",b:"4",p:"4",c:""}
  ]);
  a.els.compatModeToggle.click();
  assert.equal(a.api.getInputMode(),"compatibility");assert.equal(a.els.resultModePanel.hidden,false);
  assert.equal(a.els.pointModePanel.hidden,true);assert.equal(a.data.get(INPUT_MODE_KEY),"compatibility");
  a.els.btnP.click();assert.equal(a.api.readHistory().at(-1),"P");assert.equal(a.predictions,5,"compatibility B/P/T still advances analysis");
  a.els.compatModeToggle.click();assert.equal(a.api.getInputMode(),"points");assert.equal(a.els.pointModePanel.hidden,false);
  const disabled=make(["B"]);disabled.api.applyProfileDefaults({enabled:true,point_input:{enabled:false,default_mode:"points"}});
  assert.equal(disabled.api.getInputMode(),"compatibility");assert.equal(disabled.els.compatModeToggle.disabled,true);
  disabled.els.compatModeToggle.click();assert.equal(disabled.api.getInputMode(),"compatibility","disabled setting cannot re-enable point input from the UI");
  a.els.btnBack.click();assert.equal(a.api.getPhase(),"points");assert.equal(a.api.readHistory().at(-1),"T");
  a.els.btnEnd.click();assert.equal(a.api.getPhase(),"points");assert.equal(a.data.get(PHASE_KEY),"points");

  const failed=make(["B"],"history",true);failed.els.btnStart.click();failed.choose("6","2");
  assert.equal(failed.api.readHistory().join(""),"B","failed persistence cannot be reported as committed");
  assert.equal(failed.predictions,1);assert.match(failed.els.pointResultPreview.textContent,/未確認寫入成功/);
  const forecastError=make(["B"]);forecastError.els.btnStart.click();
  forecastError.els.btnStart.addEventListener("click",()=>{throw Error("synthetic next-forecast failure");});
  forecastError.choose("7","4");assert.equal(forecastError.api.readHistory().join(""),"BB");
  assert.match(forecastError.els.pointResultPreview.textContent,/已自動記錄第 2 局/);
  assert.match(forecastError.els.pointResultPreview.textContent,/下一局分析未完成/);
  assert.equal(forecastError.els.macroBankerScore.value,"");
  const restored=make(["B","P"],"points");assert.equal(restored.api.getPhase(),"points");
  await new Promise(resolve=>setImmediate(resolve));assert.equal(restored.predictions,1,"reload restores active analysis once");
  restored.choose("8","1");assert.equal(restored.api.readHistory().join(""),"BPB");

  const html=fs.readFileSync("index.html","utf8"),css=fs.readFileSync("styles.css","utf8"),auth=fs.readFileSync("auth-client.js","utf8");
  for(const id of ids)assert(html.includes('id="'+id+'"'),"missing element "+id);
  assert(html.includes('id="resultModePanel" class="bbb-mode-panel" hidden'));
  assert(!html.includes("historySeed"),"B/P/T history seeding UI is removed");
  assert(!html.includes('id="historyTrack"'),"historical B/P/T pattern display is removed");
  assert(html.includes('id="macroCardCount"'));assert(html.includes('建議使用點數輸入以啟用準算牌'));
  assert(!html.includes('id="btnRecordPoints"'),"score entry remains automatic");
  assert(css.includes("grid-template-columns:repeat(3,minmax(0,1fr))"));
  assert(auth.includes('"point_input_mode.js?v=20261027-score-only-input-v2"'));
  assert(auth.indexOf('"point_input_mode.js?v=20261027-score-only-input-v2"')>auth.indexOf('"final_probability_runtime.js?'));
  assert(!html.includes('<script src="point_input_mode.js"'),"point workflow must run after successful license verification");
  console.log(JSON.stringify({passed:true,default_point_mode:true,no_history_seed:true,automatic_score_commit:true,
    optional_card_count:true,compatibility_toggle:true,undo_and_end:true,reload:true,post_license_only:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
