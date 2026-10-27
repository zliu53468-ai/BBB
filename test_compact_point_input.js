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
const ids=["resultModePanel","pointModePanel","historySeedPanel","entryStageTitle","entryStageNote",
  "historySeed","macroCardCount","macroBankerScore","macroPlayerScore","pointResultPreview","btnStart",
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
  assert.equal(a.api.getPhase(),"history");
  assert.equal(a.api.getInputMode(),"points","point input is the default mode");
  assert.equal(a.els.resultModePanel.hidden,true,"B/P/T buttons are hidden by default");
  assert.equal(a.els.pointModePanel.hidden,true);
  assert.equal(a.api.deriveOutcome("0","0"),"T");
  assert.equal(a.api.deriveOutcome("9","0"),"B");
  assert.equal(a.api.deriveOutcome("0","9"),"P");
  for(const value of ["","-1","10","4.5","NaN"," "])assert.equal(a.api.deriveOutcome(value,"5"),null);
  a.els.btnStart.click();
  assert.equal(a.api.getPhase(),"points","empty shoe can begin with point entry");assert.equal(a.predictions,0);
  assert.equal(a.els.pointModePanel.hidden,false);
  const fresh=make();fresh.els.btnStart.click();fresh.choose("9","0","4");
  assert.equal(fresh.api.readHistory().join(""),"B");assert.equal(fresh.predictions,1,"first scored hand starts next-hand analysis");
  assert.deepEqual(fresh.observed.at(-1),{outcome:"B",b:"9",p:"0",c:"4"});
  const freshCompat=make();freshCompat.els.compatModeToggle.click();
  assert.equal(freshCompat.api.getPhase(),"points");assert.equal(freshCompat.els.resultModePanel.hidden,false);
  freshCompat.els.btnT.click();assert.equal(freshCompat.api.readHistory().join(""),"T");
  assert.equal(freshCompat.predictions,1,"compatibility mode can start a fresh shoe");
  a.els.historySeed.value="B P,T";a.els.btnStart.click();
  assert.equal(a.api.readHistory().join(""),"BPT");
  assert.equal(a.predictions,1,"imported historical results start one forecast");
  assert.equal(a.api.getPhase(),"points");
  assert.equal(a.els.historySeedPanel.hidden,true);assert.equal(a.els.pointModePanel.hidden,false);
  assert.equal(a.data.get(PHASE_KEY),"points");
  const n=a.observed.length;
  a.els.macroCardCount.value="5";a.els.macroCardCount.change();
  a.els.macroBankerScore.value="7";a.els.macroBankerScore.change();
  assert.equal(a.api.readHistory().length,3,"one sco