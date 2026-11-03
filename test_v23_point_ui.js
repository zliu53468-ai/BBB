"use strict";
/* V2.4 web-only: no history bootstrap; each point pair gets a fresh PF model. */
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm");
const script=fs.readFileSync("quant256_online_panel.js","utf8");
const OLD="bgs256d_frozen_6x15_forward_v18",
  AUDIT="bgs_v24_independent_point_audit";
const values=new Map([[OLD,JSON.stringify({history:["B","P","T","B","P"]})]]);
const storage={
  getItem:k=>values.has(k)?values.get(k):null,
  setItem:(k,v)=>values.set(k,String(v)),
  removeItem:k=>values.delete(k)
};
const models=[];
class FakeModel{
  constructor(){
    models.push(this);this.quant={physics:{size:256}};
    this.reset();
  }
  reset(){this.history=[];this.scoreRows=[];}
  syncHistory(history,scores){
    assert.equal(history.length,1,"MUST process exactly one scored outcome");
    assert.equal(scores.length,1,"MUST process exactly one score pair");
    this.history=history.slice();this.scoreRows=scores.slice();
    return{history_length:1,research_signal:{direction:"B",available:true},
      action:"SKIP",ev_gap_deviation:.02,dynamic_gate:.005,
      ev_banker:-.01,ev_player:-.02,gate_phase:"base",
      win_streak:0,last_settlement:"none",hit_rate:null,signal_rate:null,
      point_score_observation_count:1,
      shoe_distribution_guard:{current_stage:"insufficient_data"},
      markov_residual:{stats:{selected_support:0,eligible:false,
        posterior_conditional:{B:.5068}},shadow:{conditional_shift:0}}};
  }
  setObservedRounds(){}
}
function mount(){
  const elements={},handlers={},classNames=new Set();
  const ids=["predictionApp","bbbPointEntry","pointPlayerFinal","pointBankerFinal",
    "pointInputStatus","btnStart","btnB","btnP","btnT","btnBack","btnEnd",
    "roundPill","roundCount","directionText","directionCode","confidence",
    "regime","strength","directionOrb","modePill","message"];
  const dispatch=(id,type="click")=>{
    const event={cancelled:false,preventDefault(){},
      stopImmediatePropagation(){this.cancelled=true;}};
    for(const handler of [...handlers[id][type]].sort(
        (a,b)=>Number(b.capture)-Number(a.capture))){
      if(event.cancelled)break;
      handler.fn(event);
    }
  };
  for(const id of ids){
    handlers[id]={click:[],change:[]};
    elements[id]={hidden:false,value:"",textContent:"",className:"",
      classList:{toggle(name,on){if(on)classNames.add(name);else classNames.delete(name);}},
      addEventListener(type,fn,capture=false){
        handlers[id][type].push({fn,capture:!!capture});
      },
      click(){dispatch(id,"click");}};
  }
  let legacyStart=0,legacyBack=0;
  elements.btnStart.addEventListener("click",()=>{legacyStart++;});
  elements.btnBack.addEventListener("click",()=>{legacyBack++;
    const data=JSON.parse(storage.getItem(OLD));data.history.pop();
    storage.setItem(OLD,JSON.stringify(data));
  });
  const root={document:{getElementById:id=>elements[id]||null},
    localStorage:storage,location:{search:""},
    BBBPhysicsForecastFirst:{PhysicsForecastFirstEngine:FakeModel,VERSION:"TEST_V24"}};
  vm.runInNewContext(script,{window:root,document:root.document,URLSearchParams,console});
  return{ui:root.__BBB_QUANT256_ONLINE_PANEL__,elements,dispatch,
    get legacyStart(){return legacyStart;},
    get legacyBack(){return legacyBack;}};
}
let v=mount(),oldStored=storage.getItem(OLD);
assert.equal(v.ui.pointMode,true,"scores only from first authorized render");
assert.equal(v.ui.independentCount,0);
assert.equal(v.elements.bbbPointEntry.hidden,false);
assert.equal(v.ui.model.history.length,0);
v.elements.btnStart.click();
assert.equal(v.legacyStart,0,"Start must not activate frozen legacy model");
v.elements.pointPlayerFinal.value="7";
v.dispatch("pointPlayerFinal","change");
assert.equal(v.ui.independentCount,0,"one partial score is NOT an observation");
v.elements.pointBankerFinal.value="5";
v.dispatch("pointBankerFinal","change");
const first=v.ui.model;
assert.equal(v.ui.independentCount,1);
assert.equal(first.history.join(""),"P");
assert.equal(first.scoreRows[0].player,7);
assert.equal(first.scoreRows[0].banker,5);
assert.equal(v.elements.pointPlayerFinal.value,"");
assert.equal(v.elements.pointBankerFinal.value,"");
assert(v.elements.pointInputStatus.textContent.includes("獨立分析"));
assert.equal(JSON.parse(storage.getItem(AUDIT)).rows.length,1);
assert.equal(storage.getItem(OLD),oldStored,
  "old cumulative B/P/T history MUST NEVER be used or modified");
v.elements.pointPlayerFinal.value="2";
v.elements.pointBankerFinal.value="9";
v.dispatch("pointBankerFinal","change");
assert.notEqual(v.ui.model,first,"second score MUST use NEW particle estimator");
assert.equal(v.ui.model.history.join(""),"B");
assert.equal(v.ui.model.history.length,1);
assert.equal(v.ui.model.scoreRows.length,1);
assert.equal(v.ui.independentCount,2);
assert.equal(storage.getItem(OLD),oldStored);
v.elements.btnBack.click();
assert.equal(v.legacyBack,0,"undo MUST NOT invoke historical B/P/T back");
assert.equal(v.ui.independentCount,1);
assert.equal(v.ui.model.history.join(""),"P");
assert.notEqual(v.ui.model,first,"undo also reconstructs from fresh particles");
assert.equal(v.ui.model.scoreRows[0].player,7);
assert.equal(JSON.parse(storage.getItem(AUDIT)).rows.length,1);
// License-scoped audit may restore the most recent score but never replay
// any earlier scores into the live PF state.
v=mount();
assert.equal(v.ui.pointMode,true);
assert.equal(v.ui.independentCount,1);
assert.equal(v.ui.model.history.length,1);
assert.equal(v.ui.model.scoreRows[0].banker,5);
assert.equal(storage.getItem(OLD),oldStored);
v.elements.btnBack.click();
assert.equal(v.ui.model.history.length,0);
assert.equal(v.ui.independentCount,0);
assert.equal(JSON.parse(storage.getItem(AUDIT)).rows.length,0);
v.elements.pointPlayerFinal.value="8";
v.elements.pointBankerFinal.value="8";
v.dispatch("pointBankerFinal","change");
assert.equal(v.ui.model.history.join(""),"T");
assert.equal(v.ui.model.history.length,1);
assert.equal(v.ui.model.scoreRows[0].player,8);
assert.equal(v.ui.model.scoreRows[0].banker,8);
assert.equal(v.ui.independentCount,1);
v.elements.btnEnd.click();
assert.equal(storage.getItem(AUDIT),null);
assert.equal(storage.getItem(OLD),oldStored,
  "legacy state remains untouched until the authorized end-analysis purge");
assert.equal(v.ui.model.history.length,0);
console.log("PASS V2.4 web independent scores: no bootstrap, fresh model per score, no legacy history, undo, reload, tie, end");
