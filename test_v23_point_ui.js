"use strict";
/** V2.3: browser state test without slow physics; real engine tested separately. */
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm");
const script=fs.readFileSync("quant256_online_panel.js","utf8");
const KEY="bgs256d_frozen_6x15_forward_v18";
const STORE="bgs_v23_point_score_log",STAGE="bgs_v23_point_entry_mode";
const values=new Map([[KEY,JSON.stringify({history:["B","P","T"]})]]);
const storage={
  getItem:k=>values.has(k)?values.get(k):null,
  setItem:(k,v)=>values.set(k,String(v)),
  removeItem:k=>values.delete(k)
};
class FakeModel{
  constructor(){this.reset();}
  reset(){this.history=[];this.scores=[];}
  syncHistory(history,scores){
    this.history=history.slice();this.scores=scores.slice();
    return{history_length:history.length,research_signal:{direction:"B",available:true},
      action:"SKIP",ev_gap_deviation:.02,dynamic_gate:.005,
      ev_banker:-.01,ev_player:-.02,gate_phase:"base",
      win_streak:0,last_settlement:"none",hit_rate:null,signal_rate:null,
      point_score_observation_count:scores.filter(Boolean).length,
      shoe_distribution_guard:{current_stage:"insufficient_data"},
      markov_residual:{stats:{selected_support:0,eligible:false,
        posterior_conditional:{B:.5068}},shadow:{conditional_shift:0}}};
  }
  setObservedRounds(){}
}
function mount(){
 const elements={},handlers={};
 const ids=["predictionApp","bbbPointEntry","pointPlayerFinal","pointBankerFinal",
   "pointInputStatus","btnStart","btnB","btnP","btnT","btnBack","btnEnd",
   "directionText","directionCode","confidence","regime","strength",
   "directionOrb","modePill","message"];
 const classNames=new Set();
 const dispatch=(id,type="click")=>{
   const ev={cancelled:false,preventDefault(){},
     stopImmediatePropagation(){this.cancelled=true;}};
   for(const h of [...handlers[id][type]].sort((a,b)=>Number(b.capture)-Number(a.capture))){
     if(ev.cancelled)break;
     h.fn(ev);
   }
 };
 for(const id of ids){
   handlers[id]={click:[],change:[]};
   elements[id]={hidden:id==="bbbPointEntry",value:"",textContent:"",
     className:"",classList:{toggle(name,on){if(on)classNames.add(name);else classNames.delete(name);}},
     addEventListener(type,fn,capture=false){handlers[id][type].push({fn,capture:!!capture});},
     click(){dispatch(id,"click");}};
 }
 const core={getHistory:()=>JSON.parse(storage.getItem(KEY)).history,
   append(v){const a=this.getHistory();a.push(v);storage.setItem(KEY,JSON.stringify({history:a}));},
   back(){const a=this.getHistory();a.pop();storage.setItem(KEY,JSON.stringify({history:a}));}};
 for(const outcome of ["B","P","T"])
   elements["btn"+outcome].addEventListener("click",()=>core.append(outcome));
 elements.btnBack.addEventListener("click",()=>core.back());
 let startFallback=0;
 elements.btnStart.addEventListener("click",()=>{startFallback++;});
 const root={document:{getElementById:id=>elements[id]||null},localStorage:storage,
   location:{search:""},BBBPhysicsForecastFirst:{PhysicsForecastFirstEngine:FakeModel,VERSION:"TEST_V23"}};
 vm.runInNewContext(script,{window:root,document:root.document,URLSearchParams,console});
 return{ui:root.__BBB_QUANT256_ONLINE_PANEL__,elements,dispatch,classNames,core,
   get startFallback(){return startFallback;}};
}
let view=mount();
assert.equal(view.ui.pointMode,false);
assert(view.elements.bbbPointEntry.hidden);
view.elements.btnStart.click();
assert.equal(view.ui.pointMode,true);
assert.equal(view.elements.bbbPointEntry.hidden,false);
assert.equal(storage.getItem(STAGE),"1");
assert.equal(view.startFallback,0,"start must prevent old Frozen prediction handler");
assert.deepEqual(view.ui.model.scores,[null,null,null]);
view.elements.pointPlayerFinal.value="7";
view.dispatch("pointPlayerFinal","change");
assert.equal(view.core.getHistory().length,3,"single score cannot settle a hand");
view.elements.pointBankerFinal.value="5";
view.dispatch("pointBankerFinal","change");
assert.deepEqual(view.core.getHistory(),["B","P","T","P"]);
assert.equal(view.ui.model.scores.length,4);
assert.equal(view.ui.model.scores[3].player,7);
assert.equal(view.ui.model.scores[3].banker,5);
assert.equal(view.elements.pointPlayerFinal.value,"");
assert.equal(view.elements.pointBankerFinal.value,"");
assert(view.elements.pointInputStatus.textContent.includes("閒"));
const stored=JSON.parse(storage.getItem(STORE));
assert.deepEqual(stored.history,["B","P","T","P"]);
assert.equal(stored.scores[3].player,7);
view.elements.btnBack.click();
assert.deepEqual(view.core.getHistory(),["B","P","T"]);
assert.equal(view.ui.model.scores.length,3);
assert.equal(JSON.parse(storage.getItem(STORE)).scores.length,3,
  "undo should physically trim point evidence, not resurrect old score data");
view.elements.pointPlayerFinal.value="4";
view.elements.pointBankerFinal.value="6";
view.dispatch("pointBankerFinal","change");
assert.deepEqual(view.core.getHistory(),["B","P","T","B"]);
assert.equal(view.ui.model.scores[3].player,4);
assert.equal(view.ui.model.scores[3].banker,6);
// Recreated browser panel must restore the licensed point mode and point rows.
view=mount();
assert.equal(view.ui.pointMode,true);
assert.equal(view.elements.bbbPointEntry.hidden,false);
assert.equal(view.ui.model.scores[3].player,4);
assert.equal(view.ui.model.scores[3].banker,6);
view.elements.btnEnd.click();
assert.equal(storage.getItem(STAGE),null);
assert.equal(storage.getItem(STORE),null);
console.log("PASS V2.3 browser UI: historical Start -> compact score pair -> single outcome -> undo -> new score -> reload -> license-scope clear");
