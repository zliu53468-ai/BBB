"use strict";
/* V2.5 web: typed final-score input only, explicit start, physical-EV direction. */
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
    models.push(this);this.quant={physics:{size:96}};
    this.reset();
  }
  reset(){this.history=[];this.scoreRows=[];}
  syncHistory(history,scores){
    assert.equal(history.length,1,"each scored hand MUST create a fresh one-round model");
    assert.equal(scores.length,1);
    this.history=history.slice();this.scoreRows=scores.slice();
    // Deliberately disagree with research direction and reject the gate.
    // The web should display the physical EV winner, not SKIP.
    const evB=history[0]==="P"?-.03:-.01;
    const evP=history[0]==="P"?-.01:-.02;
    return{history_length:1,research_signal:{
        direction:history[0]==="P"?"B":"P",available:false
      },
      action:"SKIP",ev_gap_deviation:.02,dynamic_gate:.005,
      ev_banker:evB,ev_player:evP,gate_phase:"base",
      win_streak:0,last_settlement:"none",hit_rate:null,signal_rate:null,
      point_score_observation_count:1,
      physical_ev:{simulation_exhausted:false},
      early_shoe_guard:{mode:"independent_final_score_quality",quality:.95,
        ess_ratio:.95,ancestor_diversity:.95,mc_gap_se:.008,
        posterior_gap_spread:.06},
      shoe_distribution_guard:{current_stage:"insufficient_data"},
      markov_residual:{stats:{selected_support:0,eligible:false,
        posterior_conditional:{B:.5068}},shadow:{conditional_shift:0}}};
  }
  setObservedRounds(){}
}
function mount({mobile=false}={}){
  const elements={},handlers={};
  const ids=["predictionApp","bbbPointEntry","pointPlayerFinal","pointBankerFinal",
    "pointInputStatus","btnStart","btnPointAnalyze","btnB","btnP","btnT","btnBack","btnEnd",
    "directionText","directionCode","confidence","regime","strength",
    "directionOrb","modePill","message"];
  const dispatch=(id,type="click",key)=>{
    const event={cancelled:false,key,prevented:false,
      preventDefault(){this.prevented=true;},
      stopImmediatePropagation(){this.cancelled=true;}};
    for(const handler of [...handlers[id][type]].sort(
        (a,b)=>Number(b.capture)-Number(a.capture))){
      if(event.cancelled)break;
      handler.fn(event);
    }
    return event;
  };
  for(const id of ids){
    handlers[id]={click:[],change:[],keydown:[]};
    elements[id]={hidden:false,value:"",textContent:"",className:"",
      classList:{toggle(){}},
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
    matchMedia:()=>({matches:mobile}),
    navigator:{maxTouchPoints:mobile?5:0},innerWidth:mobile?390:1280,
    BBBPhysicsForecastFirst:{PhysicsForecastFirstEngine:FakeModel,VERSION:"TEST_V25"},
    BBBPointSignalQualityGuard:require("./point_signal_quality_guard.js")};
  vm.runInNewContext(script,{window:root,document:root.document,URLSearchParams,console});
  return{ui:root.__BBB_QUANT256_ONLINE_PANEL__,elements,dispatch,
    get legacyStart(){return legacyStart;},
    get legacyBack(){return legacyBack;}};
}
let v=mount(),oldStored=storage.getItem(OLD);
assert.equal(v.ui.pointMode,true);
assert.equal(v.ui.independentCount,0);
assert.equal(v.ui.model.history.length,0);
v.elements.btnStart.click();
assert.equal(v.legacyStart,0,"legacy Start must not activate frozen model");
v.elements.pointPlayerFinal.value="7";
v.elements.pointBankerFinal.value="5";
v.dispatch("pointPlayerFinal","change");
v.dispatch("pointBankerFinal","change");
assert.equal(v.ui.independentCount,0,"typing or leaving a field must not auto-run");
assert.equal(v.ui.latestReport,null,"no prediction before explicit button/desktop Enter");
v.elements.btnPointAnalyze.click();
const first=v.ui.model;
assert.equal(v.ui.independentCount,1);
assert.equal(first.history.join(""),"P");
assert.equal(first.scoreRows[0].player,7);
assert.equal(first.scoreRows[0].banker,5);
assert.equal(v.elements.directionText.textContent,"閒",
  "even when research gate says SKIP, the larger physical EV must show P");
assert(v.elements.regime.textContent.includes("原研究門檻未通過"));
assert(v.elements.regime.textContent.includes("未確認正 EV"));
assert(v.elements.message.textContent.includes("下一局顯示訊號：閒"));
assert(v.elements.message.textContent.includes("觀望判斷："));
assert(v.elements.strength.textContent.includes("觀望門檻 "));
assert(v.elements.strength.textContent.includes("目前 EV 差距 "));
assert.equal(v.elements.pointPlayerFinal.value,"");
assert.equal(v.elements.pointBankerFinal.value,"");
assert.equal(storage.getItem(OLD),oldStored,"old B/P/T history never modified");
assert.equal(JSON.parse(storage.getItem(AUDIT)).rows.length,1);
// Desktop Enter runs only when both scores are valid.
v.elements.pointPlayerFinal.value="2";
v.elements.pointBankerFinal.value="9";
v.dispatch("pointBankerFinal","keydown","Tab");
assert.equal(v.ui.independentCount,1);
const key=v.dispatch("pointBankerFinal","keydown","Enter");
assert.equal(key.prevented,true);
assert.equal(v.ui.independentCount,2);
assert.notEqual(v.ui.model,first,"second score MUST use a fresh 96-particle model");
assert.equal(v.ui.model.history.join(""),"B");
assert.equal(v.elements.directionText.textContent,"莊");
assert.equal(v.ui.latestReport.action,"SKIP",
  "pure presentation change must NOT override BET/EV gate");
assert.equal(storage.getItem(OLD),oldStored);
// Bad score must remain editable; no accidental 0 from empty/whitespace.
for(const bad of [""," ","-1","10","3.5","1e0"]){
  v.elements.pointPlayerFinal.value=bad;
  v.elements.pointBankerFinal.value="0";
  v.elements.btnPointAnalyze.click();
  assert.equal(v.ui.independentCount,2,"invalid final total cannot trigger PF");
}
v.elements.btnBack.click();
assert.equal(v.legacyBack,0);
assert.equal(v.ui.independentCount,1);
assert.equal(v.ui.model.history.join(""),"P");
assert.equal(v.elements.directionText.textContent,"閒");
assert.equal(JSON.parse(storage.getItem(AUDIT)).rows.length,1);
// Page reload may restore only the previous independent score, never old history.
v=mount();
assert.equal(v.ui.independentCount,1);
assert.equal(v.ui.model.history.length,0,"reload must not auto-predict a stale score");
assert.equal(v.elements.directionText.textContent,"—","mobile/desktop waits for explicit new analysis");
// Mobile Enter must not run: tap explicit button.
let m=mount({mobile:true});
m.elements.pointPlayerFinal.value="8";
m.elements.pointBankerFinal.value="8";
m.dispatch("pointPlayerFinal","change");
m.dispatch("pointBankerFinal","keydown","Enter");
assert.equal(m.ui.independentCount,1,"mobile Enter must not run a prediction");
assert(m.elements.pointInputStatus.textContent.includes("手機請點選"));
m.elements.btnPointAnalyze.click();
assert.equal(m.ui.independentCount,2);
assert.equal(m.ui.model.history.join(""),"T");
assert.equal(m.elements.directionText.textContent,"莊",
  "ties in the observed round do not cause a SKIP presentation");
assert.equal(m.ui.model.scoreRows[0].player,8);
assert.equal(m.ui.model.scoreRows[0].banker,8);
m.elements.btnEnd.click();
assert.equal(storage.getItem(AUDIT),null);
assert.equal(storage.getItem(OLD),oldStored);
assert.equal(m.ui.model.history.length,0);
console.log("PASS V2.5 web: manual typed points only, desktop Enter/mobile button, always stronger physical EV direction, no SKIP display, no model/legacy mutation");
