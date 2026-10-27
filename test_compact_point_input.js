"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
class Element{
  constructor(id){this.id=id;this.listeners={};this.value="";this.hidden=false;this.textContent="";this.attributes={};
    this.classList={names:new Set(),toggle:(k,on)=>{if(on)this.classList.names.add(k);else this.classList.names.delete(k);}};
  }
  addEventListener(name,fn){(this.listeners[name] ||= []).push(fn);}
  setAttribute(name,value){this.attributes[name]=value;}
  click(){for(const fn of this.listeners.click||[])fn({});}
  change(){for(const fn of this.listeners.change||[])fn({});}
}
const ids=["modeResults","modePoints","resultModePanel","pointModePanel","macroBankerScore",
  "macroPlayerScore","btnRecordPoints","pointResultPreview","btnB","btnP","btnT","btnBack","btnEnd"];
const els=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
const storage=new Map([["bgs256d_frozen_6x15_forward_v18",JSON.stringify({history:[]})]]);
const localStorage={getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,String(v)),
  removeItem:k=>storage.delete(k)};
const captured=[];
for(const outcome of ["B","P","T"])els["btn"+outcome].addEventListener("click",()=>{
  const b=els.macroBankerScore.value,p=els.macroPlayerScore.value;
  captured.push({outcome,b,p});
  const record=JSON.parse(localStorage.getItem("bgs256d_frozen_6x15_forward_v18"));
  record.history.push(outcome);
  localStorage.setItem("bgs256d_frozen_6x15_forward_v18",JSON.stringify(record));
  els.macroBankerScore.value="";els.macroPlayerScore.value="";
});
els.btnBack.addEventListener("click",()=>{
  const j=JSON.parse(localStorage.getItem("bgs256d_frozen_6x15_forward_v18"));
  j.history.pop();localStorage.setItem("bgs256d_frozen_6x15_forward_v18",JSON.stringify(j));
});
const context={window:{},document:{getElementById:id=>els[id]||null},localStorage};
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync("point_input_mode.js","utf8"),context);
const api=context.__BBB_POINT_INPUT__;
assert(api);
assert.equal(els.resultModePanel.hidden,false);
assert.equal(els.pointModePanel.hidden,true);
assert.equal(api.deriveOutcome("0","0"),"T");
assert.equal(api.deriveOutcome("9","0"),"B");
assert.equal(api.deriveOutcome("0","9"),"P");
for(const val of ["","10","-1","2.5","NaN"])assert.equal(api.deriveOutcome(val,"5"),null);
els.modePoints.click();
assert.equal(els.pointModePanel.hidden,false);
assert.equal(els.modePoints.attributes["aria-pressed"],"true");
assert.equal(els.modeResults.attributes["aria-pressed"],"false");
assert.equal(api.recordPointHand(),false,"blank scores never trigger bets");
els.macroBankerScore.value="7";els.macroPlayerScore.value="2";els.macroPlayerScore.change();
assert.match(els.pointResultPreview.textContent,/莊勝/);
els.btnRecordPoints.click();
assert.equal(captured.length,1);
assert.deepEqual(captured[0],{outcome:"B",b:"7",p:"2"});
assert.equal(api.readHistory().join(""),"B");
els.macroBankerScore.value="4";els.macroPlayerScore.value="4";els.btnRecordPoints.click();
assert.equal(api.readHistory().join(""),"BT");
els.modeResults.click();
assert.equal(els.macroBankerScore.value,"");
assert.equal(els.macroPlayerScore.value,"");
els.btnP.click();
assert.deepEqual(captured[2],{outcome:"P",b:"",p:""});
assert.equal(api.readHistory().join(""),"BTP");
els.btnBack.click();
assert.equal(api.readHistory().join(""),"BT");
els.modePoints.click();els.macroBankerScore.value="1";els.macroPlayerScore.value="9";
els.modeResults.click();
assert.equal(els.macroBankerScore.value,"","switching cannot leak half-entered scores");
assert.equal(els.macroPlayerScore.value,"");
const html=fs.readFileSync("index.html","utf8"),css=fs.readFileSync("styles.css","utf8"),auth=fs.readFileSync("auth-client.js","utf8");
for(const id of ids)assert(html.includes('id="'+id+'"'),"missing "+id);
assert(css.includes(".bbb-points-row")&&css.includes("@media(max-width:620px)"));
assert(auth.includes('"point_input_mode.js?v=20261027-compact-score-v1"'));
assert(auth.indexOf('"point_input_mode.js?v=20261027-compact-score-v1"')>auth.indexOf('"final_probability_runtime.js?'));
assert(!html.includes('<script src="point_input_mode.js"'),"must not load before authorization");
console.log(JSON.stringify({passed:true,score_outcome_and_validation:true,original_buttons:true,mixed_history_and_undo:true,auth_loader_order:true,compact_css:true}));
