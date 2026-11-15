/* BBB trial-only completed prediction accuracy counter.
 * Excludes skips and tied outcomes. Does not infer winnings or account balances.
 */
(function(){
"use strict";
const key="bbb-trial-prediction-hit-v1";
const minResolved=10;
const limit=.60;
const el=id=>document.getElementById(id);
const box=el("bbbTrialAccuracyPanel");
if(!box)return;
let active=false,ledger=[],redirectTimer=null,ending=false;
function safeGet(){
  try{return JSON.parse(sessionStorage.getItem(key)||"null");}catch(_){return null;}
}
function save(){
  try{sessionStorage.setItem(key,JSON.stringify({version:1,ledger,ending}));}catch(_){}
}
function sameHistory(items,history){
  return Array.isArray(items)&&Array.isArray(history)&&items.length===history.length&&
    items.every((row,i)=>row&&row.outcome===history[i]);
}
function tally(){
 let hits=0,valid=0;
 for(const item of ledger){
   if(["B","P"].includes(item.prediction)&&["B","P"].includes(item.outcome)){
     valid++;
     if(item.prediction===item.outcome)hits++;
   }
 }
 return{hits,valid,rate:valid?hits/valid:null};
}
function render(){
 const t=tally();
 el("bbbTrialAccuracyHits").textContent=String(t.hits);
 el("bbbTrialAccuracyValid").textContent=String(t.valid);
 el("bbbTrialAccuracyRate").textContent=t.rate===null?"—":(t.rate*100).toFixed(1)+"%";
 const done=t.valid>=minResolved&&t.rate>=limit;
 el("bbbTrialAccuracyStatus").textContent=done
   ?"本次試用已達有效預測命中率 60% 的結束條件，即將顯示後續聯繫方式。"
   :"滿 "+minResolved+" 次有效預測結算且命中率達 60%，試用將提前結束；觀望與和局不計入。";
 return done;
}
function cancelRedirect(){
 if(redirectTimer!==null)clearTimeout(redirectTimer);
 redirectTimer=null;ending=false;
}
function scheduleIfQualified(){
 if(!active)return;
 const matched=render();
 if(!matched){cancelRedirect();save();return;}
 if(redirectTimer!==null)return;
 ending=true;save();
 redirectTimer=setTimeout(()=>{
   const {valid,rate}=tally();
   if(!active||valid<minResolved||rate<limit)return;
   window.location.replace("./trial-ended.html?reason=accuracy60");
 },1600);
}
function mount(opts={}){
 active=true;
 box.hidden=false;
 const observed=window.__BBB_SOFT_SHOE_PANEL__?.history||[];
 const saved=opts.new_session?null:safeGet();
 // Untracked historical hands are baselines, never retroactively scored.
 ledger=saved?.version===1&&sameHistory(saved.ledger,observed)
   ?saved.ledger:[...observed].map(outcome=>({outcome,prediction:null}));
 ending=false;
 scheduleIfQualified();
}
window.addEventListener("bbb:trial-result",event=>{
 if(!active)return;
 const d=event.detail;
 if(!Array.isArray(d?.before)||!Array.isArray(d.after)||!["B","P","T"].includes(d.outcome))return;
 if(!sameHistory(ledger,d.before))
   ledger=d.before.map(outcome=>({outcome,prediction:null}));
 if(d.after.length!==d.before.length+1||d.after.at(-1)!==d.outcome)return;
 ledger.push({outcome:d.outcome,prediction:["B","P"].includes(d.prediction)?d.prediction:null});
 scheduleIfQualified();
});
window.addEventListener("bbb:trial-back",event=>{
 if(!active)return;
 const history=event.detail?.after;
 if(!Array.isArray(history))return;
 cancelRedirect();
 ledger=ledger.slice(0,history.length);
 if(!sameHistory(ledger,history))
   ledger=history.map(outcome=>({outcome,prediction:null}));
 scheduleIfQualified();
});
window.addEventListener("bbb:trial-reset",()=>{
 if(!active)return;
 cancelRedirect();ledger=[];save();render();
});
window.BBB_TRIAL_ACCURACY={
 onTrialMounted:mount,
 snapshot(){return tally();}
};
})();
