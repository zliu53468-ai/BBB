/* Authenticated, opt-in Quant256 V2 UI adapter. Existing Frozen Core remains untouched. */
(function(root){
"use strict";
if(!root||!root.location||typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location.search||"").get("quant256")!=="1")return;
const Engine=root.BBBQuantEngine;
if(typeof Engine!=="function")return;
const KEY="bgs256d_frozen_6x15_forward_v18";
const FALLBACK=["bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15"];
const el=id=>document.getElementById(id);
function history(){
  let raw=null;
  try{
    for(const k of [KEY,...FALLBACK]){
      const value=localStorage.getItem(k);
      if(value){raw=JSON.parse(value);break;}
    }
  }catch(_){return [];}
  return Array.isArray(raw?.history)?raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500):[];
}
function context256(seq){
  const baseline=root.__BGS256_TEST__;
  if(!baseline||typeof baseline.context256!=="function")throw Error("missing_frozen_256D_diagnostics");
  const old=baseline.context256(seq),quant=Engine.fromHistory(seq);
  // Preserve all original 128 road slots exactly; V2 is explicitly a separate schema.
  return {vector:quant.injectShoeContext(old.vector,{experimental:true}),
    baseline:old,quant:quant.predictNext(),schema:"BBB_QUANT_128_V2"};
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){if(message)message.textContent="請先輸入牌局紀錄";return null;}
  const result=context256(seq),q=result.quant,skip=q.direction==="Skip";
  const output=el("directionText"),code=el("directionCode"),conf=el("confidence"),
    regime=el("regime"),strength=el("strength"),orb=el("directionOrb");
  if(output)output.textContent=skip?"觀望":q.direction==="B"?"莊":"閒";
  if(code)code.textContent=skip?"QUANT NO BET":q.direction==="B"?"BANKER":"PLAYER";
  if(conf)conf.textContent=skip?"—":(q.probabilities[q.direction]*100).toFixed(2)+"%";
  if(regime)regime.textContent=q.count_source==="observed"?
    "已觀測卡牌："+q.reason:"僅 B/P/T：卡牌數量為估計，不能據此下注";
  if(strength)strength.textContent=(q.bet_fraction*100).toFixed(1)+"% 注碼";
  if(orb)orb.className="direction-orb "+(skip?"idle":q.direction==="B"?"banker":"player");
  const mode=el("modePill");if(mode)mode.textContent="QUANT256 V2 實驗";
  if(message)message.textContent="估計TC "+q.estimated_shoe.trueCount.toFixed(2)+
    "｜算牌權重 "+(100*q.count_weight).toFixed(0)+"%｜EV莊 "+(100*q.ev_banker).toFixed(2)+
    "%／閒 "+(100*q.ev_player).toFixed(2)+"%（未校準）";
  return result;
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(event){
  event.preventDefault();event.stopImmediatePropagation();
  try{run();}catch(err){const m=el("message");if(m)m.textContent="Quant256 無法計算："+err.message;}
},true);
root.__BBB_QUANT256_TEST__={context256,run,history,mode:"experimental",wager_recommendation_validated:false};
})(typeof window!=="undefined"?window:null);
