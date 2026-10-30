/**
 * Opt-in authenticated BGS 256D experimental panel (?quant256=1).
 * Keeps the pinned Frozen Core, its storage, and licensing unchanged.
 * Experimental 128D count features dominate an explicitly UNTRAINED
 * two-arm LinUCB projection. No historical coefficients are reused.
 */
(function(root){
"use strict";
if(!root||!root.location||typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location.search||"").get("quant256")!=="1")return;
const Engine=root.BBBShoeQuantEngine||root.BBBQuantEngine;
if(typeof Engine!=="function")return;
const KEY="bgs256d_frozen_6x15_forward_v18";
const FALLBACK=["bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15"];
const el=id=>document.getElementById(id);
function history(){
  let raw=null;
  try{
    for(const key of [KEY,...FALLBACK]){
      const value=localStorage.getItem(key);
      if(value){raw=JSON.parse(value);break;}
    }
  }catch(_){return [];}
  return Array.isArray(raw?.history)?raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500):[];
}
/**
 * Diagonal two-arm LinUCB shadow. No online fitting and no accuracy claims.
 * The new 128D coordinates have DIFFERENT meaning from frozen 128D:
 * custom count coefficients dominate road coefficients by construction.
 * Alpha is an exploration parameter, not a win-probability calibration.
 */
function scoreTwoArms(vector){
  if(!Array.isArray(vector)||vector.length!==256||!vector.every(Number.isFinite))
    throw Error("expected finite 256D");
  const dim=256,ridge=1,alpha=.5;
  const theta=new Array(dim).fill(0);
  theta[0]=.028;theta[1]=.045;theta[2]=.018;theta[3]=.008;
  // Include contextual rank densities and progress with intentionally
  // small directional priors. Without fit data these remain diagnostics.
  for(let i=4;i<128;i++)theta[i]=(i%2?-1:1)*.0001;
  for(let i=128;i<256;i++)theta[i]=(i%2?-1:1)*.00025;
  let projected=0,variance=0,shoeContribution=0,roadContribution=0;
  for(let i=0;i<dim;i++){
    const term=vector[i]*theta[i]/ridge;
    projected+=term;
    if(i<128)shoeContribution+=term;else roadContribution+=term;
    variance+=(vector[i]*vector[i])/ridge;
  }
  const uncertainty=alpha*Math.sqrt(variance);
  return {B:{mean:projected,score:projected+uncertainty},
    P:{mean:-projected,score:-projected+uncertainty},
    gap:2*projected,shoeContribution,roadContribution,
    alpha,ridge,trained:false,direction:projected>=0?"B":"P"};
}
function context256(seq){
  const frozen=root.__BGS256_TEST__;
  if(!frozen||typeof frozen.context256!=="function")throw Error("missing frozen 256D diagnostics");
  const old=frozen.context256(seq),engine=Engine.fromHistory(seq);
  root.__BBB_SHOE_QUANT_ENGINE__=engine;
  // Requirement: literally overwrite slots 0..127, preserve slots 128..255.
  const vector=old.vector.slice();
  const shoe=engine.generateShoeContext();
  if(vector.length!==256||shoe.length!==128)throw Error("quant 128D layout mismatch");
  for(let i=0;i<128;i++)vector[i]=shoe[i];
  const arms=scoreTwoArms(vector);
  return {vector,baseline:old,quant:engine.predictNext(),arms,
    schema:"BBB_SHOE_QUANT_128_V3"};
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){if(message)message.textContent="請先輸入牌局紀錄";return null;}
  const result=context256(seq),q=result.quant,skip=q.direction==="Skip";
  const output=el("directionText"),code=el("directionCode"),conf=el("confidence"),
    regime=el("regime"),strength=el("strength"),orb=el("directionOrb");
  if(output)output.textContent=skip?"觀望":q.direction==="B"?"莊":"閒";
  if(code)code.textContent=skip?"QUANT SKIP":q.direction==="B"?"BANKER":"PLAYER";
  if(conf)conf.textContent=skip?"—":(q.probabilities[q.direction]*100).toFixed(2)+"%（模型假設）";
  if(regime)regime.textContent=q.count_source==="estimated"?
    "估計扣牌／未校準："+(skip?"未達門檻":"僅研究方向"):"觀測卡牌／未經回測校準";
  if(strength)strength.textContent=skip?"0%（觀望）":
    "模擬凱利 "+(q.simulated_bet_fraction*100).toFixed(1)+"%（非建議注碼）";
  if(orb)orb.className="direction-orb "+(skip?"idle":q.direction==="B"?"banker":"player");
  const mode=el("modePill");if(mode)mode.textContent="QUANT256 V3 實驗";
  if(message)message.textContent="估計TC "+q.estimated_shoe.trueCount.toFixed(2)+
    "｜算牌權重 "+(100*q.count_weight).toFixed(0)+"%"+
    "｜模型EV莊 "+(q.ev_banker*100).toFixed(2)+"%／閒 "+(q.ev_player*100).toFixed(2)+
    "%｜兩臂分數差 "+result.arms.gap.toFixed(2)+"（非真實勝率）";
  return result;
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(e){
  e.preventDefault();e.stopImmediatePropagation();
  try{run();}catch(err){const m=el("message");if(m)m.textContent="Quant256 計算失敗："+err.message;}
},true);
root.__BBB_QUANT256_TEST__={
  context256,scoreTwoArms,run,history,mode:"experimental_v3",
  wager_recommendation_validated:false
};
})(typeof window!=="undefined"?window:null);
