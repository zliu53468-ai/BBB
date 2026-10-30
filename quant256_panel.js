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
function scoreTwoArms(vector,quant=null){
  if(!Array.isArray(vector)||vector.length!==256||!vector.every(Number.isFinite))
    throw Error("expected finite 256D");
  const dim=256,ridge=1,alpha=.5;
  const theta=new Array(dim).fill(0);
  theta[0]=.028;theta[1]=.045;theta[2]=.018;theta[3]=.008;
  // No trained coefficients exist for the remaining dimensions. Keep all
  // 256 input coordinates for diagnostics, but give road (128..255) ZERO
  // directional weight AND ZERO LinUCB uncertainty ownership. Do not
  // invent sign priors for ratios or B/P/T history windows.
  for(let i=4;i<256;i++)theta[i]=0;
  let projected=0,variance=0,shoeContribution=0,roadContribution=0;
  for(let i=0;i<dim;i++){
    const term=vector[i]*theta[i]/ridge;
    projected+=term;
    if(i<128)shoeContribution+=term;else roadContribution+=term;
    if(i<4)variance+=(vector[i]*vector[i])/ridge;
  }
  const uncertainty=alpha*Math.sqrt(variance);
  const rawDirection=projected>=0?"B":"P";
  // Hard gate is applied *after* LinUCB scoring. The 128 road coordinates
  // remain for traceability, but NEVER authorize a bet, override TC, or
  // flip the count engine's experimental direction.
  const countGate=quant?.gates?.count_passed===true;
  const evGate=quant?.gates?.ev_passed===true;
  const allowed=countGate&&["B","P"].includes(quant.direction);
  const direction=allowed?quant.direction:"Skip";
  return {B:{mean:projected,score:projected+uncertainty},
    P:{mean:-projected,score:-projected+uncertainty},
    gap:2*projected,shoeContribution,roadContribution,
    alpha,ridge,trained:false,raw_direction:rawDirection,direction,
    hard_gate:{count_passed:countGate,ev_passed:evGate,
      count_is_authority:true,threshold_exclusive:2.5,
      ev_is_diagnostic_only:true,road_override_allowed:false,
      linucb_disagreed:allowed&&rawDirection!==direction,
      simulated_only:true,bet_fraction:0}};
}
function context256(seq){
  const frozen=root.__BGS256_TEST__;
  if(!frozen||typeof frozen.context256!=="function")throw Error("missing frozen 256D diagnostics");
  const old=frozen.context256(seq),engine=Engine.fromHistory(seq);
  root.__BBB_SHOE_QUANT_ENGINE__=engine;
  // V5 explicitly decouples both 128D blocks. The old vector is only a
  // diagnostic baseline. No road coordinate can reach LinUCB on this panel.
  const vector=old.vector.slice();
  const shoe=engine.generateShoeContext();
  if(vector.length!==256||shoe.length!==128)throw Error("quant 128D layout mismatch");
  for(let i=0;i<128;i++)vector[i]=shoe[i];
  for(let i=128;i<256;i++)vector[i]=0;
  const features=engine.survivalFeatures();
  if(features.length!==16||!features.every(Number.isFinite))
    throw Error("invalid_survival_feature_schema");
  // Old big-eye/small/cockroach windows do NOT enter these slots:
  // 17..32 (16..31) and 129..144 (128..143) contain CPSI/HSMM data.
  for(let i=0;i<16;i++){
    vector[16+i]=features[i];
    vector[128+i]=features[i];
  }
  const quant=engine.predictNext(),arms=scoreTwoArms(vector,quant);
  return {vector,baseline:old,quant,arms,
    cpsi_features:features,road_masked:true,
    raw_road_diagnostics:old.vector.slice(128),
    schema:"BBB_SHOE_QUANT_128_V6_CPSI"};
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){if(message)message.textContent="請先輸入牌局紀錄";return null;}
  const result=context256(seq),q=result.quant,
    // This is a research indicator, not a wagering instruction.
    finalDirection=result.arms.direction,skip=finalDirection==="Skip";
  const output=el("directionText"),code=el("directionCode"),conf=el("confidence"),
    regime=el("regime"),strength=el("strength"),orb=el("directionOrb");
  if(output)output.textContent=skip?"觀望":finalDirection==="B"?"莊":"閒";
  if(code)code.textContent=skip?"QUANT SKIP":finalDirection==="B"?"BANKER":"PLAYER";
  if(conf)conf.textContent=skip?"—":"代理 TC "+q.estimated_shoe.trueCount.toFixed(2)+"（非勝率）";
  if(regime)regime.textContent=q.count_source==="estimated"?
    "估計扣牌／未校準："+(skip?"未達門檻":"僅研究方向"):"觀測卡牌／未經回測校準";
  if(strength)strength.textContent=skip?"0%（觀望）":
    "模擬比率 "+(q.simulated_bet_fraction*100).toFixed(1)+"%（實際注碼 0%）";
  if(orb)orb.className="direction-orb "+(skip?"idle":finalDirection==="B"?"banker":"player");
  const mode=el("modePill");if(mode)mode.textContent="QUANT256 V6 長龍生存測試";
  // Side bets are independent research indicators; do not treat unobserved
  // pairs/Lucky6 as physical evidence or automatically recommend a stake.
  const sideNames={banker_pair:"莊對",player_pair:"閒對",tie:"和局",lucky_six:"幸運六"};
  const sideSummary=Object.entries(q.side_bets).map(([key,v])=>
    sideNames[key]+":"+(v.signal?"觀察訊號":"資料不足／無訊號")).join("、");
  if(message)message.textContent="代理TC "+q.estimated_shoe.trueCount.toFixed(2)+
    "｜長龍 "+q.survival.run_length+" 局／"+q.survival.reason+
    "｜CPSI "+q.survival.cpsi.toFixed(3)+
    "／Hazard指標 "+q.survival.hazard_rate.toFixed(3)+
    "｜PF條件重採樣 "+(q.particle_survival.active?"觀測牌值研究":"未啟動")+
    "｜原始牌路特徵不參與模型"+
    "｜診斷EV莊 "+(q.ev_banker*100).toFixed(2)+"%／閒 "+(q.ev_player*100).toFixed(2)+"%"+
    "｜兩臂差 "+result.arms.gap.toFixed(2)+
    "｜邊注觀察："+sideSummary+
    (result.arms.hard_gate.linucb_disagreed?"｜LinUCB分歧但無權翻轉":"")+
    "（未校準；非下注建議）";
  return result;
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(e){
  e.preventDefault();e.stopImmediatePropagation();
  try{run();}catch(err){const m=el("message");if(m)m.textContent="Quant256 計算失敗："+err.message;}
},true);
root.__BBB_QUANT256_TEST__={
  context256,scoreTwoArms,run,history,mode:"experimental_v6_cpsi_hazard",
  wager_recommendation_validated:false
};
})(typeof window!=="undefined"?window:null);
