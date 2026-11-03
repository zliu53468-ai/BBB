/**
 * BBB V2.4 WEB ONLY — independent point-score input and per-round PF.
 *
 * No historical B/P/T bootstrap and NO accumulated particle posterior:
 * EACH (Player final total, Banker final total) pair is evaluated with a
 * freshly created V2.3 PhysicsForecastFirstEngine(N256/L384/M160).
 * Old V2.3 licensed history and score logs are never read or mutated.
 * An optional license-scoped audit of prior score pairs is NOT model input.
 * All physics/EV/Markov/gate implementations remain unchanged.
 */
(function(root){
"use strict";
const signalGuard=root?.BBBPointSignalQualityGuard||
  (typeof module==="object"&&module.exports&&typeof require==="function"?
    require("./point_signal_quality_guard.js"):null);
if(!root||!root.document||!root.BBBPhysicsForecastFirst||
  !signalGuard||
  typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location?.search||"").get("quant256")==="frozen")return;
const {PhysicsForecastFirstEngine,VERSION}=root.BBBPhysicsForecastFirst;
const CONFIG={threshold:.005,lossGate:.035,simulationsPerParticle:160,
  observedSamples:12000,quantOptions:{lambda:.82,alpha:.15,particles:256},
  independentFinalScoreMode:true};
// Compatibility-only name: never import or replay this old B/P/T history.
const KEY="bgs256d_frozen_6x15_forward_v18";
const AUDIT_KEY="bgs_v24_independent_point_audit";
const el=id=>root.document.getElementById(id);
let model=new PhysicsForecastFirstEngine(CONFIG);
let busy=false,lastReport=null;
function validPair(pair){
  return Boolean(pair&&Number.isInteger(pair.player)&&
    Number.isInteger(pair.banker)&&pair.player>=0&&pair.player<=9&&
    pair.banker>=0&&pair.banker<=9);
}
function resultOf(pair){
  return pair.banker>pair.player?"B":pair.player>pair.banker?"P":"T";
}
function loadAudit(){
  try{
    const value=JSON.parse(root.localStorage?.getItem(AUDIT_KEY)||"null");
    if(!value||value.version!==1||!Array.isArray(value.rows))return[];
    // The audit is display/undo data only; never passed as model history.
    return value.rows.filter(validPair).slice(-500).map(x=>({
      player:x.player,banker:x.banker
    }));
  }catch(_){return[];}
}
let audit=loadAudit();
function saveAudit(rows){
  if(!root.localStorage||typeof root.localStorage.setItem!=="function")
    throw Error("license_storage_unavailable");
  root.localStorage.setItem(AUDIT_KEY,
    JSON.stringify({version:1,rows:rows.slice(-500)}));
}
function status(value){
  const t=el("pointInputStatus");
  if(t)t.textContent=value;
}
function clearView(){
  const dir=el("directionText"),code=el("directionCode"),
    message=el("message"),conf=el("confidence"),regime=el("regime"),
    strength=el("strength"),orb=el("directionOrb"),pill=el("modePill");
  if(dir)dir.textContent="—";
  if(code)code.textContent="等待點數";
  if(message)message.textContent=
    "輸入閒、莊最終點數（0–9），再按開始分析；每次都重新建立 256 個粒子。";
  if(conf)conf.textContent="—";
  if(regime)regime.textContent="等待本局結算";
  if(strength)strength.textContent="—";
  if(orb)orb.className="direction-orb idle";
  if(pill)pill.textContent="PHYSICS 5× · ROAD 0.1× · V2.6 點數平滑";
  root.__BBB_V3_CAPITAL_UI__?.renderStake(null);
}
function render(report,pair){
  // One weak EV signal may be marked as display-only SKIP.
  // The physical EV, independent PF and official BET gate are unchanged.
  const evB=report.ev_banker,evP=report.ev_player;
  if(!Number.isFinite(evB)||!Number.isFinite(evP)||
    report.physical_ev?.simulation_exhausted)throw Error("physical_ev_unavailable");
  const display=signalGuard.decide(report);
  const dir=display.direction,skip=dir==="SKIP";
  const research=report.research_signal;
  const qualified=report.action==="BET";
  const signal=research.available;
  const text=el("directionText"),code=el("directionCode"),
    conf=el("confidence"),regime=el("regime"),strength=el("strength"),
    orb=el("directionOrb"),pill=el("modePill"),message=el("message");
  if(text)text.textContent=skip?"觀望":dir==="B"?"莊":"閒";
  if(code)code.textContent=skip?"SKIP / LOW SIGNAL":
    (dir==="B"?"BANKER":"PLAYER")+
    (qualified?" / EV VERIFIED":" / EV COMPARISON");
  if(conf)conf.textContent="模擬 EV 偏移 "+
    (report.ev_gap_deviation*100).toFixed(2)+"%（非勝率）";
  if(regime)regime.textContent="單局獨立模擬｜"+
    (skip?"EV 差距較小，觀望":"顯示模擬 EV 較高方向")+
    (signal?"｜原研究門檻已通過":"｜原研究門檻未通過")+
    (qualified?"｜實測 EV 通過":"｜未確認正 EV");
  if(strength)strength.textContent="門檻 "+(report.dynamic_gate*100).toFixed(2)+
    "%｜莊 EV "+(evB*100).toFixed(2)+
    "%｜閒 EV "+(evP*100).toFixed(2)+"%";
  if(orb)orb.className="direction-orb "+(skip?"idle":dir==="B"?"banker":"player");
  if(pill)pill.textContent="PHYSICS 5× · ROAD 0.1× · V2.6 點數平滑";
  const name=resultOf(pair)==="B"?"莊":resultOf(pair)==="P"?"閒":"和";
  if(message)message.textContent=
    "已結算：閒 "+pair.player+"／莊 "+pair.banker+"（"+name+"）"+
    "｜本次全新 256 粒子，L384／M160"+
    "｜下一局顯示訊號："+(skip?"觀望":dir==="B"?"莊":"閒")+
    "｜模擬莊 EV "+(evB*100).toFixed(2)+"%"+
    "、閒 EV "+(evP*100).toFixed(2)+"%"+
    "｜原研究訊號門檻 "+(signal?"通過":"未通過")+
    "｜最終點數不等於實際牌張，較高 EV 不代表正 EV 或較高實際勝率";
  // Suppress stake suggestions when quality display guard abstains.
  const stakeReport=skip?{...report,research_signal:{...report.research_signal,available:false}}:report;
  root.__BBB_V3_CAPITAL_UI__?.renderStake(stakeReport);
}
/** Fresh PF256 with one scored hand; NEVER sync more than one outcome. */
function computeSingle(pair){
  if(!validPair(pair))throw Error("invalid_point_observation");
  const fresh=new PhysicsForecastFirstEngine(CONFIG);
  const outcome=resultOf(pair);
  const report=fresh.syncHistory([outcome],
    [{player:pair.player,banker:pair.banker}]);
  if(fresh.history.length!==1||report.history_length!==1||
    report.point_score_observation_count!==1||
    fresh.quant.physics.size!==256)throw Error("independent_particle_contract_failed");
  if(!Number.isFinite(report.ev_banker)||!Number.isFinite(report.ev_player)||
    report.physical_ev?.simulation_exhausted)throw Error("physical_ev_unavailable");
  return{fresh,report,outcome};
}
function submitPoints(player,banker){
  if(busy)return null;
  const pair={player,banker};
  if(!validPair(pair))throw Error("invalid_point_observation");
  busy=true;
  try{
    const {fresh,report,outcome}=computeSingle(pair);
    const next=[...audit,pair].slice(-500);
    saveAudit(next);
    // Only after successful score persistence replace the active model.
    audit=next;model=fresh;lastReport=report;
    render(report,pair);
    status("本次 "+(outcome==="B"?"莊":outcome==="P"?"閒":"和")+
      "局已獨立分析；下一局可直接輸入新點數（不累積前局粒子）。");
    return outcome;
  }finally{busy=false;}
}
function analyzeEnteredPoints(){
  const p=el("pointPlayerFinal"),b=el("pointBankerFinal");
  if(!p||!b)return null;
  const player=String(p.value??"").trim(),banker=String(b.value??"").trim();
  // Check strings BEFORE Number conversion: empty/whitespace must not become 0.
  if(!/^[0-9]$/.test(player)||!/^[0-9]$/.test(banker)){
    status("請在左閒、右莊各輸入一個整數點數（0–9）。");
    return null;
  }
  try{
    const outcome=submitPoints(Number(player),Number(banker));
    if(outcome!==null){p.value="";b.value="";}
    return outcome;
  }catch(error){
    status("點數分析失敗："+String(error?.message||error));
    const m=el("message");
    if(m)m.textContent="本次分析未完成；請確認點數及本機儲存。";
    return null;
  }
}
function isMobileKeyboard(){
  return root.matchMedia?.("(pointer: coarse)")?.matches===true||
    Boolean(root.navigator?.maxTouchPoints>0&&root.innerWidth<=820);
}
function restorePrevious(){
  if(busy)return null;
  busy=true;
  try{
    const next=audit.slice(0,-1);
    // Undo affects only V2.4 score audit, not historical B/P/T license data.
    if(!audit.length){status("沒有可返回的點數記錄");return null;}
    let restored=null;
    if(next.length)restored=computeSingle(next[next.length-1]);
    saveAudit(next);
    audit=next;
    if(restored){
      model=restored.fresh;lastReport=restored.report;
      render(restored.report,next[next.length-1]);
      status("已返回上一筆點數；該筆重新使用全新的 256 粒子模擬。");
      return restored.report;
    }
    model=new PhysicsForecastFirstEngine(CONFIG);lastReport=null;
    clearView();status("已清空本次分析；可直接輸入下一局點數。");
    return null;
  }finally{busy=false;}
}
function run(){
  // Re-evaluating the last score is also independent: no history is replayed.
  if(!audit.length){lastReport=null;clearView();return null;}
  const pair=audit[audit.length-1],r=computeSingle(pair);
  model=r.fresh;lastReport=r.report;render(r.report,pair);
  return r.report;
}
const analyze=el("btnPointAnalyze");
if(analyze)analyze.addEventListener("click",()=>analyzeEnteredPoints(),false);
for(const id of ["pointPlayerFinal","pointBankerFinal"]){
  const field=el(id);
  if(field)field.addEventListener("keydown",event=>{
    if(event.key!=="Enter")return;
    event.preventDefault();
    // Desktop Enter runs analysis; mobile users explicitly press the button.
    if(isMobileKeyboard()){
      status("手機請點選「開始分析」按鈕。");return;
    }
    analyzeEnteredPoints();
  },false);
}
const back=el("btnBack");
if(back)back.addEventListener("click",event=>{
  event?.preventDefault?.();event?.stopImmediatePropagation?.();
  try{restorePrevious();}
  catch(error){status("返回失敗："+String(error?.message||error));}
},true);
const start=el("btnStart");
if(start)start.addEventListener("click",event=>{
  event?.preventDefault?.();event?.stopImmediatePropagation?.();
},true);
const end=el("btnEnd");
if(end)end.addEventListener("click",()=>{
  model.reset();lastReport=null;audit=[];
  try{root.localStorage?.removeItem(AUDIT_KEY);}catch(_){}
  clearView();
},false);
root.__BBB_QUANT256_ONLINE_PANEL__={
  get model(){return model;},run,
  history(){return model.history.slice();},
  pointScores(){return model.scoreRows.slice();},
  submitPoints,analyzeEnteredPoints,restorePrevious,
  get pointMode(){return true;},
  get independentCount(){return audit.length;},
  get latestReport(){return lastReport;},
  version:VERSION,configuration:{...CONFIG},
  setObservedRounds(rows){
    if(rows!==null)throw Error("web_point_only_mode");
    model.setObservedRounds(null);return lastReport;
  },
  wager_recommendation_validated:false,validated_edge:false
};
// Loading/reloading a licensed session must NOT auto-run the last analysis.
// The score audit is for undo only; the user explicitly starts new predictions.
clearView();
status("輸入閒、莊點數後按開始分析；電腦也可按 Enter。");
})(typeof window!=="undefined"?window:null);
