/**
 * BBB Physics Forecast First（授權成功後的主面板）。
 * B/P/T 原儲存按鈕及登入流程保持不變；Final action 必須來自物理 EV Gate。
 * 無真實卡牌資料時只顯示虛擬推估，不報稱有正期望值或進場資格。
 */
(function(root){
"use strict";
if(!root||!root.document||!root.BBBPhysicsForecastFirst||
  typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location?.search||"").get("quant256")==="frozen")return;
const {PhysicsForecastFirstEngine,VERSION}=root.BBBPhysicsForecastFirst;
const CONFIG={threshold:.015,simulationsPerParticle:64,observedSamples:12000,
  quantOptions:{lambda:.82,alpha:.15}};
const model=new PhysicsForecastFirstEngine(CONFIG);
const KEY="bgs256d_frozen_6x15_forward_v18";
const FALLBACK=["bgs256d_frozen_6x15_sensitive_v17",
  "bgs256d_frozen_6x15_bigroad_v16","bgs256d_frozen_column_linucb_v15",
  "bgs256d_128plus128_frozen_direct_tech_panel_v1"];
const el=id=>root.document.getElementById(id);
function history(){
  try{
    for(const key of [KEY,...FALLBACK]){
      const s=root.localStorage?.getItem(key);
      if(!s)continue;
      const data=JSON.parse(s);
      return Array.isArray(data?.history)?
        data.history.filter(v=>["B","P","T"].includes(v)).slice(-500):[];
    }
  }catch(_){}
  return[];
}
function render(p){
  const skip=p.action!=="BET",dir=p.direction;
  const text=el("directionText"),code=el("directionCode"),conf=el("confidence");
  const regime=el("regime"),strength=el("strength"),orb=el("directionOrb"),
    pill=el("modePill"),message=el("message");
  if(text)text.textContent=skip?"觀望":dir==="B"?"莊":"閒";
  if(code)code.textContent=skip?"PHYSICS SKIP":dir==="B"?"BANKER":"PLAYER";
  if(conf)conf.textContent="EV 差異 "+(p.ev_gap_deviation*100).toFixed(2)+"%（非勝率）";
  if(regime)regime.textContent=!p.observed_card_evidence?
    "僅 B/P/T｜無真實殘牌證據｜強制觀望":
    skip?"實際牌值 MC：EV 優勢尚未過關":"已觀測牌值：通過物理 EV 閘門";
  if(strength)strength.textContent="莊 EV "+(p.ev_banker*100).toFixed(2)+"%｜閒 EV "+
    (p.ev_player*100).toFixed(2)+"%";
  if(orb)orb.className="direction-orb "+(skip?"idle":dir==="B"?"banker":"player");
  if(pill)pill.textContent="PHYSICS FORECAST FIRST · 5× / 0.1×";
  if(message)message.textContent=(skip?"未達真實物理出手資格":"實測牌值已通過 EV 閘門")+
    "｜資料來源 "+p.card_count_source+
    "｜模擬莊EV "+(p.ev_banker*100).toFixed(2)+"%、閒EV "+
    (p.ev_player*100).toFixed(2)+"%"+
    "｜相對基準差 "+(p.ev_gap_deviation*100).toFixed(2)+"%"+
    "｜EV 門檻 "+(p.ev_threshold*100).toFixed(2)+"%"+
    "｜假設方向 "+(p.hypothetical_candidate==="B"?"莊":"閒")+
    "｜路單只占 0.1×，不可翻轉物理方向"+
    "｜僅 B/P/T 不知道真正 A~K 殘量；模擬勝負不能當成實際套利證據";
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){
    model.reset();
    if(message)message.textContent="請先輸入牌局，僅有 B/P/T 時會顯示模擬物理 EV（無真實算牌資格）";
    return null;
  }
  const result=model.syncHistory(seq);
  render(result);return result;
}
function safeRun(){
  try{return run();}
  catch(error){const m=el("message");if(m)m.textContent="物理 EV 計算錯誤："+String(error?.message||error);return null;}
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(event){
  event.preventDefault();event.stopImmediatePropagation();safeRun();
},true);
for(const id of ["btnB","btnP","btnT","btnBack"]){
  const btn=el(id);
  if(btn)btn.addEventListener("click",()=>safeRun(),false);
}
const end=el("btnEnd");if(end)end.addEventListener("click",()=>model.reset(),false);
root.__BBB_QUANT256_ONLINE_PANEL__={
  model,run,history,version:VERSION,configuration:{...CONFIG},
  setObservedRounds(rows){model.setObservedRounds(rows);return safeRun();},
  wager_recommendation_validated:false,validated_edge:false
};
})(typeof window!=="undefined"?window:null);