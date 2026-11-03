/**
 * BBB Physics Forecast First（授權成功後的主面板）。
 * B/P/T 原儲存按鈕及登入流程保持不變；正式 BET 仍由真實殘牌 EV Gate 決定；主顯示為模擬研究方向。
 * 無真實卡牌資料時只顯示虛擬推估，不報稱有正期望值或進場資格。
 */
(function(root){
"use strict";
if(!root||!root.document||!root.BBBPhysicsForecastFirst||
  typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location?.search||"").get("quant256")==="frozen")return;
const {PhysicsForecastFirstEngine,VERSION}=root.BBBPhysicsForecastFirst;
const CONFIG={threshold:.005,lossGate:.035,simulationsPerParticle:160,
  observedSamples:12000,quantOptions:{lambda:.82,alpha:.15,particles:256}};
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
  // 即使 B/P/T 無法確認真實算牌優勢，也顯示「研究方向」而非整頁 SKIP。
  // 注意：research signal 不等於 action=BET；必須通過實測牌值 EV Gate 才能 BET。
  const research=p.research_signal,dir=research.direction;
  const qualified=p.action==="BET";
  const signal=research.available;
  const text=el("directionText"),code=el("directionCode"),conf=el("confidence");
  const regime=el("regime"),strength=el("strength"),orb=el("directionOrb"),
    pill=el("modePill"),message=el("message");
  // 維持目前手機 UI：只有莊、閒或觀望，不恢復過長的「（研究）」後綴。
  if(text)text.textContent=!signal?"觀望":dir==="B"?"莊":"閒";
  if(code)code.textContent=!signal?"SKIP / NO ACTION":
    (dir==="B"?"BANKER":"PLAYER")+
    (qualified?" / EV VERIFIED":" / SIMULATION ONLY");
  if(conf)conf.textContent="模擬 EV 偏移 "+(p.ev_gap_deviation*100).toFixed(2)+"%（非勝率）";
  const phase=p.gate_phase==="cushion"?"失誤後單局緩衝":
    p.gate_phase==="overdrive"?"連勝順風門檻 0":"基礎研究門檻";
  if(regime)regime.textContent=phase+
    (signal?"｜已通過研究訊號門檻":"｜研究差值未過門檻")+
    (qualified?"｜實測 EV 通過":"｜未確認正 EV");
  if(strength)strength.textContent="門檻 "+(p.dynamic_gate*100).toFixed(2)+
    "%｜莊 EV "+(p.ev_banker*100).toFixed(2)+
    "%｜閒 EV "+(p.ev_player*100).toFixed(2)+"%";
  if(orb)orb.className="direction-orb "+(signal?(dir==="B"?"banker":"player"):"idle");
  if(pill)pill.textContent="PHYSICS 5× · ROAD 0.1× · V2.2 MARKOV";
  const hit=p.hit_rate===null?"無有效樣本":(p.hit_rate*100).toFixed(1)+"%";
  const rate=p.signal_rate===null?"—":(p.signal_rate*100).toFixed(1)+"%";
  const markov=p.markov_residual?.stats,shadow=p.markov_residual?.shadow;
  const transition=markov?.from==="B"?"莊→莊／莊→閒":
    markov?.from==="P"?"閒→莊／閒→閒":"尚無方向";
  const posterior=Number.isFinite(markov?.posterior_conditional?.B)?
    (markov.posterior_conditional.B*100).toFixed(2)+"%":"—";
  const calibration=Number.isFinite(shadow?.conditional_shift)?
    (shadow.conditional_shift*100).toFixed(3)+"pp":"—";
  if(message)message.textContent="本局研究方向："+(dir==="B"?"莊":"閒")+
    "｜研究訊號 "+(signal?"通過":"未通過")+
    "｜實測 EV 行動："+(qualified?"BET":"未達 BET 資格")+
    "｜Gate "+(p.dynamic_gate*100).toFixed(2)+"%"+
    "｜連中 "+p.win_streak+" 局"+
    "｜上局結算 "+p.last_settlement+
    "｜研究訊號命中率 "+hit+"，訊號密度 "+rate+
    "｜模擬莊 EV "+(p.ev_banker*100).toFixed(2)+
    "%、閒 EV "+(p.ev_player*100).toFixed(2)+"%"+
    "｜馬可夫 "+transition+"，樣本 "+(markov?.selected_support??0)+
    "，莊條件後驗 "+posterior+
    "（"+(markov?.eligible?"已達統計門檻":"樣本不足")+"）"+
    "｜研究校準候選 "+calibration+"（未啟用，不影響真實 EV）"+
    "｜分布監測 "+(p.shoe_distribution_guard?.current_stage==="distribution_shift"?
      "近期比例明顯改變":p.shoe_distribution_guard?.current_stage==="insufficient_data"?
      "資料不足":"未見強烈變化")+
    "｜B/P/T 不揭露實際牌值，模擬 EV 不是可驗證的套利優勢";
  // 舊資金面板僅呈現曝險計算，不參與 e323a243 的預測決策。
  root.__BBB_V3_CAPITAL_UI__?.renderStake(p);
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){
    model.reset();root.__BBB_V3_CAPITAL_UI__?.renderStake(null);
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
const end=el("btnEnd");if(end)end.addEventListener("click",()=>{
  model.reset();root.__BBB_V3_CAPITAL_UI__?.renderStake(null);
},false);
root.__BBB_QUANT256_ONLINE_PANEL__={
  model,run,history,version:VERSION,configuration:{...CONFIG},
  setObservedRounds(rows){model.setObservedRounds(rows);return safeRun();},
  wager_recommendation_validated:false,validated_edge:false
};
})(typeof window!=="undefined"?window:null);