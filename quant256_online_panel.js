/**
 * 授權頁面 256D 線上學習介面：直接接管「開始分析」，
 * B/P/T 按鈕原始儲存及返回/結束行為保留，結算後同步模型。
 * 若 URL 含 ?quant256=frozen，可回到原本 Frozen 頁面作對照。
 */
(function(root){
"use strict";
if(!root||!root.document||!root.BBBQuant256Online||
  typeof URLSearchParams!=="function"||
  new URLSearchParams(root.location?.search||"").get("quant256")==="frozen")return;
const {OnlineLinUCB256,VERSION}=root.BBBQuant256Online;
const CONFIG={lambda:.94,confidenceThreshold:.025,minSamples:12,alpha:.15};
const model=new OnlineLinUCB256(CONFIG);
const KEY="bgs256d_frozen_6x15_forward_v18";
const FALLBACK=["bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15"];
const el=id=>root.document.getElementById(id);
function history(){
  try{
    for(const key of [KEY,...FALLBACK]){
      const s=root.localStorage?.getItem(key);
      if(!s)continue;
      const data=JSON.parse(s);
      return Array.isArray(data?.history)?data.history.filter(v=>["B","P","T"].includes(v)).slice(-500):[];
    }
  }catch(_){}
  return[];
}
function render(pred){
  const skip=!pred.bet_allowed;
  const text=el("directionText"),code=el("directionCode"),confidence=el("confidence");
  const regime=el("regime"),strength=el("strength"),orb=el("directionOrb"),mode=el("modePill"),message=el("message");
  if(text)text.textContent=skip?"觀望":pred.direction==="B"?"莊":"閒";
  if(code)code.textContent=skip?"SKIP / NO ACTION":pred.direction==="B"?"BANKER":"PLAYER";
  if(confidence)confidence.textContent="UCB 差 "+pred.delta.toFixed(4)+"（非勝率）";
  if(regime)regime.textContent=skip?
    pred.reason==="warmup"?"資料暖機中，強制觀望":"差距不足，強制觀望":
    "通過模型差距閘門（未校準）";
  if(strength)strength.textContent="閘門 "+pred.confidence_threshold.toFixed(4)+"｜學習 "+pred.trained_rounds+" 局";
  if(orb)orb.className="direction-orb "+(skip?"idle":pred.direction==="B"?"banker":"player");
  if(mode)mode.textContent="256D Online · λ="+pred.lambda.toFixed(2);
  if(message)message.textContent="線上 LinUCB｜UCB 莊 "+pred.ucb_banker.toFixed(3)+
    "／閒 "+pred.ucb_player.toFixed(3)+
    "｜3/8 局 Entropy 差 "+pred.features.entropy_delta.toFixed(3)+
    "｜Regularity 差 "+pred.features.regularity_delta.toFixed(3)+
    "｜盲眼物理：僅虛擬抽樣（非實際牌值）"+
    "｜既有觀望判斷只代表模型分數，未經真實樣本校準或證明有投注優勢";
}
function run(){
  const seq=history(),message=el("message");
  if(!seq.length){
    model.reset();
    if(message)message.textContent="請先輸入莊／閒／和結果";
    return null;
  }
  const prediction=model.syncHistory(seq);
  render(prediction);
  return prediction;
}
function safeRun(){
  try{return run();}
  catch(error){const m=el("message");if(m)m.textContent="256D 模型錯誤："+String(error?.message||error);return null;}
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(event){
  // 禁止同一個點擊再次執行 Frozen Direct 的 click callback。
  event.preventDefault();event.stopImmediatePropagation();
  safeRun();
},true);
for(const id of ["btnB","btnP","btnT","btnBack"]){
  const btn=el(id);
  if(btn)btn.addEventListener("click",()=>{safeRun();},false);
}
const end=el("btnEnd");
if(end)end.addEventListener("click",()=>model.reset(),false);
root.__BBB_QUANT256_ONLINE_PANEL__={
  model,run,history,version:VERSION,configuration:{...CONFIG},
  // 僅供正式驗證/自動測試使用；不提供保證勝率。
  validated_edge:false
};
})(typeof window!=="undefined"?window:null);