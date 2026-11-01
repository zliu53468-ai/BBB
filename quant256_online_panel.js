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
const CONFIG={lambda:.82,confidenceThreshold:0,lossGate:.04,minSamples:0,alpha:.15};
const model=new OnlineLinUCB256(CONFIG);
const KEY="bgs256d_frozen_6x15_forward_v18";
const FALLBACK=["bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15",
  "bgs256d_128plus128_frozen_direct_tech_panel_v1"];
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
  if(code)code.textContent=skip?"LOSS GATE / SKIP":pred.direction==="B"?"BANKER":"PLAYER";
  if(confidence)confidence.textContent="UCB 差 "+pred.delta.toFixed(4)+"（非勝率）";
  if(regime)regime.textContent=skip?"上一局失誤，短暫觀望（0.04）":
    pred.reason==="neutral_default_signal"?"初始中立訊號（非預測優勢）":
    "極短窗即時方向（未校準）";
  if(strength)strength.textContent="動態閘門 "+pred.dynamic_gate.toFixed(3)+
    "｜有效出手 "+pred.prequential.eligible+" 局";
  if(orb)orb.className="direction-orb "+(skip?"idle":pred.direction==="B"?"banker":"player");
  if(mode)mode.textContent="256D Hyper Micro · λ="+pred.lambda.toFixed(2);
  const hit=pred.hit_rate===null?"樣本不足":(pred.hit_rate*100).toFixed(1)+"%";
  const rate=pred.signal_rate===null?"—":(pred.signal_rate*100).toFixed(1)+"%";
  if(message)message.textContent="實驗性高頻 LinUCB｜莊分 "+pred.ucb_banker.toFixed(3)+
    "／閒分 "+pred.ucb_player.toFixed(3)+
    "｜2/5 Turn Δ "+pred.features.turn_rate_delta.toFixed(3)+
    "｜Regularity Δ "+pred.features.regularity_delta.toFixed(3)+
    "｜上局結果 "+pred.last_settlement+
    "｜前推命中率 "+hit+"／出手密度 "+rate+
    "｜盲眼物理僅為虛擬抽樣"+
    "｜模型分數不是真實勝率，連勝不代表下一局較容易獲勝";
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