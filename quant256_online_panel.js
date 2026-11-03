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
const POINT_KEY="bgs_v23_point_score_log",STAGE_KEY="bgs_v23_point_entry_mode";
let scoreStage=false,scoreBusy=false;
function validScore(pair,outcome){
  return Boolean(pair&&Number.isInteger(pair.player)&&Number.isInteger(pair.banker)&&
    pair.player>=0&&pair.player<=9&&pair.banker>=0&&pair.banker<=9&&
    (pair.banker>pair.player?"B":pair.player>pair.banker?"P":"T")===outcome);
}
function pointScores(seq){
  const rows=Array(seq.length).fill(null);
  try{
    const data=JSON.parse(root.localStorage?.getItem(POINT_KEY)||"null");
    if(!Array.isArray(data?.history)||!Array.isArray(data?.scores))return rows;
    for(let i=0;i<seq.length&&i<data.history.length;i++){
      if(data.history[i]!==seq[i])break;
      const pair=data.scores[i];
      if(pair===null||pair===undefined)continue;
      if(!validScore(pair,seq[i]))break;
      rows[i]={player:pair.player,banker:pair.banker};
    }
  }catch(_){}
  return rows;
}
function storeScores(seq,rows){
  root.localStorage.setItem(POINT_KEY,JSON.stringify({history:seq,scores:rows}));
}
function setStage(on){
  scoreStage=Boolean(on);
  const app=el("predictionApp"),entry=el("bbbPointEntry");
  if(app?.classList?.toggle)app.classList.toggle("bbb-point-mode",scoreStage);
  if(entry)entry.hidden=!scoreStage;
  root.localStorage?.setItem(STAGE_KEY,scoreStage?"1":"0");
}
function pointStatus(message){
  const elStatus=el("pointInputStatus");
  if(elStatus)elStatus.textContent=message;
}
/**
 * Outcome is committed only after BOTH final totals exist. Reuse the existing
 * B/P/T button click so Frozen Core, local storage and licensed history are
 * updated once by their original handlers. The point row is stored before
 * the synthetic click, allowing our existing bubble handler to apply a
 * conditional PF observation in the same pass.
 */
function submitPoints(player,banker){
  if(!scoreStage||scoreBusy)return null;
  const p=Number(player),b=Number(banker);
  if(!Number.isInteger(p)||!Number.isInteger(b)||p<0||p>9||b<0||b>9)
    throw Error("invalid_point_observation");
  const outcome=b>p?"B":p>b?"P":"T";
  const seq=history(),button=el("btn"+outcome);
  if(!seq.length||!button||typeof button.click!=="function")
    throw Error("point_input_unavailable");
  const rows=pointScores(seq),old= root.localStorage?.getItem(POINT_KEY);
  scoreBusy=true;
  try{
    storeScores([...seq,outcome],[...rows,{player:p,banker:b}]);
    button.click();
    const updated=history();
    if(updated.length!==seq.length+1||updated.at(-1)!==outcome)
      throw Error("point_result_not_recorded");
    pointStatus("第 "+updated.length+" 局 "+(outcome==="B"?"莊":outcome==="P"?"閒":"和")+
      "勝已記錄（閒 "+p+"／莊 "+b+"）；下局已更新");
    return outcome;
  }catch(error){
    if(old===null||old===undefined)root.localStorage?.removeItem(POINT_KEY);
    else root.localStorage?.setItem(POINT_KEY,old);
    pointStatus("記錄失敗，請確認本機資料："+String(error?.message||error));
    throw error;
  }finally{scoreBusy=false;}
}
function onPointChange(){
  const p=el("pointPlayerFinal"),b=el("pointBankerFinal");
  if(!p||!b||p.value===""||b.value==="")return;
  try{submitPoints(Number(p.value),Number(b.value));}
  catch(error){const m=el("message");if(m)m.textContent="點數結算失敗："+String(error?.message||error);}
  finally{p.value="";b.value="";}
}

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
  const result=model.syncHistory(seq,pointScores(seq));
  render(result);return result;
}
function safeRun(){
  try{return run();}
  catch(error){const m=el("message");if(m)m.textContent="物理 EV 計算錯誤："+String(error?.message||error);return null;}
}
const start=el("btnStart");
if(start)start.addEventListener("click",function(event){
  event.preventDefault();event.stopImmediatePropagation();
  const result=safeRun();
  if(result){
    try{setStage(true);pointStatus("已初始化 "+result.history_length+" 局歷史，請輸入當局最終點數");}
    catch(error){const m=el("message");if(m)m.textContent="點數模式啟動失敗："+String(error?.message||error);}
  }
},true);
for(const id of ["btnB","btnP","btnT","btnBack"]){
  const btn=el(id);
  if(btn)btn.addEventListener("click",()=>{
    const seq=history();
    if(id==="btnBack"){
      try{storeScores(seq,pointScores(seq));}
      catch(error){pointStatus("返回後記錄未保存："+String(error?.message||error));}
      if(!seq.length&&scoreStage){try{setStage(false);}catch(_){}}
    }
    safeRun();
  },false);
}
for(const id of ["pointPlayerFinal","pointBankerFinal"]){
  const field=el(id);
  if(field)field.addEventListener("change",onPointChange,false);
}
const end=el("btnEnd");if(end)end.addEventListener("click",()=>{
  model.reset();root.__BBB_V3_CAPITAL_UI__?.renderStake(null);
  try{
    root.localStorage?.removeItem(POINT_KEY);
    root.localStorage?.removeItem(STAGE_KEY);
  }catch(_){}
  scoreStage=false;
},false);
root.__BBB_QUANT256_ONLINE_PANEL__={
  model,run,history,pointScores,submitPoints,
  get pointMode(){return scoreStage;},
  version:VERSION,configuration:{...CONFIG},
  setObservedRounds(rows){model.setObservedRounds(rows);return safeRun();},
  wager_recommendation_validated:false,validated_edge:false
};
try{
  if(root.localStorage?.getItem(STAGE_KEY)==="1"&&history().length){
    setStage(true);safeRun();
  }
}catch(_){} 
})(typeof window!=="undefined"?window:null);