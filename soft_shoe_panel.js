/** B/P/T-only panel: retains existing licensed shell, result controls and exports. */
(function(root){
 "use strict";
 if(!root||!root.document||!root.BBBSoftShoe||!root.BBBBigRoad)
  throw Error("soft_shoe_ui_dependencies_missing");
 const core=root.BBBSoftShoe,KEY="bgs_soft_shoe_bpt_v1",VERSION=core.VERSION;
 const $=id=>root.document.getElementById(id);
 const set=(id,x)=>{if($(id))$(id).textContent=x;};
 const pct=x=>Number.isFinite(x)?(100*x).toFixed(2)+"%":"—";
 const syncBigRoad=()=>root.BBBBigRoad.render($("bptBigRoad"),history,$("bptRoundCounter"));
 const old=root.localStorage?.getItem(KEY);
 let history=[],busy=false,worker=null,request=0,last=null;
 try{
  const v=JSON.parse(old||"null");
  if(v?.version===1&&Array.isArray(v.history)&&
     v.history.every(x=>["B","P","T"].includes(x))&&v.history.length<=72)
   history=v.history.slice();
 }catch(_){}
 function persist(){
  if(!root.localStorage?.setItem)throw Error("license_storage_not_ready");
  root.localStorage.setItem(KEY,JSON.stringify({version:1,history}));
 }
 function uiIdle(message){
  set("directionText","—");set("directionCode","等待分析");
  set("confidence","—");set("regime","等待輸入");set("strength","—");
  set("message",message);$("directionOrb").className="direction-orb idle";
  set("localCalibrationStatus","B/P/T 軟算牌 · 目前已記錄 "+history.length+" 局（同一牌靴）");
 }
 function show(report){
  // Match the LINE Soft Shoe adapter: use the quality-gated research direction.
  // The unqualified candidate remains diagnostic-only and never becomes a BET.
  last=report;const view=report.research_view||{},
   qualified=view.eligible===true,
   d=["B","P"].includes(view.direction)?view.direction:"SKIP",
   q=report.probabilities||{};
  set("directionText",d==="B"?"莊":d==="P"?"閒":"觀望");
  set("directionCode",d==="B"?"研究方向 / BANKER":d==="P"?"研究方向 / PLAYER":"品質觀望 / SKIP");
  $("directionOrb").className="direction-orb "+(d==="B"?"banker":d==="P"?"player":"idle");
  set("confidence","莊 EV "+pct(report.ev_banker)+"／閒 EV "+pct(report.ev_player)+
   "｜置信指標 "+pct(report.confidence));
  const reasonNames={
   insufficient_rounds:"觀望：樣本局數不足",
   weak_ev_separation:"觀望：莊閒 EV 差距不足",
   high_estimator_uncertainty:"觀望：EV 差異估計誤差過大",
   low_gap_to_noise:"觀望：方向訊號未超過噪音門檻",
   low_particle_ess:"觀望：有效粒子不足",
   low_particle_diversity:"觀望：粒子多樣性不足"
  };
  set("regime",qualified?"品質篩選通過（研究方向，非下注建議）":
   reasonNames[view.reason]||"觀望：方向品質未達門檻");
  set("strength","下一局模型估計：莊 "+pct(q.B)+"｜閒 "+pct(q.P)+"｜和 "+pct(q.T));
  set("message","已輸入 "+report.observations+" 局｜候選方向 "+(report.candidate==="B"?"莊":"閒")+
   "｜估計 EV 差距 "+pct(view.abs_ev_gap)+
   "｜差距標準誤 "+pct(view.ev_gap_se)+
   "｜差距／噪音 "+(Number(view.ev_gap_snr)||0).toFixed(2)+"×"+
   "｜後驗 ESS "+pct(view.ess_ratio)+
   "｜後驗多樣性 "+pct(view.ancestor_diversity)+
   "｜保守 EV 下界 "+pct(report.conservative_lower_ev)+
   "｜估計誤差（莊 "+pct(report.se_banker)+"／閒 "+pct(report.se_player)+"）"+
   "｜模型粒子 "+core.DEFAULT.PARTICLE_COUNT+
   "、似然抽樣 "+core.DEFAULT.LIKELIHOOD_DRAWS+
   "、預測抽樣 "+core.DEFAULT.NEXT_HAND_DRAWS+
   "｜有效粒子 "+Number(report.particle_ess||0).toFixed(1)+
   "｜合格下注比例 "+pct(report.bankroll_fraction)+
   "｜品質狀態 "+(qualified?"通過研究篩選":"未通過研究篩選（顯示觀望）")+
   "｜候選方向不代表下注建議；只有 B/P/T，實際發牌與殘牌未知；未驗證獲利優勢。");
  set("modePill","Soft Shoe · LINE Soft Shoe 觀望規則");
  set("localCalibrationStatus","48 維物理特徵＋Frozen 序列診斷＋XGBoost 安全閘門"+
   "｜模型："+String(report.model||"不可用"));
 }
 function compute(h){
  if(!root.Worker)return Promise.reject(Error("worker_unavailable"));
  const id=++request;
  return new Promise((resolve,reject)=>{
   const w=new root.Worker("soft_shoe_worker.js?v=20261108-pf448-l80-m128-v8");worker=w;
   const close=()=>{clearTimeout(timer);w.terminate();if(worker===w)worker=null;};
   const timer=setTimeout(()=>{close();reject(Error("worker_timeout"));},15000);
   w.onerror=()=>{close();reject(Error("worker_execution_failed"));};
   w.onmessage=e=>{
    if(e.data?.id!==id)return;
    close();
    if(e.data.error)reject(Error(e.data.error));
    else if(e.data.report?.version!==VERSION)reject(Error("worker_version_mismatch"));
    else resolve(e.data.report);
   };
   w.postMessage({id,history:h});
  });
 }
 async function recalculate(){
  if(busy)return;busy=true;set("directionText","計算中");
  const controls=["btnB","btnP","btnT","btnStart","btnBack"].map($).filter(Boolean);
  controls.forEach(x=>x.disabled=true);
  set("message","正在計算方向與 EV；品質未通過時顯示觀望。");
  try{
   const report=await compute(history.slice());
   if(!report.research_view||!["B","P","SKIP"].includes(report.research_view.direction)||
      !["B","P"].includes(report.candidate)){
    last=report;uiIdle("分析未完成："+String(report.reason));set("directionText","無法分析");
   }else show(report);
   set("pointInputStatus","同一牌靴已記錄 "+history.length+" 局，輸入下一局結果。");
  }catch(e){
   last=null;uiIdle("本次分析失敗，未產生方向："+String(e?.message||e));
   set("directionText","無法分析");
  }finally{busy=false;controls.forEach(x=>x.disabled=false);}
 }
 function add(outcome){
  if(busy||!["B","P","T"].includes(outcome))return;
  if(history.length>=core.DEFAULT.MAX_HANDS){
   set("message","達到預設牌靴深度上限 72 局；請結束分析後重新開始新靴。");return;
  }
  const before=history.slice();history.push(outcome);
  try{persist();}catch(e){
   history=before;set("message","紀錄未成功儲存，已拒絕預測："+String(e.message||e));return;
  }
  syncBigRoad();
  recalculate();
 }
 function back(){
  if(busy||!history.length)return;
  const before=history.slice();history.pop();
  try{persist();}catch(e){history=before;set("message","無法保存返回操作，未修改紀錄。");return;}
  syncBigRoad();
  recalculate();
 }
 function exportHistory(){
  const data=JSON.stringify({version:1,source:"BPT_only",history},null,2);
  const url=URL.createObjectURL(new Blob([data],{type:"application/json"}));
  const anchor=root.document.createElement("a");anchor.href=url;
  anchor.download="BBB-BPT-observations.json";anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
 for(const [id,side] of [["btnB","B"],["btnP","P"],["btnT","T"]]){
  const btn=$(id);if(btn){btn.hidden=false;btn.addEventListener("click",()=>add(side));}
 }
 $("btnStart")?.addEventListener("click",()=>recalculate());
 $("btnBack")?.addEventListener("click",()=>back());
 $("btnEnd")?.addEventListener("click",()=>{
  request++;worker?.terminate();worker=null;busy=true;last=null;
  set("directionText","—");
 });
 for(const id of ["btnExportTraining","btnExportObservations"])
  $(id)?.addEventListener("click",exportHistory);
 const point=$("bbbPointEntry");if(point)point.hidden=true;
 for(const id of ["btnPointAnalyze","btnNoSkipTest","forecastGap"])
  if($(id))$(id).hidden=true;
 const label=root.document.querySelector('label[for="forecastGap"]');
 if(label)label.hidden=true;
 root.__BBB_SOFT_SHOE_PANEL__={
  version:VERSION,add,back,recalculate,
  get history(){return history.slice();},get latestReport(){return last;}
 };
 syncBigRoad();
 uiIdle("點選「莊／閒／和」記錄本局結果，再估計下一局；每局觀測會累積在同一牌靴。");
 set("modePill","Soft Shoe · 適度觀望");
 set("pointInputStatus","已記錄 "+history.length+" 局；更換牌靴請按「結束分析」。");
})(typeof window!=="undefined"?window:null);
