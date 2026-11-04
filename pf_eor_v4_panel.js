/**
 * BBB V4 正式網頁：每局只有閒、莊最終總點 0–9；每次獨立建立 PF。
 * 歷史僅供「返回上一筆」與 Markov 旁路診斷，從不進入方向或觀望門檻。
 */
(function(root){
 "use strict";
 if(!root?.document||!root.BBBPhysicalV4)throw Error("physical_v4_ui_dependency_missing");
 const {PhysicalPointV4,CONFIG,VERSION}=root.BBBPhysicalV4;
 const KEY="bgs_v24_independent_point_audit";
 const el=id=>root.document.getElementById(id);
 const valid=x=>Number.isInteger(x)&&x>=0&&x<=9;
 const isPair=x=>x&&valid(x.player)&&valid(x.banker);
 const outcome=x=>x.banker>x.player?"B":x.player>x.banker?"P":"T";
 const pct=n=>Number.isFinite(n)?(n*100).toFixed(3)+"%":"—";
 const engine=new PhysicalPointV4();
 let busy=false,last=null,audit=[];
 const set=(id,text)=>{if(el(id))el(id).textContent=text;};
 function restoreAudit(){
  try{
   const obj=JSON.parse(root.localStorage?.getItem(KEY)||"null");
   if(obj?.version!==1||!Array.isArray(obj.rows))return[];
   return obj.rows.filter(isPair).slice(-500).map(x=>({player:x.player,banker:x.banker}));
  }catch(_){return[];}
 }
 audit=restoreAudit();
 function persist(rows){
  if(!root.localStorage?.setItem)throw Error("license_storage_not_ready");
  root.localStorage.setItem(KEY,JSON.stringify({version:1,rows:rows.slice(-500)}));
 }
 function markovDiagnostics(){
  const stats=root.BBBMarkovResidual?.transitionStats?.(audit.map(outcome));
  if(!stats)return"Markov 診斷不可用";
  return"Markov 僅旁路紀錄："+(stats.current_side||"—")+
   "｜歷史非和局 "+stats.non_tie_rounds+
   "｜同狀態後續樣本 "+stats.selected_support+
   "（不影響方向）";
 }
 function setIdle(message){
  set("directionText","—");set("directionCode","等待分析");
  set("confidence","—");set("strength","—");
  set("regime","等待輸入總點數");
  set("message",message);
  if(el("directionOrb"))el("directionOrb").className="direction-orb idle";
 }
 function render(report,pair){
  const choice=report.v4,direction=choice.direction;
  set("directionText",direction==="SKIP"?"觀望":direction==="B"?"莊":"閒");
  set("directionCode",direction==="SKIP"?"V4 / SKIP":direction==="B"?"PF / BANKER":"PF / PLAYER");
  set("confidence","相對基準 EV 訊號 "+pct(choice.relative_ev_signal)+"（非勝率）");
  set("regime",choice.reason==="quality_pass"?"V4 品質通過":choice.reason==="weak_ev_gap"?
   "V4 觀望：EV 差距偏小":choice.reason==="opposite_deplete_direction"?
   "V4 觀望：Deplete 方向衝突":choice.reason==="deplete_disagreement"?
   "V4 觀望：Deplete 差異過大":"V4 觀望：品質資料不足");
  const p=report.simulated_probability;
  set("strength","下一局模擬｜莊 "+pct(p.B)+"｜閒 "+pct(p.P)+"｜和 "+pct(p.T));
  if(el("directionOrb"))el("directionOrb").className="direction-orb "+
   (direction==="SKIP"?"idle":direction==="B"?"banker":"player");
  const deplete=report.deplete_mc;
  set("message","本局閒 "+pair.player+"／莊 "+pair.banker+
   "｜PF96（每次全新）、每粒子 MC160"+
   "｜莊 EV "+pct(report.ev_banker)+"、閒 EV "+pct(report.ev_player)+
   "｜相對八副牌基準訊號 "+pct(choice.relative_ev_signal)+
   "｜自適應平滑 "+pct(report.adaptive_score_smoothing.blend)+
   "｜V4 門檻 "+pct(choice.threshold)+
   "｜EOR 最大敏感度 "+pct(report.eor.max_abs_ev_gap_delta)+
   "｜Deplete "+(deplete.status==="complete"?
     "交叉核對 "+(choice.deplete_consistent?"一致":"有分歧")+
     "（"+deplete.samples+" 次、未融合）":"尚未完成")+
   "｜假想組成不是實際牌值；訊號不代表已驗證盈利優勢。");
  set("localCalibrationStatus","獨立點數 PF → EOR → 組成機率／EV → Deplete MC 交叉檢查 → V4"+
   "｜"+markovDiagnostics());
  set("modePill","PF96 · 自適應平滑 · EOR · V4");
 }
 function saveAndRender(pair,{undo=false}={}){
  if(busy)return null;
  if(!isPair(pair))throw Error("total_points_must_be_0_to_9");
  busy=true;
  const btn=el("btnPointAnalyze");if(btn)btn.disabled=true;
  try{
   // 先算後存；任一階段失敗時不覆寫已確認的紀錄或方向。
   const r=engine.analyze(pair.player,pair.banker);
   const next=undo?audit.slice(0,-1):[...audit,pair].slice(-500);
   persist(next);audit=next;last=r;render(r,pair);
   set("pointInputStatus","已完成一次獨立分析，僅需輸入下一局雙方最終總點（0–9）。");
   return r;
  }catch(e){
   last=null;setIdle("分析未完成：獨立組成估計或資料儲存失敗，未產生方向。");
   set("pointInputStatus","本次觀望，請檢查輸入／授權儲存："+String(e.message||e));
   return null;
  }finally{
   busy=false;if(btn)btn.disabled=false;
  }
 }
 function submitPoints(player,banker){return saveAndRender({player,banker});}
 function analyzeEnteredPoints(){
  const p=String(el("pointPlayerFinal")?.value??"").trim(),
   b=String(el("pointBankerFinal")?.value??"").trim();
  if(!/^[0-9]$/.test(p)||!/^[0-9]$/.test(b)){
   set("pointInputStatus","閒／莊只能各輸入一個 0–9 最終總點。");return null;
  }
  const r=submitPoints(Number(p),Number(b));
  if(r){el("pointPlayerFinal").value="";el("pointBankerFinal").value="";}
  return r;
 }
 function restorePrevious(){
  if(busy||!audit.length){set("pointInputStatus","已無可返回的點數紀錄。");return null;}
  if(audit.length===1){
   persist([]);audit=[];last=null;setIdle("已返回至空白分析。");
   set("localCalibrationStatus",markovDiagnostics());return null;
  }
  return saveAndRender(audit[audit.length-2],{undo:true});
 }
 for(const id of ["btnB","btnP","btnT","btnStart","btnNoSkipTest"])
  if(el(id))el(id).hidden=true;
 root.document.querySelectorAll(".bbb-results-block").forEach(x=>x.hidden=true);
 if(el("forecastGap"))el("forecastGap").hidden=true;
 const label=root.document.querySelector('label[for="forecastGap"]');
 if(label)label.hidden=true;
 if(el("drawLengthDiagnostics"))el("drawLengthDiagnostics").hidden=true;
 el("btnPointAnalyze")?.addEventListener("click",e=>{
  e.preventDefault();analyzeEnteredPoints();
 });
 for(const id of ["pointPlayerFinal","pointBankerFinal"])el(id)?.addEventListener("keydown",e=>{
  if(e.key!=="Enter")return;e.preventDefault();
  if(root.matchMedia?.("(pointer: coarse)")?.matches){
   set("pointInputStatus","手機請按「開始分析」。");return;
  }
  analyzeEnteredPoints();
 });
 el("btnBack")?.addEventListener("click",e=>{
  e.preventDefault();e.stopImmediatePropagation();restorePrevious();
 },true);
 el("btnEnd")?.addEventListener("click",()=>{
  audit=[];last=null;setIdle("正在結束分析與清除紀錄…");
 },false);
 for(const id of ["btnExportTraining","btnExportObservations"])
  el(id)?.addEventListener("click",()=>{
   const value=JSON.stringify({version:1,rows:audit},null,2);
   const url=URL.createObjectURL(new Blob([value],{type:"application/json"}));
   const a=root.document.createElement("a");
   a.href=url;a.download="BBB-V4-point-audit.json";a.click();
   setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
 root.__BBB_PHYSICAL_V4_PANEL__={
  version:VERSION,configuration:CONFIG,submitPoints,analyzeEnteredPoints,
  restorePrevious,get latestReport(){return last;},
  get audit(){return audit.map(x=>({...x}));},
  get markov(){return root.BBBMarkovResidual?.transitionStats?.(audit.map(outcome))||null;}
 };
 setIdle("只輸入本局閒／莊最終點數各一個 0–9；每次重新建立 PF，不使用歷史改變方向。");
 set("modePill","PF96 · 自適應平滑 · EOR · V4");
 set("pointInputStatus","已記錄 "+audit.length+" 次總點數（僅供返回／Markov 診斷）；可開始新的獨立分析。");
 set("localCalibrationStatus",markovDiagnostics());
})(typeof window!=="undefined"?window:null);
