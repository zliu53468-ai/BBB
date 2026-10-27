/* History-first, then automatic score input; loaded only after verified license.
 * Authoritative Frozen Core button clicks are preserved (including settlement,
 * macro point telemetry and per-hand smoothing) and never replaced.
 * This UI deliberately does not alter Physics/PF weights or formal bet policy.
 */
(() => {
"use strict";
const PHASE_KEY="bgs_score_entry_phase_v2";
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const el=id=>document.getElementById(id);
const resultPanel=el("resultModePanel"),pointPanel=el("pointModePanel");
const stageTitle=el("entryStageTitle"),stageNote=el("entryStageNote");
const banker=el("macroBankerScore"),player=el("macroPlayerScore");
const preview=el("pointResultPreview"),start=el("btnStart");
if(!resultPanel||!pointPanel||!banker||!player||!preview||!start)return;
const parseScore=value=>/^[0-9]$/.test(String(value))?Number(value):null;
const deriveOutcome=(b,p)=>{
  const bankerScore=parseScore(b),playerScore=parseScore(p);
  return bankerScore===null||playerScore===null?null:
    bankerScore>playerScore?"B":bankerScore<playerScore?"P":"T";
};
const readHistory=()=>{
  try{
    const saved=JSON.parse(localStorage.getItem(HISTORY_KEY)||"null");
    return Array.isArray(saved?.history)?saved.history.filter(x=>["B","P","T"].includes(x)):[];
  }catch(_){return[];}
};
const clear=()=>{banker.value="";player.value="";};
const note=(text,error=false)=>{
  preview.textContent=text;
  preview.classList.toggle("is-error",!!error);
};
let phase="history",submitting=false;
const savePhase=()=>{
  // Managed bgs* key remains isolated by the existing licensed IndexedDB adapter.
  try{localStorage.setItem(PHASE_KEY,phase);}catch(_){}
};
function paint(){
  const started=phase==="points";
  resultPanel.hidden=started;
  pointPanel.hidden=!started;
  if(stageTitle)stageTitle.textContent=started?"② 開獎點數自動紀錄":"① 補齊進場前牌路";
  if(stageNote)stageNote.textContent=started?"第 "+(readHistory().length+1)+" 局：選莊、閒點數":"輸入過往莊／閒／和後，按開始分析";
  start.textContent=started?"重新分析（選用）":"開始分析";
}
function setPhase(next){
  phase=next==="points"?"points":"history";
  if(phase==="history")clear();
  savePhase();paint();
  if(phase==="points")note("選好莊、閒最終點數就會自動記錄；可按「返回上一局」更正。");
}
function recordPointHand(){
  if(phase!=="points"||submitting)return false;
  const outcome=deriveOutcome(banker.value,player.value);
  if(outcome===null)return false; // first selected field alone never commits
  const resultButton=el("btn"+outcome);
  if(!resultButton||typeof resultButton.click!=="function"){
    note("原本結果按鈕未就緒，沒有記錄。",true);return false;
  }
  submitting=true;
  try{
    const before=readHistory();
    // Reuse the existing click path: settle previous prediction -> Frozen Core
    // B/P/T history write -> macro scores/first-layer journal update.
    let clickError=null;
    try{resultButton.click();}catch(error){clickError=error;}
    const after=readHistory();
    if(after.length!==before.length+1||after.at(-1)!==outcome||
       !before.every((value,i)=>value===after[i])){
      note("本局未確認寫入成功，請檢查上方提示；請勿重複輸入。",true);
      return false;
    }
    clear();
    paint();
    const success="已自動記錄第 "+after.length+" 局："+(outcome==="B"?"莊勝":outcome==="P"?"閒勝":"和局");
    if(clickError){note(success+"；附加處理異常，請確認紀錄。",true);return true;}
    note(success+"；正在分析下一局。");
    // This is the existing authorized start handler, not a new betting path.
    // Never report a committed hand as failed if the next forecast errors.
    try{start.click();}catch(_){note(success+"；下一局分析未完成，請按重新分析。",true);}
    return true;
  }catch(error){
    note("自動紀錄失敗，請檢查歷史紀錄與上方提示。",true);
    return false;
  }finally{submitting=false;}
}
function maybeCommit(){
  if(phase!=="points"||submitting)return;
  if(deriveOutcome(banker.value,player.value)===null){
    note("請選擇另一方點數；兩邊都有數字才會自動記錄。");
    return;
  }
  recordPointHand();
}
start.addEventListener("click",()=>{
  // The normal Start handler has already run (script loads last). No history
  // means neither the model nor the phase is allowed to advance.
  if(readHistory().length===0){paint();return;}
  if(phase==="history"){clear();setPhase("points");}
  else paint();
});
banker.addEventListener("change",maybeCommit);
player.addEventListener("change",maybeCommit);
const back=el("btnBack");if(back)back.addEventListener("click",()=>{
  clear();
  if(phase==="points"&&readHistory().length===0)setPhase("history");
  paint();
  if(phase==="points")note("已返回上一局；請重新輸入本局雙方點數。");
});
const end=el("btnEnd");if(end)end.addEventListener("click",()=>{
  clear();phase="history";
  try{localStorage.removeItem(PHASE_KEY);}catch(_){}
  paint();
});
let stored="history";
try{stored=localStorage.getItem(PHASE_KEY)||"history";}catch(_){}
phase=stored==="points"&&readHistory().length>0?"points":"history";
savePhase();paint();
if(phase==="points"){
  note("已恢復點數輸入；兩邊選完後會自動記錄。");
  // On reload the original runtime does not resume its active prediction;
  // once models are loaded, use the same Start handler to restore it.
  const api=typeof window!=="undefined"?window.__BGS_FINAL56__:null;
  if(typeof api?.loadModels==="function"){
    Promise.resolve().then(()=>api.loadModels()).then(()=>{
      if(phase==="points"&&readHistory().length>0)start.click();
    }).catch(()=>{note("模型未成功恢復，請按「重新分析（選用）」。",true);});
  }
}
if(typeof window!=="undefined")window.__BBB_POINT_INPUT__={
  deriveOutcome,recordPointHand,readHistory,getPhase:()=>phase
};
})();
