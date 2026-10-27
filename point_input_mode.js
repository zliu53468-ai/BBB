/*
 * Point-first input. Historical B/P/T results may be pasted as a seed, while
 * each new hand is recorded from final scores. The old B/P/T buttons remain a
 * one-click compatibility mode and are disabled as the default input path.
 */
(() => {
"use strict";
const PHASE_KEY="bgs_score_entry_phase_v2";
const INPUT_MODE_KEY="bgs_score_input_mode_v1";
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const el=id=>document.getElementById(id);
const resultPanel=el("resultModePanel"),pointPanel=el("pointModePanel"),historyPanel=el("historySeedPanel");
const stageTitle=el("entryStageTitle"),stageNote=el("entryStageNote");
const banker=el("macroBankerScore"),player=el("macroPlayerScore"),cardCount=el("macroCardCount");
const preview=el("pointResultPreview"),start=el("btnStart"),historySeed=el("historySeed");
const compatibilityToggle=el("compatModeToggle"),compatibilityHint=el("compatModeHint");
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
const clear=()=>{banker.value="";player.value="";if(cardCount)cardCount.value="";};
const note=(text,error=false)=>{
  preview.textContent=text;
  preview.classList.toggle("is-error",!!error);
};
let phase="history",inputMode="points",submitting=false,inputModeStored=false,pointInputEnabled=true;
try{const savedMode=localStorage.getItem(INPUT_MODE_KEY);inputModeStored=savedMode!==null;inputMode=savedMode==="compatibility"?"compatibility":"points";}catch(_){}
const savePhase=()=>{try{localStorage.setItem(PHASE_KEY,phase);}catch(_){}};
const saveInputMode=()=>{try{localStorage.setItem(INPUT_MODE_KEY,inputMode);}catch(_){}};
function paint(){
  const started=phase==="points",compatible=inputMode==="compatibility";
  if(historyPanel)historyPanel.hidden=started;
  resultPanel.hidden=!compatible;
  pointPanel.hidden=!started||compatible;
  if(stageTitle)stageTitle.textContent=started?(compatible?"② B/P/T 相容輸入":"② 開牌點數自動紀錄"):"① 載入舊牌路或直接開始";
  if(stageNote)stageNote.textContent=started
    ?"第 "+(readHistory().length+1)+" 局："+(compatible?"按莊／閒／和記錄":"先選張數（可略過），再選莊、閒點數")
    :"可貼上過往 B/P/T 路單；新靴可直接輸入第一局點數";
  if(compatibilityToggle){
    compatibilityToggle.disabled=!pointInputEnabled;
    compatibilityToggle.textContent=!pointInputEnabled?"點數輸入已由設定停用":compatible?"切回閒／莊點數輸入":"開啟 B/P/T 相容模式";
    compatibilityToggle.setAttribute("aria-pressed",String(compatible));
  }
  if(compatibilityHint)compatibilityHint.textContent=!pointInputEnabled
    ?"點數輸入已由設定停用，目前使用 B/P/T 相容模式。"
    :compatible
    ?"目前為相容模式，使用舊 B/P/T 粒子更新；點此可切回點數條件後驗。"
    :"建議使用點數輸入以啟用準算牌；舊 B/P/T 按鈕模式預設關閉。";
}
function setPhase(next){
  phase=next==="points"?"points":"history";
  if(phase==="history")clear();
  savePhase();paint();
  if(phase==="points"&&inputMode==="points")note("選填總張數請先選，再輸入閒、莊最終點數；兩邊完成後會自動記錄並分析下一局。");
}
function importHistory(value){
  const compact=String(value||"").trim().toUpperCase().replace(/[\s,，;；、|/]+/g,"");
  if(!compact)return {ok:false,reason:"請貼上過往 B/P/T 路單，或用相容模式輸入。"};
  if(compact.length>500)return {ok:false,reason:"一次最多載入 500 局。"};
  if(!/^[BPT]+$/.test(compact))return {ok:false,reason:"路單只能包含 B、P、T；可用逗號或空格分隔。"};
  const before=readHistory();
  for(const outcome of compact){
    const button=el("btn"+outcome);
    if(!button||typeof button.click!=="function")return {ok:false,reason:"結果按鈕尚未就緒，請重新整理後再試。"};
    button.click();
  }
  const after=readHistory(),expected=[...before,...compact];
  if(after.length!==expected.length||!expected.every((value,i)=>after[i]===value))
    return {ok:false,reason:"牌路沒有完整寫入，請檢查本機紀錄後再試。"};
  if(historySeed)historySeed.value="";
  return {ok:true,count:compact.length,total:after.length};
}
function recordPointHand(){
  if(phase!=="points"||inputMode!=="points"||submitting)return false;
  const outcome=deriveOutcome(banker.value,player.value);
  if(outcome===null)return false;
  const cardValue=cardCount?.value||null;
  const validation=window.__BGS_PARTIC