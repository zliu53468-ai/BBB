/*
 * Point-first input. Each completed hand is recorded from final scores. The
 * old B/P/T buttons remain a one-click compatibility mode, never a history seed.
 */
(() => {
"use strict";
const PHASE_KEY="bgs_score_entry_phase_v2";
const INPUT_MODE_KEY="bgs_score_input_mode_v1";
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const el=id=>document.getElementById(id);
const resultPanel=el("resultModePanel"),pointPanel=el("pointModePanel");
const stageTitle=el("entryStageTitle"),stageNote=el("entryStageNote");
const banker=el("macroBankerScore"),player=el("macroPlayerScore"),cardCount=el("macroCardCount");
const preview=el("pointResultPreview"),start=el("btnStart");
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
let phase="points",inputMode="points",submitting=false,inputModeStored=false,pointInputEnabled=true;
try{const savedMode=localStorage.getItem(INPUT_MODE_KEY);inputModeStored=savedMode!==null;inputMode=savedMode==="compatibility"?"compatibility":"points";}catch(_){}
const savePhase=()=>{try{localStorage.setItem(PHASE_KEY,phase);}catch(_){}};
const saveInputMode=()=>{try{localStorage.setItem(INPUT_MODE_KEY,inputMode);}catch(_){}};
function paint(){
  const started=phase==="points",compatible=inputMode==="compatibility";
  resultPanel.hidden=!compatible;
  pointPanel.hidden=!started||compatible;
  if(stageTitle)stageTitle.textContent=compatible?"B/P/T 相容輸入":"開牌點數自動紀錄";
  if(stageNote)stageNote.textContent=started
    ?"第 "+(readHistory().length+1)+" 局："+(compatible?"按莊／閒／和記錄":"先選張數（可略過），再選莊、閒點數")
    :"每局結束後輸入閒、莊最終點數；程式會自動記錄並分析下一局";
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
function setPhase(){
  phase="points";
  savePhase();paint();
  if(phase==="points"&&inputMode==="points")note("選填總張數請先選，再輸入閒、莊最終點數；兩邊完成後會自動記錄並分析下一局。");
}
function recordPointHand(){
  if(phase!=="points"||inputMode!=="points"||submitting)return false;
  const outcome=deriveOutcome(banker.value,player.value);
  if(outcome===null)return false;
  const cardValue=cardCount?.value||null;
  const validation=window.__BGS_PARTICLE500__?.validatePointObservation?.(player.value,banker.value,cardValue);
  if(validation?.valid===false){
    note("點數與總張數不符合百家樂補牌規則；請更正張數或點數。",true);return false;
  }
  const resultButton=el("btn"+outcome);
  if(!resultButton||typeof resultButton.click!=="function"){
    note("原本結果按鈕未就緒，沒有記錄。",true);return false;
  }
  submitting=true;
  try{
    const before=readHistory();
    // Reuse the existing licensed result path so settlement, local isolation,
    // per-hand telemetry, first-layer smoothing and undo stay unchanged.
    let clickError=null;
    try{resultButton.click();}catch(error){clickError=error;}
    const after=readHistory();
    if(after.length!==before.length+1||after.at(-1)!==outcome||!before.every((value,i)=>value===after[i])){
      note("本局未確認寫入成功，請檢查上方提示；請勿重複輸入。",true);
      return false;
    }
    clear();paint();
    const success="已自動記錄第 "+after.length+" 局："+(outcome==="B"?"莊勝":outcome==="P"?"閒勝":"和局");
    if(clickError){note(success+"；附加處理異常，請確認紀錄。",true);return true;}
    note(success+"；正在分析下一局。");
    try{start.click();}catch(_){note(success+"；下一局分析未完成，請按重新分析。",true);}
    return true;
  }catch(_){
    note("自動紀錄失敗，請檢查歷史紀錄與上方提示。",true);
    return false;
  }finally{submitting=false;}
}
function maybeCommit(){
  if(phase!=="points"||inputMode!=="points"||submitting)return;
  if(deriveOutcome(banker.value,player.value)===null){
    note("請選擇另一方點數；兩邊都有數字才會自動記錄。");
    return;
  }
  recordPointHand();
}
if(compatibilityToggle)compatibilityToggle.addEventListener("click",()=>{
  if(!pointInputEnabled)return;
  inputMode=inputMode==="compatibility"?"points":"compatibility";inputModeStored=true;saveInputMode();paint();
  note(inputMode==="compatibility"?"已切換為 B/P/T 相容模式；新點數紀錄不會用於點數條件 PF。":"已切回點數輸入；已記錄的點數會用於即時粒子後驗。");
});
start.addEventListener("click",()=>{
  if(readHistory().length===0){
    setPhase("points");
    note(inputMode==="compatibility"
      ?"尚未有牌局；可直接按莊／閒／和輸入第一局。"
      :"尚未有牌局；請先輸入第一局閒、莊最終點數，記錄後會分析下一局。");
    return;
  }
  phase="points";savePhase();paint();
});
banker.addEventListener("change",maybeCommit);
player.addEventListener("change",maybeCommit);
for(const id of ["btnB","btnP","btnT"]){
  const button=el(id);if(!button)continue;
  button.addEventListener("click",()=>{
    if(phase==="points"&&inputMode==="compatibility"&&!submitting){
      paint();
      // The old B/P/T mode still forecasts after each completed hand.
      start.click();
    }
  });
}
const back=el("btnBack");if(back)back.addEventListener("click",()=>{
  clear();
  phase="points";savePhase();
  paint();
  if(phase==="points"&&inputMode==="points")note("已返回上一局；請重新輸入本局張數與雙方點數。");
});
const end=el("btnEnd");if(end)end.addEventListener("click",()=>{
  clear();phase="points";savePhase();
  paint();
});
phase="points";
savePhase();paint();
if(phase==="points"){
  note(inputMode==="compatibility"?"已恢復 B/P/T 相容輸入。":"已恢復點數輸入；雙方完成後會自動記錄。");
  const api=typeof window!=="undefined"?window.__BGS_FINAL56__:null;
  if(typeof api?.loadModels==="function"){
    Promise.resolve().then(()=>api.loadModels()).then(()=>{
      if(phase==="points"&&readHistory().length>0)start.click();
    }).catch(()=>{note("模型未成功恢復，請按「重新分析（選用）」。",true);});
  }
}
if(typeof window!=="undefined")window.__BBB_POINT_INPUT__={
  deriveOutcome,recordPointHand,readHistory,getPhase:()=>phase,
  getInputMode:()=>inputMode,setInputMode:value=>{
    inputMode=!pointInputEnabled||value==="compatibility"?"compatibility":"points";inputModeStored=true;saveInputMode();paint();return inputMode;
  },applyProfileDefaults:config=>{
    const pointConfig=config?.point_input||config;
    pointInputEnabled=!!config&&config.enabled!==false&&pointConfig?.enabled!==false;
    if(!pointInputEnabled)inputMode="compatibility";
    else if(!inputModeStored)inputMode=pointConfig?.compatibility_mode_default===true||pointConfig?.default_mode==="compatibility"?"compatibility":"points";
    saveInputMode();paint();return inputMode;
  }
};
})();
