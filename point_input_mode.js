/* Optional compact point input. Loaded ONLY after the license gate.
 * Does not replace Frozen Core result handlers or adjust production prediction.
 * Score observations are recorded via the existing MACRO journal for replay.
 */
(() => {
"use strict";
const KEY="bgs_score_input_mode_v1";
const HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const el=id=>document.getElementById(id);
const buttonMode=el("modeResults"),pointMode=el("modePoints");
const buttonPanel=el("resultModePanel"),pointPanel=el("pointModePanel");
const banker=el("macroBankerScore"),player=el("macroPlayerScore");
const submit=el("btnRecordPoints"),preview=el("pointResultPreview");
if(!buttonMode||!pointMode||!buttonPanel||!pointPanel||!banker||!player||!submit||!preview)return;
const parseScore=v=>/^[0-9]$/.test(String(v))?Number(v):null;
const deriveOutcome=(bankerScore,playerScore)=>{
  const b=parseScore(bankerScore),p=parseScore(playerScore);
  if(b===null||p===null)return null;
  return b>p?"B":b<p?"P":"T";
};
const clear=()=>{banker.value="";player.value="";};
const note=(message,isError=false)=>{
  preview.textContent=message;
  preview.classList.toggle("is-error",!!isError);
};
const readHistory=()=>{
  try{
    const saved=JSON.parse(localStorage.getItem(HISTORY_KEY)||"null");
    return Array.isArray(saved?.history)?saved.history.filter(x=>["B","P","T"].includes(x)):[];
  }catch(_){return[];}
};
function showPreview(){
  const outcome=deriveOutcome(banker.value,player.value);
  if(outcome===null){note("選擇莊、閒最終點數，自動判定勝方。");return;}
  note("本局："+(outcome==="B"?"莊勝":outcome==="P"?"閒勝":"和局")+"（"+banker.value+"："+player.value+"）");
}
function selectMode(mode){
  const points=mode==="points";
  buttonPanel.hidden=points;
  pointPanel.hidden=!points;
  buttonMode.setAttribute("aria-pressed",String(!points));
  pointMode.setAttribute("aria-pressed",String(points));
  buttonMode.classList.toggle("is-active",!points);
  pointMode.classList.toggle("is-active",points);
  if(!points)clear(); // Never attach half-entered scores to plain B/P/T clicks.
  showPreview();
  try{localStorage.setItem(KEY,points?"points":"buttons");}catch(_){}
}
function recordPointHand(){
  if(pointPanel.hidden)return false;
  const outcome=deriveOutcome(banker.value,player.value);
  if(outcome===null){note("請先選擇莊、閒的 0～9 點。",true);return false;}
  const button=el("btn"+outcome);
  if(!button||typeof button.click!=="function"){note("原本結果按鈕未就緒，未儲存。",true);return false;}
  const before=readHistory();
  // The authoritative Frozen Core handler, plus pre/post telemetry listeners,
  // must all run through this exact real button click path.
  button.click();
  const after=readHistory();
  if(after.length!==before.length+1||after.at(-1)!==outcome){
    note("此局未確認儲存成功，請查看上方提示，勿重複按鍵。",true);
    return false;
  }
  // The runtime clears the score fields after its post-click journal commit.
  clear();
  note("已記錄："+(outcome==="B"?"莊":outcome==="P"?"閒":"和")+" "+before.length+" → "+after.length+" 局");
  return true;
}
buttonMode.addEventListener("click",()=>selectMode("buttons"));
pointMode.addEventListener("click",()=>selectMode("points"));
banker.addEventListener("change",showPreview);
player.addEventListener("change",showPreview);
submit.addEventListener("click",recordPointHand);
const back=el("btnBack");if(back)back.addEventListener("click",()=>{clear();showPreview();});
const end=el("btnEnd");if(end)end.addEventListener("click",()=>{clear();});
let saved="buttons";try{saved=localStorage.getItem(KEY)||"buttons";}catch(_){}
selectMode(saved==="points"?"points":"buttons");
if(typeof window!=="undefined")window.__BBB_POINT_INPUT__={deriveOutcome,recordPointHand,selectMode,readHistory};
})();
