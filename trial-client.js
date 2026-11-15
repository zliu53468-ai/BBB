/* Public 30-minute visitor preview; never changes BBB inference or licensed storage.
 * Cloudflare D1 determines the fixed expiry. The visitor can reload but not renew
 * a trial bound to the existing browser device ID.
 */
(function(){
"use strict";
const api=String(window.BBB_AUTH_API_BASE||"").replace(/\/+$/,"");
const sessionKey="bbb-public-trial-session-v1",deviceKey="bbb-public-trial-device-v1";
const startBtn=document.getElementById("trialStartBtn");
const startMsg=document.getElementById("trialStartMessage");
const gate=document.getElementById("licenseGate");
const panel=document.getElementById("predictionApp");
const banner=document.getElementById("bbbTrialBanner");
const timerEl=document.getElementById("bbbTrialCountdown");
const statusEl=document.getElementById("bbbTrialStatus");
if(!startBtn||!gate||!panel||!banner)return;
let mounted=false,pending=false,deadline=0,ticker=null,checker=null,token="",verifying=false;
// Observe the original paid-license gate instead of changing its protected code.
const ready=new Promise(resolve=>{
  const msg=document.getElementById("licenseMessage");
  const isSettled=()=>gate.hidden||!(/正在載入|授權中|正在核對/.test(msg?.textContent||""));
  if(isSettled()){resolve();return;}
  let finished=false;
  const observer=new MutationObserver(()=>{
    if(isSettled()&&!finished){finished=true;observer.disconnect();clearTimeout(limit);resolve();}
  });
  observer.observe(gate,{attributes:true,childList:true,subtree:true,characterData:true});
  const limit=setTimeout(()=>{if(!finished){finished=true;observer.disconnect();resolve();}},12000);
});
function showError(message){if(startMsg){startMsg.textContent=message;startMsg.setAttribute("role","status");}}
function readSession(){
  try {const data=JSON.parse(localStorage.getItem(sessionKey)||"null");
    return data&&typeof data.token==="string"&&/^[A-Za-z0-9_-]{43}$/.test(data.token)?data:null;
  }catch(_){return null;}
}
function goEnded(reason="expired"){
  // Stop further user interaction with the analysis panel before navigation.
  mounted=false;
  panel.hidden=true;
  if(ticker)clearInterval(ticker);
  if(checker)clearInterval(checker);
  window.location.replace("./trial-ended.html?reason="+encodeURIComponent(reason));
}
async function request(path,data){
  if(!api.startsWith("https://"))throw Error("試用驗證服務尚未設定");
  const r=await fetch(api+path,{method:"POST",mode:"cors",cache:"no-store",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  const j=await r.json().catch(()=>({error:"invalid_response"}));
  if(!r.ok){const error=Error(j.error||"trial_api_error");error.status=r.status;throw error;}
  return j;
}
function updateTime(){
  if(!mounted)return;
  const ms=Math.max(0,deadline-performance.now());
  const sec=Math.ceil(ms/1000);
  if(timerEl)timerEl.textContent=String(Math.floor(sec/60)).padStart(2,"0")+":"+String(sec%60).padStart(2,"0");
  if(ms<=0)goEnded("expired");
}
function updateDeadline(response){
  if(!Number.isFinite(response?.expires_at)||!Number.isFinite(response?.server_now))
    throw Error("invalid_trial_time");
  if(response.expired===true||response.active===false||response.expires_at<=response.server_now){
    goEnded("expired");return false;
  }
  deadline=performance.now()+Math.max(0,response.expires_at-response.server_now);
  return true;
}
async function confirm(){
  if(!mounted||verifying||!token)return;
  verifying=true;
  try {
    const data=await request("/api/trial/status",{token});
    if(!mounted)return;
    if(updateDeadline(data)){updateTime();if(statusEl)statusEl.textContent="試用期間每 30 秒由伺服器確認，重新整理不會重設倒數。";}
  }catch(_){goEnded("verification");}
  finally{verifying=false;}
}
async function script(src){
  await new Promise((resolve,reject)=>{
    const el=document.createElement("script");el.src=src;el.async=false;
    el.onload=resolve;el.onerror=()=>reject(Error("無法載入分析面板"));
    document.body.appendChild(el);
  });
}
async function mount(response){
  if(mounted||gate.hidden||!panel.hidden)return;
  if(!updateDeadline(response))return;
  window.__BBB_AUTH_MODE__="trial";
  token=String(response.token||token);
  const u=new URL(window.location.href);
  u.searchParams.set("trial","1");
  window.history.replaceState(window.history.state,"",u.href);
  // Avoid carrying any legacy unlicensed analysis history into a fresh preview.
  if(response.new_session)try{localStorage.removeItem("bgs_soft_shoe_bpt_v1");}catch(_){}
  gate.hidden=true;
  panel.hidden=false;
  banner.hidden=false;
  try {
    // Mirror the public read-only script URLs from the untouched licensed loader.
    const urls=[
      "derived_road_aux.js?v=20261113-road-aux-adaptive-v1",
      "soft_shoe_decision_runtime.js?v=20261113-road-aux-adaptive-v1",
      "bpt_big_road.js?v=20261106-bigroad-v1",
      "soft_shoe_panel.js?v=20261113-road-aux-adaptive-v1"
    ];
    for(const src of urls)await script(src);
  }catch(error){
    gate.hidden=false;panel.hidden=true;banner.hidden=true;window.__BBB_AUTH_MODE__=null;
    showError("無法啟動試用面板："+String(error.message||error));return;
  }
  mounted=true;
  ticker=setInterval(updateTime,1000);
  checker=setInterval(confirm,30000);
  document.addEventListener("visibilitychange",()=>{if(!document.hidden)confirm();});
  window.addEventListener("focus",confirm);
  // This button normally clears *licensed* history. In trial mode it only resets
  // the anonymous local B/P/T history, keeping the same server-side deadline.
  document.getElementById("btnEnd")?.addEventListener("click",()=>{
    try{localStorage.removeItem("bgs_soft_shoe_bpt_v1");}catch(_){}
    window.location.reload();
  });
  updateTime();
}
async function startOrResume(){
  if(pending||mounted)return;
  pending=true;startBtn.disabled=true;
  showError("正在取得 30 分鐘試用授權…");
  try {
    await ready;
    if(gate.hidden||!panel.hidden)return;
    if(!window.crypto?.randomUUID)throw Error("瀏覽器不支援安全試用識別，請更新瀏覽器");
    const existing=readSession();
    if(existing){
      token=existing.token;
      const status=await request("/api/trial/status",{token});
      if(!updateDeadline(status))return;
      await mount({...status,token,new_session:false});
    }else{
      let device=localStorage.getItem(deviceKey);
      if(!device){device=crypto.randomUUID();localStorage.setItem(deviceKey,device);}
      const result=await request("/api/trial/start",{device_id:device});
      if(!result.ok||!result.token)throw Error("試用授權建立失敗");
      localStorage.setItem(sessionKey,JSON.stringify({token:result.token}));
      await mount({...result,new_session:true});
    }
  }catch(error){
    if(error.status===409){goEnded("used");return;}
    showError("試用啟動失敗："+String(error.message||error)+"。請稍後再試。");
  }finally{pending=false;startBtn.disabled=false;}
}
startBtn.addEventListener("click",startOrResume);
ready.then(()=>{
  if(gate.hidden||!panel.hidden)return;
  if(new URLSearchParams(window.location.search).get("trial")==="1"&&readSession())startOrResume();
});
})();
