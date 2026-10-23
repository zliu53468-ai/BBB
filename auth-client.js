// BBB login gate. Loads the original prediction scripts only after server verification.
// This is a UI access gate; GitHub Pages cannot protect publicly hosted model sources.
(() => {
"use strict";
const DB_NAME="bbb-license-device-v1",KEY="device-key",LICENSE="license-code";
const gate=document.getElementById("licenseGate"),panel=document.getElementById("predictionApp");
const form=document.getElementById("licenseForm"),input=document.getElementById("licenseCode");
const message=document.getElementById("licenseMessage"),submit=document.getElementById("licenseSubmit");
// Public server URL is not a secret; only Cloudflare holds ADMIN_TOKEN and LICENSE_PEPPER.
const api=String(window.BBB_AUTH_API_BASE||"https://bbb-license-gateway.zliu53468.workers.dev").replace(/\/+$/,"");
const scripts=[
  "app256forward.js",
  "app256continuation.js",
  "particle_filter_runtime.js?v=20261023-middle-late-v2",
  "macro_ema.js?v=20261023-middle-late-v2",
  "browser_calibration.js?v=20261023-middle-late-v2",
  "final_probability_runtime.js?v=20261023-middle-late-v2",
  "bbb-screenshot-road.js?v=20261018-button-only",
  "bbb-desktop-capture.js?v=20261013-desktop-snip-v1"
];
let dbPromise,activeCode="",activeLicenseId="",authorized=false,busy=false,runtimeLoaded=false,endBusy=false;
function show(text,type="info"){message.textContent=text;message.dataset.status=type;}
function openDb(){
  if(!dbPromise)dbPromise=new Promise((resolve,reject)=>{
    const r=indexedDB.open(DB_NAME,1);
    r.onupgradeneeded=()=>{if(!r.result.objectStoreNames.contains("data"))r.result.createObjectStore("data");};
    r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  });
  return dbPromise;
}
async function getValue(key){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const t=db.transaction("data","readonly"),q=t.objectStore("data").get(key);
    q.onsuccess=()=>resolve(q.result);q.onerror=()=>reject(q.error);
  });
}
async function putValue(key,value){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const t=db.transaction("data","readwrite");
    t.objectStore("data").put(value,key);
    t.oncomplete=resolve;t.onerror=()=>reject(t.error);
  });
}
async function eraseValue(key){
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const t=db.transaction("data","readwrite");
    t.objectStore("data").delete(key);
    t.oncomplete=resolve;t.onerror=()=>reject(t.error);
  });
}
function b64url(ab){
  let s="";for(const x of new Uint8Array(ab))s+=String.fromCharCode(x);
  return btoa(s).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
}
function cleanCode(v){return String(v||"").toUpperCase().replace(/[\s-]/g,"");}
async function request(route,body){
  const response=await fetch(api+route,{
    method:"POST",mode:"cors",cache:"no-store",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)
  });
  const data=await response.json().catch(()=>({error:"invalid_server_response"}));
  if(!response.ok){
    const error=new Error(data.error||"server_rejected");
    error.status=response.status;throw error;
  }
  return data;
}
async function deviceKey(){
  let pair=await getValue(KEY);
  if(!pair){
    pair=await crypto.subtle.generateKey({name:"ECDSA",namedCurve:"P-256"},false,["sign","verify"]);
    // Non-extractable private key is retained by the browser's IndexedDB.
    await putValue(KEY,pair);
  }
  return pair;
}
async function signedChallenge(code,route="/api/verify"){
  const pair=await deviceKey();
  const challenge=await request("/api/challenge",{code});
  const bytes=new TextEncoder().encode("bbb-activate-v1:"+challenge.challenge_id+":"+challenge.nonce);
  const signature=b64url(await crypto.subtle.sign({name:"ECDSA",hash:"SHA-256"},pair.privateKey,bytes));
  const jwk=await crypto.subtle.exportKey("jwk",pair.publicKey);
  return request(route,{
    code,challenge_id:challenge.challenge_id,signature,
    public_key:{kty:"EC",crv:"P-256",x:jwk.x,y:jwk.y}
  });
}
async function verify(code){
  if(!api||!api.startsWith("https://"))throw new Error("server_not_configured");
  if(!/^[A-HJ-NP-Z2-9]{24}$/.test(code))throw new Error("invalid_code_format");
  const result=await signedChallenge(code);
  if(!result.ok)throw new Error("verification_failed");
  await putValue(LICENSE,code);
  return result;
}
function errorText(e){
  const names={
    server_not_configured:"管理者尚未完成開通伺服器設定，請聯絡客服。",
    invalid_code_format:"開通碼格式不正確，請確認 24 碼。",
    invalid_or_inactive_code:"開通碼不存在、已到期或已停用。",
    code_bound_to_another_device:"這組開通碼已綁定其他裝置，請聯絡管理者重置。",
    invalid_device_signature:"裝置身分驗證失敗。",
    challenge_expired_or_used:"驗證已過期，請再試一次。",
    too_many_requests:"嘗試過於頻繁，請稍後再試。",
    internal_error:"驗證服務暫時無法使用。"
  };
  return names[e.message]||"驗證未成功，請確認網路或聯絡管理者。";
}
async function injectScript(src){
  return new Promise((resolve,reject)=>{
    const el=document.createElement("script");el.src=src;el.async=false;
    el.onload=resolve;el.onerror=()=>reject(new Error("script_load_failed:"+src));
    document.body.appendChild(el);
  });
}
async function onEndAnalysis(){
  if(!authorized||endBusy)return;
  endBusy=true;busy=true;
  const storage=window.__BBB_LICENSE_STORAGE__;
  const msg=document.getElementById("message");
  if(msg)msg.textContent="正在完整清除本次牌靴的分析資料…";
  try{
    if(!await storage.end())throw Error("local_analysis_delete_failed");
    // Physical deletion is local. Server epoch invalidates stale copies when
    // another authorized page verifies again (requires updated Worker).
    let remoteError="";
    try{
      const result=await signedChallenge(activeCode,"/api/session/end");
      if(!result.ok||!Number.isInteger(result.generation))throw Error("invalid_end_response");
      await storage.acknowledgeRemoteEnd(activeLicenseId,result.generation);
    }catch(error){remoteError=error?.message||"remote_sync_unavailable";}
    if(remoteError){
      try{sessionStorage.setItem("bbb-analysis-sync-warning","本機紀錄已清空；跨裝置同步尚未完成："+remoteError+"。請更新 Cloudflare Worker 並確認連線。");}catch(_){}
    }
    window.location.reload();
  }catch(error){
    if(msg){msg.textContent="無法完整刪除本次分析紀錄，請先保留目前頁面並檢查儲存空間："+String(error?.message||error);msg.classList.add("warning");}
  }finally{endBusy=false;busy=false;}
}
async function unlock(verified){
  if(authorized)return;
  if(runtimeLoaded){window.location.reload();return;}
  if(!window.__BBB_LICENSE_STORAGE__)await injectScript("bbb-license-storage.js?v=20261012-durable-reset-v2");
  if(!verified?.license_id)throw Error("verified_license_id_missing");
  const storage=window.__BBB_LICENSE_STORAGE__;
  await storage.start(verified.license_id,
    Number.isInteger(verified.generation)?verified.generation:null);
  // Retry end/reset on the next authenticated visit if an earlier local purge
  // happened while Cloudflare was offline or still serving an older version.
  if(storage.status().pending_remote_end){
    try{
      const result=await signedChallenge(activeCode,"/api/session/end");
      if(!result.ok||!Number.isInteger(result.generation))throw Error("invalid_end_response");
      await storage.acknowledgeRemoteEnd(verified.license_id,result.generation);
    }catch(error){
      try{sessionStorage.setItem("bbb-analysis-sync-warning",
        "本機紀錄已清空；跨裝置同步仍待完成："+String(error?.message||error)+
        "。系統會在下次登入時重試。");}catch(_){}
    }
  }
  for(const src of scripts)await injectScript(src);
  runtimeLoaded=true;activeLicenseId=verified.license_id;
  const end=document.getElementById("btnEnd");
  if(end)end.addEventListener("click",onEndAnalysis);
  panel.hidden=false;gate.hidden=true;authorized=true;
  try{
    const warning=sessionStorage.getItem("bbb-analysis-sync-warning");
    if(warning){
      const msg=document.getElementById("message");
      if(msg){msg.textContent=warning;msg.classList.add("warning");}
      sessionStorage.removeItem("bbb-analysis-sync-warning");
    }
  }catch(_){}
}
async function activate(code,silent=false){
  if(busy)return;
  busy=true;submit.disabled=true;
  show(silent?"驗證這台裝置的授權中…":"正在核對開通碼與裝置…");
  try{
    const verified=await verify(code);
    activeCode=code;
    await unlock(verified);
  }catch(e){
    // Never display cached private content on verification failure.
    authorized=false;panel.hidden=true;gate.hidden=false;
    if(e.status===403||e.status===409)await eraseValue(LICENSE);
    const label=e.message==="verified_license_id_missing"?"授權服務尚未更新，無法安全隔離會員資料，請更新 Worker。":
      e.message==="license_storage_not_ready"?"會員獨立儲存空間尚未準備完成，請重新整理。":errorText(e);
    show(label,"error");
  }finally{busy=false;submit.disabled=false;}
}
async function confirmSession(){
  if(!authorized||busy||!activeCode)return;
  const storage=window.__BBB_LICENSE_STORAGE__;
  try{
    const verified=await verify(activeCode);
    if(verified.license_id!==activeLicenseId){window.location.reload();return;}
    if(Number.isInteger(verified.generation))
      await storage?.syncGeneration(verified.generation);
  }catch(_){
    // License may have been revoked/reset. Lock the UI on any failed recheck.
    authorized=false;activeCode="";panel.hidden=true;gate.hidden=false;
    show("裝置授權需要重新驗證，請輸入開通碼。","error");
    return;
  }
  if(storage?.status().pending_remote_end){
    try{
      const result=await signedChallenge(activeCode,"/api/session/end");
      if(!result.ok||!Number.isInteger(result.generation))throw Error("invalid_end_response");
      await storage.acknowledgeRemoteEnd(activeLicenseId,result.generation);
    }catch(_){
      // A network failure during optional remote cleanup is NOT license revocation.
      const msg=document.getElementById("message");
      if(msg){msg.textContent="本機分析紀錄已清除，跨裝置同步尚未成功；稍後會自動重試。";msg.classList.add("warning");}
    }
  }
}
form.addEventListener("submit",event=>{
  event.preventDefault();
  activate(cleanCode(input.value));
});
(async()=>{
  panel.hidden=true;
  if(!window.isSecureContext||!window.crypto?.subtle||!window.indexedDB){
    show("此瀏覽器不支援安全金鑰保存，請使用 HTTPS 與新版瀏覽器。","error");submit.disabled=true;return;
  }
  if(!api||!api.startsWith("https://")){show("開通伺服器尚未設定，請聯絡管理者。","error");submit.disabled=true;return;}
  try{
    const code=await getValue(LICENSE);
    if(code){input.value=code.match(/.{1,6}/g).join("-");await activate(code,true);}
    else show("首次使用請輸入開通碼；完成綁定後，此瀏覽器會自動登入。");
  }catch(_){show("無法讀取裝置認證儲存空間。","error");}
})();
window.setInterval(()=>{if(authorized)confirmSession();},5*60*1000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden&&authorized)confirmSession();});
})();
