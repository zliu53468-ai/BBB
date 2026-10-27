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
  "particle_filter_runtime.js?v=20261027-point-score-pf-v1",
  "macro_ema.js?v=20261023-middle-late-v2",
  "browser_calibration.js?v=20261027-prequential-ev-v1",
  "final_probability_runtime.js?v=20261027-point-score-pf-v1",
  "point_input_mode.js?v=20261027-point-input-v1"
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
    i