// D1 contract test for the new visitor trial, independent of paid license tables.
import fs from "node:fs";
import assert from "node:assert/strict";
import {webcrypto} from "node:crypto";
if(!globalThis.crypto)globalThis.crypto=webcrypto;
const src=fs.readFileSync(new URL("./src/index.js",import.meta.url),"utf8");
const worker=(await import("data:text/javascript;base64,"+Buffer.from(src).toString("base64"))).default;
const rates=new Map(),trialsByDevice=new Map(),trialsByToken=new Map();
const norm=x=>x.replace(/\s+/g," ").trim();
const DB={prepare(raw){
  const sql=norm(raw);let args=[];
  const statement={
    bind(...v){args=v;return statement;},
    async run(){
      if(sql.startsWith("INSERT INTO rate_limits")){
        const [key,minute]=args,r=rates.get(key);
        rates.set(key,{window_start:minute,hits:r?.window_start===minute?r.hits+1:1});
        return {meta:{changes:1}};
      }
      if(sql.startsWith("INSERT OR IGNORE INTO trial_sessions")){
        const [id,device_hash,token_hash,started_at,expires_at]=args;
        if(trialsByDevice.has(device_hash))return {meta:{changes:0}};
        const row={id,device_hash,token_hash,started_at,expires_at};
        trialsByDevice.set(device_hash,row);trialsByToken.set(token_hash,row);
        return {meta:{changes:1}};
      }
      throw Error("unmocked run: "+sql);
    },
    async first(){
      if(sql.startsWith("SELECT hits FROM rate_limits"))return rates.get(args[0])||null;
      if(sql.startsWith("SELECT expires_at FROM trial_sessions WHERE device_hash"))return trialsByDevice.get(args[0])||null;
      if(sql.startsWith("SELECT started_at,expires_at FROM trial_sessions WHERE token_hash"))return trialsByToken.get(args[0])||null;
      throw Error("unmocked first: "+sql);
    }
  };
  return statement;
}};
const env={DB,LICENSE_PEPPER:"long-test-secret-please-dont-use-in-production",
  ADMIN_TOKEN:"long-test-admin-token-not-in-production",
  ALLOWED_ORIGINS:"https://bgsv1.xyz"};
async function send(path,payload,origin="https://bgsv1.xyz"){
  const req=new Request("https://localhost"+path,{method:"POST",
    headers:{"Content-Type":"application/json","Origin":origin},body:JSON.stringify(payload)});
  const response=await worker.fetch(req,env);
  return {status:response.status,body:await response.json()};
}
const device=crypto.randomUUID();
const invalid=await send("/api/trial/start",{device_id:"bad"});
assert.equal(invalid.status,400,"invalid device IDs rejected");
const started=await send("/api/trial/start",{device_id:device});
assert.equal(started.status,201,"new visitor gets a session");
assert.equal(started.body.expires_at-started.body.server_now,30*60*1000,"fixed 30-minute server expiry");
assert.match(started.body.token,/^[A-Za-z0-9_-]{43}$/,"32-byte random token");
const checked=await send("/api/trial/status",{token:started.body.token});
assert.equal(checked.status,200);
assert.equal(checked.body.active,true);
assert.equal(checked.body.expires_at,started.body.expires_at);
const reissued=await send("/api/trial/start",{device_id:device});
assert.equal(reissued.status,409,"same browser device cannot renew 30-minute trial");
const badToken=await send("/api/trial/status",{token:"bad"});
assert.equal(badToken.status,400);
const another=await send("/api/trial/start",{device_id:crypto.randomUUID()});
assert.equal(another.status,201,"other browser devices are independent");
assert.notEqual(another.body.token,started.body.token);
const deniedOrigin=await send("/api/trial/start",{device_id:crypto.randomUUID()},"https://untrusted.example");
assert.equal(deniedOrigin.status,403,"CORS origin must be allowed");
const tokenHash=await (async()=>{
  const k=await crypto.subtle.importKey("raw",new TextEncoder().encode(env.LICENSE_PEPPER),{name:"HMAC",hash:"SHA-256"},false,["sign"]);
  const hash=await crypto.subtle.sign("HMAC",k,new TextEncoder().encode("trial-token:"+started.body.token));
  return Buffer.from(hash).toString("hex");
})();
const row=trialsByToken.get(tokenHash);
assert(row,"only a hashed token is stored in D1");
row.expires_at=Date.now()-1;
const expired=await send("/api/trial/status",{token:started.body.token});
assert.equal(expired.status,200);
assert.equal(expired.body.active,false);
assert.equal(expired.body.expired,true);
assert.equal((await send("/api/trial/start",{device_id:device})).status,409,"expired device cannot restart");
console.log(JSON.stringify({ok:true,checks:["30_minutes","fixed_server_expiration","device_reuse_denied","expired_session_denied","distinct_device","origin_block","hashed_token"]}));
