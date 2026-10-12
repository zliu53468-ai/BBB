// In-memory D1 contract test. No Cloudflare credentials or network required.
import fs from "node:fs";
import assert from "node:assert/strict";
import {webcrypto} from "node:crypto";
if(!globalThis.crypto)globalThis.crypto=webcrypto;
const src=fs.readFileSync(new URL("./src/index.js",import.meta.url),"utf8");
const worker=(await import("data:text/javascript;base64,"+Buffer.from(src).toString("base64"))).default;
const rows=new Map(),challenges=new Map(),rates=new Map();
function sqlMatch(sql){return sql.replace(/\s+/g," ").trim();}
const DB={prepare(raw){
  const sql=sqlMatch(raw);
  let p=[];
  const statement={bind(...args){p=args;return statement;},
  async run(){
    if(sql.startsWith("INSERT INTO rate_limits")){const [key,minute]=p,r=rates.get(key);rates.set(key,{window_start:minute,hits:r&&r.window_start===minute?r.hits+1:1});return {meta:{changes:1}};}
    if(sql.startsWith("INSERT INTO licenses")){const [id,code_hash,code_suffix,label,created_at,expires_at]=p;rows.set(code_hash,{id,code_hash,code_suffix,label,created_at,expires_at,revoked_at:null,public_key:null,device_hash:null,bound_at:null,generation:0});return {meta:{changes:1}};}
    if(sql.startsWith("INSERT INTO challenges")){const [id,code_hash,nonce,expires_at]=p;challenges.set(id,{id,code_hash,nonce,expires_at});return {meta:{changes:1}};}
    if(sql.startsWith("UPDATE licenses SET public_key=?")){const [public_key,device_hash,bound_at,hash,now]=p,r=rows.get(hash);if(r&&!r.public_key&&!r.revoked_at&&(!r.expires_at||r.expires_at>now)){Object.assign(r,{public_key,device_hash,bound_at});return {meta:{changes:1}};}return {meta:{changes:0}};}
    if(sql.startsWith("UPDATE licenses SET generation=generation+1")){
      const [hash,key,now]=p,r=rows.get(hash);
      if(!r||r.public_key!==key||r.revoked_at!==null||r.expires_at!==null&&r.expires_at<=now)return {meta:{changes:0}};
      r.generation++;return {meta:{changes:1}};
    }
    if(sql.startsWith("UPDATE licenses SET public_key=NULL")){const r=[...rows.values()].find(x=>x.id===p[0]&&!x.revoked_at);if(!r)return {meta:{changes:0}};r.public_key=null;r.device_hash=null;r.bound_at=null;r.generation++;return {meta:{changes:1}};}
    if(sql.startsWith("UPDATE licenses SET revoked_at")){const r=[...rows.values()].find(x=>x.id===p[1]&&!x.revoked_at);if(!r)return {meta:{changes:0}};r.revoked_at=p[0];return {meta:{changes:1}};}
    throw new Error("unmocked run "+sql);
  },
  async first(){
    if(sql.startsWith("SELECT hits"))return rates.get(p[0])||null;
    if(sql.startsWith("SELECT id,revoked_at,expires_at"))return rows.get(p[0])||null;
    if(sql.startsWith("DELETE FROM challenges")){const [id,hash,now]=p,v=challenges.get(id);if(v&&v.code_hash===hash&&v.expires_at>now){challenges.delete(id);return {nonce:v.nonce};}return null;}
    if(sql.startsWith("SELECT id,public_key"))return rows.get(p[0])||null;
    if(sql.startsWith("SELECT generation"))return rows.get(p[0])||null;
    throw new Error("unmocked first "+sql);
  },
  async all(){
    if(sql.startsWith("SELECT id,code_suffix"))return {results:[...rows.values()].map(({id,code_suffix,label,created_at,expires_at,bound_at,revoked_at,generation})=>({id,code_suffix,label,created_at,expires_at,bound_at,revoked_at,generation}))};
    throw new Error("unmocked all "+sql);
  }};
  return statement;
}};
const env={DB,LICENSE_PEPPER:"test_only_random_secret_123456789",ADMIN_TOKEN:"test_only_admin_token_more_than_24",ALLOWED_ORIGINS:"https://bgsv1.xyz"};
function req(path,data,token){return new Request("https://localhost"+path,{method:data===undefined?"GET":"POST",headers:{Origin:"https://bgsv1.xyz",...(data!==undefined?{"Content-Type":"application/json"}:{}),...(token?{"Authorization":"Bearer "+token}:{})},body:data===undefined?undefined:JSON.stringify(data)});}
async function call(path,data,token){const result=await worker.fetch(req(path,data,token),env);return {status:result.status,data:await result.json()};}
const admin=env.ADMIN_TOKEN;
assert.equal((await call("/api/admin/licenses",{count:1})).status,401);
const created=await call("/api/admin/licenses",{count:1,label:"QA"},admin);
assert.equal(created.status,201);
const {id,code}=created.data.created[0];
assert.equal(code.replace(/-/g,"").length,24);
const issue=async()=>await crypto.subtle.generateKey({name:"ECDSA",namedCurve:"P-256"},false,["sign","verify"]);
async function activate(pair){
  const challenge=await call("/api/challenge",{code});
  assert.equal(challenge.status,200);
  const c=challenge.data,message=new TextEncoder().encode("bbb-activate-v1:"+c.challenge_id+":"+c.nonce);
  const sign=await crypto.subtle.sign({name:"ECDSA",hash:"SHA-256"},pair.privateKey,message);
  const jwk=await crypto.subtle.exportKey("jwk",pair.publicKey);
  const signature=Buffer.from(sign).toString("base64url");
  const payload={code,challenge_id:c.challenge_id,signature,public_key:{kty:"EC",crv:"P-256",x:jwk.x,y:jwk.y}};
  const result=await call("/api/verify",payload);
  const replay=await call("/api/verify",payload);
  assert.equal(replay.status,403,"challenge must be one-use");
  return result;
}
async function endAnalysis(pair){
  const challenge=await call("/api/challenge",{code});
  assert.equal(challenge.status,200);
  const c=challenge.data,message=new TextEncoder().encode("bbb-activate-v1:"+c.challenge_id+":"+c.nonce);
  const signature=Buffer.from(await crypto.subtle.sign({name:"ECDSA",hash:"SHA-256"},pair.privateKey,message)).toString("base64url");
  const jwk=await crypto.subtle.exportKey("jwk",pair.publicKey);
  return call("/api/session/end",{code,challenge_id:c.challenge_id,signature,
    public_key:{kty:"EC",crv:"P-256",x:jwk.x,y:jwk.y}});
}
const a=await issue(),b=await issue();
assert.equal((await activate(a)).status,200,"first device activation");
const previous=await activate(a);
assert.equal(previous.status,200,"same device is remembered");
assert.equal(previous.data.generation,0,"new code starts at generation zero");
const cleared=await endAnalysis(a);
assert.equal(cleared.status,200,"signed end-analysis request accepted");
assert.equal(cleared.data.generation,1,"end-analysis bumps the remote analysis epoch");
assert.equal((await endAnalysis(b)).status,403,"unbound device cannot reset stored analysis");
const restored=await activate(a);
assert.equal(restored.status,200,"original device remains licensed after analysis reset");
assert.equal(restored.data.generation,1,"subsequent verify returns latest analysis epoch");
const denied=await activate(b);assert.equal(denied.status,409,"different device blocked");
const listed=await call("/api/admin/licenses",undefined,admin);assert.equal(listed.status,200);
assert(!JSON.stringify(listed.data).includes(code),"admin list should not leak plaintext codes");
assert.equal((await call("/api/admin/licenses/"+id+"/reset",{},admin)).status,200);
assert.equal((await activate(b)).status,200,"admin reset transfers one-slot binding");
assert.equal((await activate(a)).status,409,"old device blocked after transfer");
assert.equal((await call("/api/admin/licenses/"+id+"/revoke",{},admin)).status,200);
assert.equal((await call("/api/challenge",{code})).status,403,"revoked code unusable");
const options=await worker.fetch(new Request("https://localhost/api/verify",{method:"OPTIONS",headers:{Origin:"https://bgsv1.xyz"}}),env);
assert.equal(options.status,204);
console.log(JSON.stringify({ok:true,tests:["admin_secret","secure_code_generation","first_bind","same_device","second_device_denied","single_use_challenge","signed_end_analysis","remote_generation_increment","foreign_device_cannot_wipe","reset_binding","revoke","cors","no_plaintext_codes_in_listing"]}));
