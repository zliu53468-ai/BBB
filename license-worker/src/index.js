// BBB activation gateway — Cloudflare Worker + D1.
// LICENSE_PEPPER and ADMIN_TOKEN are secrets: never put them in Git or frontend.
const alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const encoder=new TextEncoder();
function randomCode(n=24){const b=new Uint8Array(n);crypto.getRandomValues(b);return Array.from(b,x=>alphabet[x%alphabet.length]).join("");}
function b64url(bytes){const u=bytes instanceof ArrayBuffer?new Uint8Array(bytes):bytes;let s="";for(const x of u)s+=String.fromCharCode(x);return btoa(s).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");}
function from64(s){if(!/^[\w-]+$/.test(s)||s.length>2000)throw new Error("bad encoding");let z=atob(s.replace(/-/g,"+").replace(/_/g,"/")+"=".repeat((4-s.length%4)%4));return Uint8Array.from(z,ch=>ch.charCodeAt(0));}
function hex(bytes){return Array.from(new Uint8Array(bytes),x=>x.toString(16).padStart(2,"0")).join("");}
async function digest(s){return hex(await crypto.subtle.digest("SHA-256",encoder.encode(s)));}
async function codeHash(code,env){const key=await crypto.subtle.importKey("raw",encoder.encode(env.LICENSE_PEPPER),{name:"HMAC",hash:"SHA-256"},false,["sign"]);return hex(await crypto.subtle.sign("HMAC",key,encoder.encode(code)));}
function normalizeCode(value){return String(value||"").toUpperCase().replace(/[\s-]/g,"");}
function validCode(c){return c.length===24&&[...c].every(x=>alphabet.includes(x));}
function jwkClean(p){if(!p||p.kty!=="EC"||p.crv!=="P-256"||typeof p.x!=="string"||typeof p.y!=="string"||!p.x.match(/^[\w-]{43}$/)||!p.y.match(/^[\w-]{43}$/))throw new Error("invalid device public key");return JSON.stringify({kty:"EC",crv:"P-256",x:p.x,y:p.y});}
function constantEq(a,b){if(a.length!==b.length)return false;let diff=0;for(let i=0;i<a.length;i++)diff|=a.charCodeAt(i)^b.charCodeAt(i);return diff===0;}
function reply(obj,status=200,headers={}){return new Response(JSON.stringify(obj),{status,headers:{"Content-Type":"application/json;charset=utf-8","Cache-Control":"no-store","X-Content-Type-Options":"nosniff",...headers}});}
async function readBody(req){if(Number(req.headers.get("Content-Length")||0)>8192)throw new Error("payload_too_large");const t=await req.text();if(t.length>8192)throw new Error("payload_too_large");return JSON.parse(t);}
function isLive(row,now){return row&&row.revoked_at===null&&(row.expires_at===null||row.expires_at>now);}
async function authorize(req,env){const token=req.headers.get("Authorization")?.replace(/^Bearer\s+/i,"")||"";if(!env.ADMIN_TOKEN||token.length<24)return false;return constantEq(await digest(token),await digest(env.ADMIN_TOKEN));}
async function rateLimit(req,env,now){const ip=req.headers.get("CF-Connecting-IP")||"unknown";const key=await codeHash("ip:"+ip,env),windowStart=Math.floor(now/60000);
  await env.DB.prepare("INSERT INTO rate_limits (key,window_start,hits) VALUES (?,?,1) ON CONFLICT(key) DO UPDATE SET hits=CASE WHEN window_start=excluded.window_start THEN hits+1 ELSE 1 END,window_start=excluded.window_start").bind(key,windowStart).run();
  const row=await env.DB.prepare("SELECT hits FROM rate_limits WHERE key=?").bind(key).first();
  if(row?.hits>25)throw Object.assign(new Error("too_many_requests"),{status:429});
}
async function handle(req,env,path){
 const now=Date.now();
 if(path==="/health"&&req.method==="GET")return reply({ok:true,service:"bbb-license-api",configured:!!(env.DB&&env.LICENSE_PEPPER&&env.ADMIN_TOKEN)});
 if(!env.DB||!env.LICENSE_PEPPER||!env.ADMIN_TOKEN)return reply({error:"server_not_configured"},503);
 if(path.startsWith("/api/admin/")){
   if(!await authorize(req,env))return reply({error:"unauthorized"},401);
   if(path==="/api/admin/licenses"&&req.method==="GET"){
     const {results}=await env.DB.prepare("SELECT id,code_suffix,label,created_at,expires_at,bound_at,revoked_at,generation FROM licenses ORDER BY created_at DESC LIMIT 250").all();
     return reply({licenses:results});
   }
   if(path==="/api/admin/licenses"&&req.method==="POST"){
     const body=await readBody(req),count=Number(body.count||1),label=String(body.label||"").trim().slice(0,100),expiresAt=body.expires_at==null?null:Number(body.expires_at);
     if(!Number.isInteger(count)||count<1||count>30||expiresAt!==null&&(!Number.isSafeInteger(expiresAt)||expiresAt<=now))return reply({error:"invalid_parameters"},400);
     const created=[];
     for(let i=0;i<count;i++){
       const code=randomCode(),id=crypto.randomUUID(),normalized=normalizeCode(code),hash=await codeHash(normalized,env);
       await env.DB.prepare("INSERT INTO licenses (id,code_hash,code_suffix,label,created_at,expires_at) VALUES (?,?,?,?,?,?)").bind(id,hash,code.slice(-4),label,now,expiresAt).run();
       created.push({id,code:code.match(/.{1,6}/g).join("-")});
     }
     return reply({created,warning:"Activation codes are displayed once; save them securely."},201);
   }
   const match=path.match(/^\/api\/admin\/licenses\/([0-9a-f-]{36})\/(reset|revoke)$/);
   if(match&&req.method==="POST"){
     const [,id,action]=match;
     const result=action==="reset"
       ?await env.DB.prepare("UPDATE licenses SET public_key=NULL,device_hash=NULL,bound_at=NULL,generation=generation+1 WHERE id=? AND revoked_at IS NULL").bind(id).run()
       :await env.DB.prepare("UPDATE licenses SET revoked_at=? WHERE id=? AND revoked_at IS NULL").bind(now,id).run();
     if(!result.meta?.changes)return reply({error:"not_found_or_revoked"},404);
     return reply({ok:true,action,id});
   }
   return reply({error:"not_found"},404);
 }
 if(path==="/api/challenge"&&req.method==="POST"){
   await rateLimit(req,env,now);
   const body=await readBody(req),code=normalizeCode(body.code);
   if(!validCode(code))return reply({error:"invalid_or_inactive_code"},403);
   const hash=await codeHash(code,env),license=await env.DB.prepare("SELECT id,revoked_at,expires_at FROM licenses WHERE code_hash=?").bind(hash).first();
   if(!isLive(license,now))return reply({error:"invalid_or_inactive_code"},403);
   const id=crypto.randomUUID(),nonce=b64url(crypto.getRandomValues(new Uint8Array(24)));
   await env.DB.prepare("INSERT INTO challenges (id,code_hash,nonce,expires_at) VALUES (?,?,?,?)").bind(id,hash,nonce,now+120000).run();
   return reply({challenge_id:id,nonce,expires_in_seconds:120});
 }
 if(path==="/api/verify"&&req.method==="POST"){
   await rateLimit(req,env,now);
   const body=await readBody(req),code=normalizeCode(body.code),challengeId=String(body.challenge_id||"");
   if(!validCode(code)||!(/^[a-f0-9-]{36}$/i).test(challengeId))return reply({error:"invalid_verification"},403);
   const hash=await codeHash(code,env);
   const used=await env.DB.prepare("DELETE FROM challenges WHERE id=? AND code_hash=? AND expires_at>? RETURNING nonce").bind(challengeId,hash,now).first();
   if(!used)return reply({error:"challenge_expired_or_used"},403);
   let publicKey;
   try{publicKey=jwkClean(body.public_key)}catch(_){return reply({error:"invalid_device_key"},400);}
   let ok=false;
   try{
     const key=await crypto.subtle.importKey("jwk",JSON.parse(publicKey),{name:"ECDSA",namedCurve:"P-256"},false,["verify"]);
     ok=await crypto.subtle.verify({name:"ECDSA",hash:"SHA-256"},key,from64(String(body.signature||"")),encoder.encode("bbb-activate-v1:"+challengeId+":"+used.nonce));
   }catch(_){ok=false;}
   if(!ok)return reply({error:"invalid_device_signature"},403);
   const deviceHash=await digest(publicKey);
   // Atomic first-writer binding. Different devices cannot overwrite an existing key.
   const bound=await env.DB.prepare("UPDATE licenses SET public_key=?,device_hash=?,bound_at=? WHERE code_hash=? AND public_key IS NULL AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at>?)").bind(publicKey,deviceHash,now,hash,now).run();
   const row=await env.DB.prepare("SELECT id,public_key,bound_at,expires_at,revoked_at FROM licenses WHERE code_hash=?").bind(hash).first();
   if(!isLive(row,now))return reply({error:"invalid_or_inactive_code"},403);
   if(row.public_key!==publicKey)return reply({error:"code_bound_to_another_device"},409);
   return reply({ok:true,license_id:row.id,bound_at:row.bound_at,expires_at:row.expires_at,first_activation:!!bound.meta?.changes});
 }
 return reply({error:"not_found"},404);
}
export default {
 async fetch(req,env){
   const origin=req.headers.get("Origin");
   const allow=(env.ALLOWED_ORIGINS||"https://bgsv1.xyz").split(",").map(x=>x.trim()).includes(origin);
   const cors=allow?{"Access-Control-Allow-Origin":origin,"Vary":"Origin","Access-Control-Allow-Headers":"Content-Type, Authorization","Access-Control-Allow-Methods":"GET, POST, OPTIONS"}:{};
   if(req.method==="OPTIONS")return new Response(null,{status:allow?204:403,headers:cors});
   // Browser requests are limited to configured origins; non-browser admin CLI
   // remains protected by bearer authentication.
   if(origin&&!allow)return reply({error:"origin_not_allowed"},403);
   try{
     const url=new URL(req.url),response=await handle(req,env,url.pathname);
     for(const [k,v]of Object.entries(cors))response.headers.set(k,v);
     return response;
   }catch(e){
     const status=e.status||(/payload_too_large|SyntaxError|invalid|Unexpected token/i.test(e.message)?400:500);
     const result=reply({error:status===500?"internal_error":e.message},status);
     for(const [k,v]of Object.entries(cors))result.headers.set(k,v);
     return result;
   }
 }
};
