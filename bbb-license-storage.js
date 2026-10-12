/* BBB license-isolated analysis storage. UI adapter; no prediction-model changes.
 * All bgs* keys are virtualized behind IndexedDB per verified license_id.
 * Legacy unscoped localStorage data is never automatically assigned to a code.
 * End Analysis purges the entire verified license scope and old unscoped keys.
 */
(function(root){
"use strict";
const DB_NAME="bbb-analysis-isolated-v1",DATA="records",META="epochs";
const MANAGED=/^bgs(?:[_a-z0-9])/i;
const RESET_PREFIX="bbb-isolated-reset:";
const native={get:Storage.prototype.getItem,set:Storage.prototype.setItem,
  remove:Storage.prototype.removeItem};
let dbPromise=null,active="",epoch="",remoteGeneration=null,ready=false,clearing=false;
let memory=new Map(),pending=new Set(),channel=null;
let installDone=false,writeError="";
const own=()=>root.localStorage;
const scoped=k=>String(k);
const isKey=k=>MANAGED.test(String(k));
const unique=()=>root.crypto?.randomUUID?.()||String(Date.now())+"-"+Math.random();
function notifyError(reason){
  writeError=String(reason?.message||reason||"storage_error");
  const message=root.document?.getElementById("message");
  if(message){message.textContent="分析紀錄儲存異常（IndexedDB）：" + writeError;message.classList.add("warning");}
}
function database(){
  if(dbPromise)return dbPromise;
  dbPromise=new Promise((resolve,reject)=>{
    const r=root.indexedDB.open(DB_NAME,1);
    r.onupgradeneeded=()=>{
      if(!r.result.objectStoreNames.contains(DATA)){
        const store=r.result.createObjectStore(DATA,{keyPath:["scope","key"]});
        store.createIndex("scope","scope",{unique:false});
      }
      if(!r.result.objectStoreNames.contains(META))r.result.createObjectStore(META);
    };
    r.onsuccess=()=>resolve(r.result);
    r.onerror=()=>reject(r.error||Error("idb_open_failed"));
    r.onblocked=()=>reject(Error("idb_upgrade_blocked"));
  });
  return dbPromise;
}
function transactionDone(t){
  return new Promise((resolve,reject)=>{
    t.oncomplete=()=>resolve();
    t.onerror=()=>reject(t.error||Error("idb_tx_error"));
    t.onabort=()=>reject(t.error||Error("idb_tx_aborted"));
  });
}
function subscribeWrite(promise){
  pending.add(promise);
  promise.catch(notifyError).finally(()=>pending.delete(promise));
}
function save(key,value){
  if(!ready||clearing)throw Error("license_storage_not_ready");
  const k=scoped(key),v=String(value);
  memory.set(k,v);
  const token=epoch,scope=active;
  const promise=database().then(db=>{
    const tx=db.transaction([META,DATA],"readwrite"),wait=transactionDone(tx);
    const check=tx.objectStore(META).get(scope);
    check.onsuccess=()=>{
      if(check.result?.epoch!==token){notifyError("資料已被另一個分頁結束分析，請重新整理");return;}
      tx.objectStore(DATA).put({scope,key:k,value:v});
    };
    return wait;
  });
  subscribeWrite(promise);
}
function drop(key){
  if(!ready||clearing)throw Error("license_storage_not_ready");
  const k=scoped(key);memory.delete(k);
  const token=epoch,scope=active;
  const promise=database().then(db=>{
    const tx=db.transaction([META,DATA],"readwrite"),wait=transactionDone(tx);
    const check=tx.objectStore(META).get(scope);
    check.onsuccess=()=>{
      if(check.result?.epoch!==token){notifyError("資料已被另一個分頁結束分析，請重新整理");return;}
      tx.objectStore(DATA).delete([scope,k]);
    };
    return wait;
  });
  subscribeWrite(promise);
}
function install(){
  if(installDone)return;
  Storage.prototype.getItem=function(key){
    if(this===own()&&isKey(key))
      return ready&&memory.has(scoped(key))?memory.get(scoped(key)):null;
    return native.get.call(this,key);
  };
  Storage.prototype.setItem=function(key,value){
    if(this===own()&&isKey(key)){
      if(!ready)throw Error("license_storage_not_ready");
      return save(key,value);
    }
    return native.set.call(this,key,value);
  };
  Storage.prototype.removeItem=function(key){
    if(this===own()&&isKey(key)){
      if(!ready)return;
      return drop(key);
    }
    return native.remove.call(this,key);
  };
  installDone=true;
}
async function readScope(scope,generation){
  const db=await database();
  let data=[],metaValue=null;
  const tx=db.transaction([META,DATA],"readwrite"),done=transactionDone(tx);
  const ms=tx.objectStore(META),ds=tx.objectStore(DATA),m=ms.get(scope);
  m.onsuccess=()=>{
    let record=m.result;
    if(!record){record={epoch:unique(),remote_generation:generation};ms.put(record,scope);}
    // Signed server verification determines whether this previously used code
    // has advanced to a new analysis epoch on another browser/device.
    if(Number.isInteger(generation)&&Number.isInteger(record.remote_generation)&&generation>record.remote_generation){
      const keys=ds.index("scope").openKeyCursor(IDBKeyRange.only(scope));
      keys.onsuccess=()=>{
        const cursor=keys.result;
        if(cursor){ds.delete(cursor.primaryKey);cursor.continue();}
      };
      record={epoch:unique(),remote_generation:generation};
      ms.put(record,scope);
      metaValue=record;
      data=[];
    }else{
      if(Number.isInteger(generation)&&record.remote_generation!==generation){
        record={...record,remote_generation:Math.max(generation,record.remote_generation??0)};
        ms.put(record,scope);
      }
      metaValue=record;
      const q=ds.index("scope").getAll(scope);
      q.onsuccess=()=>{data=q.result||[];};
    }
  };
  await done;
  return {data,meta:metaValue};
}
async function clearScope(scope,newGeneration=null){
  const db=await database();
  const tx=db.transaction([META,DATA],"readwrite"),done=transactionDone(tx);
  const meta=tx.objectStore(META),data=tx.objectStore(DATA);
  const r=meta.get(scope);
  r.onsuccess=()=>{
    const old=r.result||{},g=Number.isInteger(newGeneration)?Math.max(newGeneration,old.remote_generation||0):old.remote_generation??null;
    meta.put({epoch:unique(),remote_generation:g},scope);
    const cursor=data.index("scope").openKeyCursor(IDBKeyRange.only(scope));
    cursor.onsuccess=()=>{
      const item=cursor.result;
      if(item){data.delete(item.primaryKey);item.continue();}
    };
  };
  await done;
}
function purgeLegacy(){
  // Only older unscoped bgs* analysis data, never license-code/device-key IndexedDB.
  // They were created before any license namespace and have ambiguous ownership.
  for(let i=own().length-1;i>=0;i--){
    const key=own().key(i);
    if(isKey(key)){try{native.remove.call(own(),key);}catch(error){notifyError(error);}}
  }
}
function reloadOther(){
  ready=false;memory.clear();clearing=true;
  try{root.location.reload();}catch(_){}
}
function setupListeners(scope){
  if(channel){channel.close();channel=null;}
  if(typeof root.BroadcastChannel==="function"){
    channel=new root.BroadcastChannel("bbb-isolated-session:"+scope);
    channel.onmessage=e=>{if(e.data?.type==="reset"&&e.data?.scope===scope)reloadOther();};
  }
  if(typeof root.addEventListener==="function"){
    root.addEventListener("storage",e=>{
      if(ready&&active===scope&&e.key===RESET_PREFIX+scope)reloadOther();
    });
  }
}
async function start(licenseId,generation=null){
  if(!/^[a-f0-9-]{36}$/i.test(String(licenseId||"")))throw Error("verified_license_id_required");
  if(ready&&active===String(licenseId)){
    if(Number.isInteger(generation))await syncGeneration(generation);
    return;
  }
  ready=false;clearing=false;active=String(licenseId);writeError="";
  const snapshot=await readScope(active,Number.isInteger(generation)?generation:null);
  epoch=snapshot.meta.epoch;remoteGeneration=snapshot.meta.remote_generation??null;
  memory=new Map(snapshot.data.map(row=>[row.key,row.value]));
  install();ready=true;
  setupListeners(active);
  // Legacy raw records are intentionally not imported. Importing could mix
  // different licenses used on this browser before this privacy upgrade.
}
async function end(newGeneration=null){
  if(!ready||clearing)return false;
  clearing=true;
  const scope=active;
  // The Frozen Core end-click handler has already reset in-memory history.
  await Promise.allSettled([...pending]);
  try{
    await clearScope(scope,newGeneration);
    memory.clear();writeError="";
    purgeLegacy();
    if(channel)channel.postMessage({type:"reset",scope});
    try{native.set.call(own(),RESET_PREFIX+scope,String(Date.now()));}catch(_){}
    // Nothing can access this license again before the next verified start().
    ready=false;active="";epoch="";
    if(channel){channel.close();channel=null;}
    return true;
  }catch(error){notifyError(error);return false;}
  finally{clearing=false;}
}
async function syncGeneration(generation){
  if(!ready||!Number.isInteger(generation)||generation<0)return false;
  if(remoteGeneration===null){
    remoteGeneration=generation;
    // Initial older deployments can report no generation. Preserve until a
    // later *strictly greater* generation is verified.
    const db=await database(),tx=db.transaction(META,"readwrite"),done=transactionDone(tx);
    tx.objectStore(META).put({epoch,remote_generation:generation},active);
    await done;return false;
  }
  if(generation<=remoteGeneration)return false;
  const ok=await end(generation);
  if(ok){reloadOther();return true;}
  throw Error("remote_analysis_reset_failed");
}
root.__BBB_LICENSE_STORAGE__={start,end,syncGeneration,managedKey:isKey,
  status:()=>({active:!!ready,scope:active,pending:pending.size,size:memory.size,error:writeError})};
})(window);
