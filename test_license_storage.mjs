#!/usr/bin/env node
"use strict";
// No external packages. Mock only the IndexedDB primitives used by the
// per-license adapter, then exercise actual module across fresh browser tabs.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {webcrypto} from "node:crypto";
const source=fs.readFileSync("bbb-license-storage.js","utf8");
const uuidA="11111111-1111-4111-8111-111111111111";
const uuidB="22222222-2222-4222-8222-222222222222";
const raw=new Map([["bgs256d_frozen_6x15_forward_v18",JSON.stringify({history:["B","B"],legacy:true})]]);
const stores=new Map();
class MemoryTx{
  constructor(names){this.names=names;this.pending=0;this.finished=false;
    queueMicrotask(()=>this.check());}
  check(){if(this.pending===0&&!this.finished){this.finished=true;queueMicrotask(()=>this.oncomplete?.());}}
  task(fn){
    this.pending++;const r={result:undefined,error:null,onsuccess:null,onerror:null};
    queueMicrotask(()=>{
      try{r.result=fn();r.onsuccess?.();}catch(e){r.error=e;r.onerror?.();}
      this.pending--;this.check();
    });return r;
  }
  objectStore(name){
    const store=stores.get(name);
    if(!store)throw Error("missing store "+name);
    const db=this;
    return {
      get(key){return db.task(()=>store.get(JSON.stringify(key)));},
      put(value,key){return db.task(()=>{store.set(JSON.stringify(key??[value.scope,value.key]),value);return key;});},
      delete(key){return db.task(()=>store.delete(JSON.stringify(key)));},
      index(indexName){
        assert.equal(indexName,"scope");
        return {
          getAll(scope){return db.task(()=>[...store.values()].filter(v=>v.scope===scope));},
          openKeyCursor(condition){
            const keys=[...store.entries()].filter(([,v])=>v.scope===condition);
            const r={onsuccess:null,result:null,error:null};
            let i=0;
            db.pending++;
            const next=()=>queueMicrotask(()=>{
              if(i<keys.length){
                const [rawKey]=keys[i++];
                r.result={primaryKey:JSON.parse(rawKey),continue:next};
                r.onsuccess?.();
              }else{
                r.result=null;r.onsuccess?.();db.pending--;db.check();
              }
            });
            next();return r;
          }
        };
      }
    };
  }
}
const idb={
 open(){
  const r={onsuccess:null,onerror:null,onupgradeneeded:null,result:null,error:null};
  queueMicrotask(()=>{
    r.result={
      objectStoreNames:{contains:name=>stores.has(name)},
      createObjectStore(name){stores.set(name,new Map());return {createIndex(){}};},
      transaction:names=>new MemoryTx(names)
    };
    if(!stores.has("records"))r.onupgradeneeded?.();
    r.onsuccess?.();
  });return r;
 }
};
function tab(){
  class Storage{
    getItem(key){return raw.get(String(key))??null;}
    setItem(key,value){raw.set(String(key),String(value));}
    removeItem(key){raw.delete(String(key));}
    key(index){return [...raw.keys()][index]??null;}
    get length(){return raw.size;}
  }
  const localStorage=new Storage();
  const messages=[];
  const window={localStorage,indexedDB:idb,crypto:webcrypto,document:{getElementById:()=>null},
    BroadcastChannel:class{constructor(){this.onmessage=null;}postMessage(x){messages.push(x);}close(){}},
    addEventListener(){},location:{reload(){messages.push("reloaded");}}};
  const context={window,Storage,IDBKeyRange:{only:x=>x},Promise,Map,Set,Date,Math,Error,
    Number,String,RegExp,JSON,Object,Array,console,queueMicrotask};
  vm.runInNewContext(source,context,{filename:"bbb-license-storage.js"});
  return {...window,storage:window.__BBB_LICENSE_STORAGE__,messages};
}
async function idle(){await new Promise(r=>setTimeout(r,30));}
(async()=>{
  const a=tab();await a.storage.start(uuidA,0);
  assert.equal(a.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),null,"legacy data must never bleed into a newly verified license");
  const aHand=JSON.stringify({history:["P","T","B"]});
  a.localStorage.setItem("bgs256d_frozen_6x15_forward_v18",aHand);
  a.localStorage.setItem("bgs_xgb_final_training_v5","X".repeat(2_000_000));
  await idle();
  assert(!raw.has("bgs_xgb_final_training_v5"),"large model snapshots must be stored in IDB, not quota-limited localStorage");
  assert.equal(a.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),aHand);
  const b=tab();await b.storage.start(uuidB,0);
  assert.equal(b.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),null,"B may not read A");
  const bHand=JSON.stringify({history:["B","B","P"]});
  b.localStorage.setItem("bgs256d_frozen_6x15_forward_v18",bHand);
  await idle();
  assert.equal(b.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),bHand);
  assert.equal(a.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),aHand,"B did not alter A");
  assert.equal(await a.storage.end(),true,"end analysis completes a full per-license purge");
  assert(!raw.has("bgs256d_frozen_6x15_forward_v18"),"legacy unscoped data removed by end analysis");
  const aNew=tab();await aNew.storage.start(uuidA,0);
  assert.equal(aNew.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),null,"end left no A history");
  assert.equal(aNew.localStorage.getItem("bgs_xgb_final_training_v5"),null,"end left no A snapshots");
  const bReload=tab();await bReload.storage.start(uuidB,0);
  assert.equal(bReload.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),bHand,"purging A leaves B scoped data untouched");
  await bReload.storage.syncGeneration(1);
  assert.equal(bReload.messages.includes("reloaded"),true,"new server epoch forces remote browser reload");
  const bAgain=tab();await bAgain.storage.start(uuidB,1);
  assert.equal(bAgain.localStorage.getItem("bgs256d_frozen_6x15_forward_v18"),null,"server epoch purged B history");
  console.log(JSON.stringify({ok:true,checks:["license_A_not_B","large_payload_indexeddb","old_unscoped_quarantined","end_purges_all_A","B_untouched","remote_generation_purge"]}));
})().catch(e=>{console.error(e);process.exitCode=1;});
