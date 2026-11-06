"use strict";
self.onmessage=event=>{
 const {id,history}=event.data||{};
 try{
  if(!self.BBBSoftShoe)importScripts("soft_shoe_decision_runtime.js?v=20261106-bpt-v1");
  const report=self.BBBSoftShoe.run(history);
  self.postMessage({id,report});
 }catch(e){self.postMessage({id,error:String(e?.message||e)});}
};
