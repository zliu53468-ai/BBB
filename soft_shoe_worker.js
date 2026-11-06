"use strict";
self.onmessage=event=>{
 const {id,history}=event.data||{};
 try{
  if(!self.BBBSoftShoe)importScripts("soft_shoe_decision_runtime.js?v=20261106-moderate-v3");
  if(!self.BBB_SOFT_SHOE_XGB_MODEL)importScripts("soft_shoe_model_bundle.js?v=20261106-bpt-v1");
  const report=self.BBBSoftShoe.run(history,{},self.BBB_SOFT_SHOE_XGB_MODEL);
  self.postMessage({id,report});
 }catch(e){self.postMessage({id,error:String(e?.message||e)});}
};
