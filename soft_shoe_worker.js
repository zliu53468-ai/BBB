"use strict";
self.onmessage=event=>{
 const {id,history}=event.data||{};
 try{
  if(!self.BBBDerivedRoadAux)importScripts("derived_road_aux.js?v=20261113-road-aux-adaptive-v1");
  if(!self.BBBSoftShoe)importScripts("soft_shoe_decision_runtime.js?v=20261113-strict-dual-v1");
  if(!self.BBB_SOFT_SHOE_XGB_MODEL)importScripts("soft_shoe_model_bundle.js?v=20261106-layered-v4");
  if(!self.BBBFrozenSequenceLSTM)importScripts("soft_shoe_frozen_lstm_runtime.js?v=20261106-layered-v4");
  if(!self.BBB_SOFT_SHOE_LSTM_MODEL)importScripts("soft_shoe_frozen_model_bundle.js?v=20261106-layered-v4");
  const report=self.BBBSoftShoe.run(history,{},self.BBB_SOFT_SHOE_XGB_MODEL,
    self.BBBFrozenSequenceLSTM,self.BBB_SOFT_SHOE_LSTM_MODEL);
  self.postMessage({id,report,road_aux:report.road_aux,
   road_confirmed:report.research_view?.road_confirmed===true,
   road_conflict:report.research_view?.road_conflict===true});
 }catch(e){self.postMessage({id,error:String(e?.message||e)});}
};
