'use strict';
self.onmessage=async event=>{
 const {id,player,banker,settledRows}=event.data;
 try{
  if(!self.BBBPureCards){
   const r=await fetch('pure_card_eor_profile.json?v=20261105-local-card-v1');if(!r.ok)throw Error('eor_profile_load_failed');
   self.BBBCardEORProfile=await r.json();
   importScripts('pure_card_kernel.js?v=20261105-local-card-v1','pure_card_runtime.js?v=20261105-local-card-v1');
  }
  self.postMessage({id,report:self.BBBPureCards.run(player,banker,settledRows)});
 }catch(e){self.postMessage({id,error:String(e.message||e)});}
};
