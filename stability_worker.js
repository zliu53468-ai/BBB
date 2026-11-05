'use strict';
// Runtime modules are DOM-free but expose their browser UMD globals on window.
self.window=self;
self.onmessage=event=>{const {id,player,banker,options}=event.data;
 try{
  if(!self.BBBStability)importScripts(
   'quant256_online_runtime.js?v=20261106-se055-gap0256-v6',
   'shoe_distribution_guard.js?v=20261106-se055-gap0256-v6',
   'markov_residual_runtime.js?v=20261106-se055-gap0256-v6',
   'physics_forecast_first_runtime.js?v=20261106-se055-gap0256-v6',
   'deplete_mc_runtime.js?v=20261106-se055-gap0256-v6',
   'pf_eor_v4_runtime.js?v=20261106-se055-gap0256-v6',
   'stability_reference.js?v=20261106-se055-gap0256-v6',
   'stability_runtime.js?v=20261106-se055-gap0256-v6');
  const report=new self.BBBStability.StablePhysicalPointV4().analyze(player,banker,options);
  self.postMessage({id,report});
 }catch(e){self.postMessage({id,error:String(e.message||e)});}
};
