/* Display-only causal EMA. Never writes formal bet probabilities or qualifications. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_PF_DISPLAY_EMA__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';const KEY='bgs_pf10000_display_ema_v1';
function createClient(storage){
 function apply(forecast,history,shoe,tag,alpha,maxDelta=.006){
  if(!forecast?.available||!Number.isFinite(forecast.directional?.B))return forecast;
  const raw=forecast.directional.B,tie=forecast.probabilities.T;if(!Number.isFinite(tie)||tie<0||tie>=1)return forecast;
  let journal={shoe,tag,records:[],pending:null};
  try{const saved=JSON.parse(storage?.getItem(KEY)||'null');if(saved?.shoe===shoe&&saved?.tag===tag&&Array.isArray(saved.records))journal=saved;
   const valid=r=>Number.isInteger(r.index)&&r.index>=0&&r.prefix===history.slice(0,r.index).join('');
   journal.records=journal.records.filter(r=>valid(r)&&r.index<history.length&&r.actual===history[r.index]);
   const pending=journal.pending;if(pending&&valid(pending)&&history.length>pending.index&&!journal.records.some(r=>r.index===pending.index))journal.records.push({...pending,actual:history[pending.index]});
   journal.records.sort((a,b)=>a.index-b.index);journal.records=journal.records.slice(-70);
   const previous=journal.records.at(-1),prior=previous?.index===history.length-1&&Number.isFinite(previous.q)?previous.q:null;
   const a=Math.max(.35,Math.min(.75,alpha)),limit=Math.max(0,Math.min(.01,maxDelta));let q=prior===null?raw:a*raw+(1-a)*prior;q=Math.max(raw-limit,Math.min(raw+limit,q));const signReset=(q-.5)*(raw-.5)<0;if(signReset)q=raw;
   journal.pending={index:history.length,prefix:history.join(''),q};storage?.setItem(KEY,JSON.stringify(journal));
   return {...forecast,directional:{B:q,P:1-q},probabilities:{B:(1-tie)*q,P:(1-tie)*(1-q),T:tie},direction:q===.5?'Balanced':q>.5?'B':'P',display_smoothing:{enabled:true,display_only:true,alpha:prior===null?1:a,raw_p_b:raw,smoothed_p_b:q,delta:q-raw,sign_reset:signReset,version:tag}};
  }catch(_){return {...forecast,display_smoothing:{enabled:false,display_only:true,error:'storage_error_raw_fallback'}};}
 }
 return {apply,clear:()=>storage?.removeItem(KEY)};
}
return {KEY,createClient};
});
