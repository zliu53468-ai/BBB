/* Bounded, causal mixing on paired same-shoe prequential Brier evidence. */
(function(root,f){const api=f();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_XGB_ADAPTIVE__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';const finite=Number.isFinite,clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function mix(base,xgb,rows,history,shoe,tag,policyTag,noise,c={}){
 if(!finite(base)||base<0||base>1)throw Error('invalid_base_probability');
 const numeric=(k,f,a,b)=>clamp(finite(c[k])?c[k]:f,a,b),minimum=Math.round(numeric('min_samples',12,8,50)),window=Math.round(numeric('window',24,minimum,70)),prior=numeric('prior_samples',12,4,100),cap=numeric('max_weight',.25,0,.25),maxDelta=numeric('max_delta',.01,0,.01),minGain=numeric('min_brier_gain',.0005,0,.02),targetGain=numeric('target_brier_gain',.02,.001,.1);
 const low=numeric('noise_start',.55,0,.8),high=numeric('noise_stop',.85,low+.01,1),noiseGate=finite(noise)?clamp((high-noise)/(high-low),0,1):0,unique=new Map();
 for(const row of rows||[]){const i=row?.round_index-1,a=row?.bayes_markov?.xgb_adaptive;
  if(row?.shoe_id!==shoe||row.pf_primary_policy_tag!==policyTag||!Number.isInteger(i)||i<0||i>=history.length||row.history_fingerprint!==history.slice(0,i).join('')||row.actual_outcome!==history[i]||!['B','P'].includes(row.actual_outcome)||a?.model_tag!==tag||!finite(a.base_pb)||!finite(a.xgb_pb)||a.base_pb<0||a.base_pb>1||a.xgb_pb<0||a.xgb_pb>1)continue;
  unique.set(i,{i,a,y:row.actual_outcome==='B'?1:0});
 }
 const scored=[...unique.values()].sort((a,b)=>a.i-b.i).slice(-window),n=scored.length,baseLoss=n?scored.reduce((s,r)=>s+(r.a.base_pb-r.y)**2,0)/n:null,xgbLoss=n?scored.reduce((s,r)=>s+(r.a.xgb_pb-r.y)**2,0)/n:null,gain=n?baseLoss-xgbLoss:0,valid=finite(xgb)&&xgb>=0&&xgb<=1;
 const reliability=n/(n+prior),weight=c.enabled===true&&valid&&n>=minimum&&gain>minGain?cap*reliability*clamp((gain-minGain)/targetGain,0,1)*noiseGate:0,delta=valid?clamp(weight*(xgb-base),-maxDelta,maxDelta):0,value=clamp(base+delta,0,1),effective=valid&&Math.abs(xgb-base)>1e-12?(value-base)/(xgb-base):0;
 return {value,audit:{enabled:c.enabled===true,version:c.version||'XGB_ADAPTIVE_BRIER_V1',model_tag:tag,base_pb:base,xgb_pb:valid?xgb:null,samples:n,min_samples:minimum,window,base_brier:baseLoss,xgb_brier:xgbLoss,brier_gain:gain,noise_gate:noiseGate,reliability,proposed_weight:weight,weight:clamp(effective,0,cap)||0,max_weight:cap,delta:value-base,max_delta:maxDelta,reason:c.enabled!==true?'disabled':!valid?'xgb_unavailable':n<minimum?'insufficient_paired_predictions':gain<=minGain?'no_forward_brier_gain':noiseGate===0?'noise_gate_closed':'bounded_forward_brier_mix',validated:false}};
}
return {mix};});
