/* Causal finite-order expert mixture; not a full CTW implementation. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_CONTEXT_MIXTURE__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const tokens=['B','P','T'],clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function estimate(seq,c={}){
 if(!Array.isArray(seq)||!seq.every(t=>tokens.includes(t)))throw Error('mixture_invalid_history');
 const cfg=c.context_mixture_v1||{},num=(v,f,a,b)=>Number.isFinite(v)?clamp(v,a,b):f;
 const prior=c.prior||{B:.45,P:.45,T:.10};if(!tokens.every(t=>Number.isFinite(prior[t])&&prior[t]>=0)||Math.abs(tokens.reduce((s,t)=>s+prior[t],0)-1)>1e-9||!tokens.every(t=>prior[t]>0))throw Error('mixture_invalid_prior');
 const history=seq.slice(-Math.floor(num(c.window,200,4,500))),minimum=Math.floor(num(c.min_observations,12,12,500)),strength=num(c.prior_strength,24,1,100),maximum=Math.floor(num(c.order,3,1,3));
 const eta=num(cfg.learning_rate,.25,0,1),cap=num(cfg.evidence_cap,8,0,20),supportStrength=num(cfg.support_strength,24,1,100);
 const validPriors=Array.isArray(cfg.order_priors)&&cfg.order_priors.length===3&&cfg.order_priors.every(v=>Number.isFinite(v)&&v>0),orderPriors=validPriors?cfg.order_priors:[.6,.3,.1];
 const adapt=c.history_adaptation_v1,decayOn=adapt?.enabled===true;
 const countDecay=decayOn?num(adapt.count_decay,.97,.90,1):1,scoreDecay=decayOn?num(adapt.score_decay,.96,.90,1):1;
 const experts=Array.from({length:maximum},(_,i)=>({order:i+1,table:new Map(),log_evidence:0,evaluated_predictions:0}));
 // For result at t, score the distribution formed using indices < t, THEN observe t.
 for(let t=0;t<history.length;t++)for(const e of experts){
  if(decayOn)e.log_evidence*=scoreDecay;
  if(t<e.order)continue;
  const context=history.slice(t-e.order,t).join('');
  const entry=e.table.get(context)||{counts:{B:0,P:0,T:0},raw:0,last:t};
  const factor=countDecay**Math.max(0,t-1-entry.last),counts=entry.counts;
  for(const k of tokens)counts[k]*=factor;
  const n=tokens.reduce((sum,k)=>sum+counts[k],0);
  if(entry.raw>=minimum){const probability=(counts[history[t]]+strength*prior[history[t]])/(n+strength);e.log_evidence=clamp(e.log_evidence+Math.log(probability/prior[history[t]]),-cap,cap);e.evaluated_predictions++;}
  for(const k of tokens)counts[k]*=countDecay;
  counts[history[t]]++;entry.raw++;entry.last=t;e.table.set(context,entry);
 }
 const rows=experts.map(e=>{
  const context=history.slice(-e.order).join(''),entry=e.table.get(context)||{counts:{B:0,P:0,T:0},raw:0,last:history.length};
  const factor=countDecay**Math.max(0,history.length-1-entry.last),counts=Object.fromEntries(tokens.map(k=>[k,entry.counts[k]*factor])),n=tokens.reduce((sum,k)=>sum+counts[k],0),raw=entry.raw,eligible=raw>=minimum;
  const probabilities=Object.fromEntries(tokens.map(t=>[t,(counts[t]+strength*prior[t])/(n+strength)])),support=n/(n+supportStrength);
  return {order:e.order,context,counts,observations:raw,eligible,probabilities,log_evidence:e.log_evidence,evaluated_predictions:e.evaluated_predictions,support,log_weight:eligible?Math.log(orderPriors[e.order-1])+eta*e.log_evidence+Math.log(support):null,weight:0,...(decayOn?{raw_observations:raw,weighted_observations:n}:{})};
 });
 const available=rows.filter(r=>r.eligible),backoff=rows.slice().reverse().map(r=>({order:r.order,observations:r.observations,eligible:r.eligible}));
 const diagnostics={enabled:true,version:'CAUSAL_CONTEXT_MIXTURE_V1',learning_rate:eta,evidence_cap:cap,support_strength:supportStrength,order_priors:orderPriors.slice(),first_order_aux_bypassed:true,...(decayOn?{history_adaptation:{enabled:true,count_decay:countDecay,score_decay:scoreDecay}}:{}),experts:rows};
 if(!available.length){const r=rows[0];return {available:false,reason:'insufficient_road_context',history_rounds:seq.length,observations:r.observations,minimum,backoff,order:1,counts:r.counts,context_mixture:diagnostics};}
 const max=Math.max(...available.map(r=>r.log_weight)),total=available.reduce((s,r)=>s+Math.exp(r.log_weight-max),0);for(const r of available)r.weight=Math.exp(r.log_weight-max)/total;
 const probabilities=Object.fromEntries(tokens.map(t=>[t,available.reduce((s,r)=>s+r.weight*r.probabilities[t],0)])),q=probabilities.B/(probabilities.B+probabilities.P),dominant=available.reduce((a,b)=>b.weight>a.weight?b:a);
 return {available:true,reason:'context_mixture_ready',history_rounds:seq.length,observations:dominant.observations,minimum,backoff,order:dominant.order,counts:dominant.counts,prior:{...prior},prior_strength:strength,probabilities,directional_pb:q,direction:Math.abs(q-.5)<1e-12?'Balanced':q>.5?'B':'P',context_mixture:diagnostics};
}
function assess(seq,c={},forecast=null){
 const cfg=c.history_adaptation_v1;if(cfg?.enabled!==true||c.context_mixture_v1?.enabled!==true)return null;
 const num=(v,f,a,b)=>Number.isFinite(v)?clamp(v,a,b):f;
 const window=Math.floor(num(cfg.short_window,16,12,30)),minimum=Math.floor(num(cfg.short_min_context,6,6,20)),extra=Math.floor(num(cfg.min_extra_history,8,4,30));
 if(!Array.isArray(seq)||!seq.every(t=>tokens.includes(t)))throw Error('adaptation_invalid_history');
 const audit={enabled:true,short_window:window,history_length:seq.length,available:false,forced_skip:false,reason:'insufficient_window_support'};
 if(seq.length<window+extra||!forecast?.available||!Number.isFinite(forecast.directional_pb))return audit;
 const recent=seq.slice(-window),context=recent.at(-1),counts={B:0,P:0,T:0};for(let i=1;i<recent.length;i++)if(recent[i-1]===context)counts[recent[i]]++;
 const n=tokens.reduce((sum,k)=>sum+counts[k],0),bp=counts.B+counts.P;if(n<minimum||bp<minimum)return {...audit,short_counts:counts,short_observations:n};
 const strength=num(cfg.short_prior_strength,6,1,24),priorB=num(c.prior?.B,.45,.001,.999),priorP=num(c.prior?.P,.45,.001,.999),q=(counts.B+priorB*strength)/(bp+(priorB+priorP)*strength),long=forecast.directional_pb;
 const gap=Math.abs(q-long),margin=num(cfg.direction_margin,.015,.005,.10),threshold=num(cfg.disagreement_gap,.06,.03,.20),opposite=(q-.5)*(long-.5)<0&&Math.abs(q-.5)>=margin&&Math.abs(long-.5)>=margin;
 const conflict=opposite&&gap>=threshold;
 return {...audit,available:true,short_counts:counts,short_observations:n,short_directional_pb:q,long_directional_pb:long,disagreement:gap,threshold,opposite,forced_skip:conflict,reason:conflict?'short_long_direction_conflict':'no_material_conflict',probability_changed:false};
}
return {estimate,assess};
});
