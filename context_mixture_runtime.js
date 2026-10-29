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
 const experts=Array.from({length:maximum},(_,i)=>({order:i+1,table:new Map(),log_evidence:0,evaluated_predictions:0}));
 // For result at t, score the distribution formed using indices < t, THEN observe t.
 for(let t=0;t<history.length;t++)for(const e of experts){if(t<e.order)continue;const context=history.slice(t-e.order,t).join(''),counts=e.table.get(context)||{B:0,P:0,T:0},n=tokens.reduce((s,k)=>s+counts[k],0);if(n>=minimum){const probability=(counts[history[t]]+strength*prior[history[t]])/(n+strength);e.log_evidence=clamp(e.log_evidence+Math.log(probability/prior[history[t]]),-cap,cap);e.evaluated_predictions++;}counts[history[t]]++;e.table.set(context,counts);}
 const rows=experts.map(e=>{const context=history.slice(-e.order).join(''),counts=e.table.get(context)||{B:0,P:0,T:0},n=tokens.reduce((s,k)=>s+counts[k],0),eligible=n>=minimum,probabilities=Object.fromEntries(tokens.map(t=>[t,(counts[t]+strength*prior[t])/(n+strength)])),support=n/(n+supportStrength);return {order:e.order,context,counts:{...counts},observations:n,eligible,probabilities,log_evidence:e.log_evidence,evaluated_predictions:e.evaluated_predictions,support,log_weight:eligible?Math.log(orderPriors[e.order-1])+eta*e.log_evidence+Math.log(support):null,weight:0};});
 const available=rows.filter(r=>r.eligible),backoff=rows.slice().reverse().map(r=>({order:r.order,observations:r.observations,eligible:r.eligible}));
 const diagnostics={enabled:true,version:'CAUSAL_CONTEXT_MIXTURE_V1',learning_rate:eta,evidence_cap:cap,support_strength:supportStrength,order_priors:orderPriors.slice(),first_order_aux_bypassed:true,experts:rows};
 if(!available.length){const r=rows[0];return {available:false,reason:'insufficient_road_context',history_rounds:seq.length,observations:r.observations,minimum,backoff,order:1,counts:r.counts,context_mixture:diagnostics};}
 const max=Math.max(...available.map(r=>r.log_weight)),total=available.reduce((s,r)=>s+Math.exp(r.log_weight-max),0);for(const r of available)r.weight=Math.exp(r.log_weight-max)/total;
 const probabilities=Object.fromEntries(tokens.map(t=>[t,available.reduce((s,r)=>s+r.weight*r.probabilities[t],0)])),q=probabilities.B/(probabilities.B+probabilities.P),dominant=available.reduce((a,b)=>b.weight>a.weight?b:a);
 return {available:true,reason:'context_mixture_ready',history_rounds:seq.length,observations:dominant.observations,minimum,backoff,order:dominant.order,counts:dominant.counts,prior:{...prior},prior_strength:strength,probabilities,directional_pb:q,direction:Math.abs(q-.5)<1e-12?'Balanced':q>.5?'B':'P',context_mixture:diagnostics};
}
return {estimate};
});
