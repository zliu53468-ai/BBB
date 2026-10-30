/* Causal Bayesian Markov context estimator. No training or cross-shoe state. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_BAYES_MARKOV__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function estimate(history,corePB,physics,config={}){
 if(!Array.isArray(history)||history.some(x=>!['B','P','T'].includes(x)))throw Error('invalid_history');
 if(!Array.isArray(physics)||physics.length!==48||!physics.every(Number.isFinite)||!Number.isFinite(corePB))throw Error('invalid_prior_inputs');
 const mass=physics[23]+physics[24];if(mass<=0||physics[23]<0||physics[24]<0)throw Error('invalid_physics_winner');
 const number=(k,f,a,b)=>clamp(Number.isFinite(config[k])?config[k]:f,a,b);
 const strength=number('prior_strength',12,4,100),pw=number('physics_prior_weight',.75,0,1),theory=.458597/(.458597+.446247);
 const u=clamp(physics[42],0,1),prior=clamp((1-u)*(pw*physics[23]/mass+(1-pw)*clamp(corePB,0,1))+u*theory,.05,.95),orders=[];
 for(let order=1;order<=3;order++){
  const context=history.slice(-order);let B=0,P=0,T=0;
  if(context.length===order)for(let i=order;i<history.length;i++)if(context.every((x,j)=>history[i-order+j]===x)){if(history[i]==='B')B++;else if(history[i]==='P')P++;else T++;}
  const n=B+P,minimum=Math.round(clamp(Number.isFinite(config.min_support?.[order])?config.min_support[order]:order+3,2,30)),posterior=(B+strength*prior)/(n+strength);
  orders.push({order,context:context.join(''),banker:B,player:P,ties:T,support:n,min_support:minimum,eligible:context.length===order&&n>=minimum,markov_pb:n?B/n:null,bayesian_pb:posterior,prior_strength:strength,reliability:n/(n+strength)});
 }
 const selected=orders.slice().reverse().find(x=>x.eligible),raw=selected?selected.bayesian_pb:prior;
 return {enabled:true,version:config.version||'BAYES_MARKOV_1_3_V1',selected_order:selected?.order||0,selected_support:selected?.support||0,raw_pb:raw,prior_pb:prior,physics_prior_weight:pw,orders,fallback:selected?'highest_supported_context':'core_physics_prior',whole_shoe:true,causal:true,shared_evidence_counted_once:true,xgb_influence:0,validated:false};
}
/**
 * Opt-in Quant256 V5: ABSOLUTE COUNT-ONLY RESEARCH VETO.
 *
 * Deliberately independent of estimate(): no Markov, roads, Bayesian blend,
 * probability threshold, or EV hypothesis can override this classification.
 * TC from B/P/T-only data is heuristic and NOT observed shoe composition;
 * the 15-30% fraction is a synthetic display metric, NEVER a real stake.
 *
 * Strict comparisons: exactly +/- 2.5 is neutral and MUST Skip.
 */
function absoluteCountVeto(trueCount){
 if(!Number.isFinite(trueCount))throw Error('invalid_true_count');
 const tc=clamp(trueCount,-8,8),abs=Math.abs(tc);
 const count_passed=abs>2.5;
 const direction=tc>2.5?'B':tc< -2.5?'P':'Skip';
 const simulated_bet_fraction=count_passed?
  clamp(.15+.03*(abs-2.5),.15,.30):0;
 return Object.freeze({
  version:'BBB_ABSOLUTE_COUNT_VETO_V5',
  count_only:true,simulated_only:true,validated:false,
  tc,threshold_exclusive:2.5,count_passed,
  direction,direction_text:direction==='Skip'?'觀望':direction==='B'?'莊':'閒',
  reason:count_passed?'experimental_count_threshold_exceeded':'count_neutral_forced_skip',
  road_override_allowed:false,ev_override_allowed:false,
  bet_fraction:0,simulated_bet_fraction
 });
}
return {estimate,absoluteCountVeto};});
