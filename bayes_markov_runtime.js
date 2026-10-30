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
/**
 * Quant256 V6 experimental CPSI + run-length hazard.
 *
 * This is a scenario/risk INDICATOR, not a physical law, calibrated hazard,
 * estimated win probability or staking recommendation. For B/P/T-only shoes,
 * observedRankEvidence=false and the module never issues follow/break orders.
 * The zero density criterion is a model input, not proof that 0s favour B.
 * No dependence on Bayesian Markov estimate() or any road features.
 */
function countSurvivalHazard({history=[],trueCount=0,zeroRatio=1,countSource='estimated'}={}){
 if(!Array.isArray(history)||history.some(x=>!['B','P','T'].includes(x)))
  throw Error('invalid_survival_history');
 if(!Number.isFinite(trueCount)||!Number.isFinite(zeroRatio)||zeroRatio<0)
  throw Error('invalid_survival_shoe');
 const tc=clamp(trueCount,-8,8),z=clamp(zeroRatio,0,2),observed=countSource==='observed';
 // Ties are neutral interruptions: count the most recent B/P streak over T.
 let side=null,n=0;
 for(let i=history.length-1;i>=0;i--){
  const outcome=history[i];
  if(outcome==='T')continue;
  if(side===null){side=outcome;n=1;}
  else if(side===outcome)n++;
  else break;
 }
 const active=n>=4&&side!==null,sign=side==='B'?1:side==='P'?-1:0;
 const aligned=sign*tc;
 const conflict=active&&aligned<=-3;
 const supportive=active&&aligned>=1.5&&z>=.75;
 const blurry=Math.abs(tc)<=1.5;
 // These are stress multipliers. They may only be applied to particles
 // when every prior round has actual observed card POINTS, not B/P/T alone.
 let follow=1,breakFactor=1;
 if(active&&observed&&aligned>3){follow=1.5;}
 if(active&&observed&&aligned< -2.5){follow=.4;breakFactor=2;}
 const depth=clamp((n-3)/9,0,1);
 const advantage=clamp((aligned-1.5)/4,0,1);
 const cpsi=active?clamp((.2+.8*advantage)*(.6+.4*clamp(z,0,1)),0,1):0;
 const hazard=active?clamp(.55+.35*depth-.4*cpsi+
  (conflict ? .30 : 0),0,1):0;
 let signal='Skip',reason='inactive';
 if(!active)reason='run_length_below_4';
 else if(!observed)reason='no_observed_card_points';
 else if(conflict){signal=side==='B'?'P':'B';reason='observed_count_conflict_break_scenario';}
 else if(blurry)reason='count_ambiguous_skip';
 else if(supportive){signal=side;reason='observed_count_aligned_follow_scenario';}
 else reason='insufficient_count_or_zero_density';
 const continuation=signal!=='Skip'&&signal===side;
 const simulated_fraction=signal==='Skip'?0:
  continuation?clamp(.15+.01*(n-4),.15,.25):.15;
 return Object.freeze({
  version:'BBB_CPSI_HAZARD_V6',active,run_length:n,run_side:side,
  observed_card_evidence:observed,count_source:countSource,tc,
  zero_ratio:z,aligned_tc:aligned,cpsi,hazard_rate:hazard,
  follow_multiplier:follow,break_multiplier:breakFactor,
  direction:signal,direction_text:signal==='Skip'?'觀望':signal==='B'?'莊':'閒',
  reason,uncalibrated:true,simulated_only:true,
  bet_fraction:0,simulated_bet_fraction:simulated_fraction,
  road_override_allowed:false,pf_weighting_allowed:active&&observed,
  risk_is_probability:false
 });
}
return {estimate,absoluteCountVeto,countSurvivalHazard};});
