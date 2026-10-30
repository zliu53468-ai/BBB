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
return {estimate};});

/**
 * BBBQuantEngine V2: causal 8-deck composition estimator and observed-rank counter.
 * Outcomes B/P/T alone cannot reveal dealt ranks. Proxies are labelled uncalibrated;
 * no betting signal is emitted without card observations for the complete shoe.
 * This module does NOT replace __BBB_BAYES_MARKOV__.estimate().
 */
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)Object.assign(module.exports,api);
  if(root)root.BBBQuantEngine=api.BBBQuantEngine;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const INITIAL=Object.freeze({zero:128,low:128,neutral:32,six:32,mid:96});
const GROUPS=Object.keys(INITIAL),TOTAL=416,BASE=Object.freeze({B:0.458597,P:0.446247,T:0.095156});
const ALPHA=0.0045,SCHEMA="BBB_QUANT_128_V2",EPS=1e-9;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),finite=Number.isFinite;
const TAG=Object.freeze({zero:0,low:1,neutral:0,six:-1,mid:-1});
function group(rank){
  if(!Number.isInteger(rank)||rank<0||rank>9)throw Error("card point must be 0..9");
  if(rank===0)return "zero";
  if(rank<=4)return "low";
  if(rank===5)return "neutral";
  if(rank===6)return "six";
  return "mid";
}
class BBBQuantEngine{
  constructor(){this.shoeNumber=0;this.reset();}
  reset(){
    this.shoeNumber++;this.counts={...INITIAL};this.remainingCards=TOTAL;
    this.runningCount=0;this.trueCount=0;this.observedRounds=0;this.estimatedRounds=0;
    this.pairObserved=0;this.bankerPairs=0;this.playerPairs=0;
    this.luckySixObserved=0;this.luckySixHits=0;this.history=[];
    this.lastSide=null;this.streak=0;this.lastUpdate=null;this._ratios();return this;
  }
  _ratios(){
    this.remainingCards=GROUPS.reduce((s,k)=>s+this.counts[k],0);
    this.cardRatios=Object.fromEntries(GROUPS.map(k=>[k,this.counts[k]/INITIAL[k]]));
    this.cardDensities=Object.fromEntries(GROUPS.map(k=>[k,this.remainingCards?this.counts[k]/this.remainingCards:0]));
    this.trueCount=this.runningCount/Math.max(1,this.remainingCards/52);
  }
  _consume(draws,tilt){
    let left=Math.min(draws,this.remainingCards);
    const taken=Object.fromEntries(GROUPS.map(k=>[k,0]));
    for(let step=0;step<32&&left>EPS;step++){
      const available=GROUPS.filter(k=>this.counts[k]>EPS);
      if(!available.length)break;
      const weights=available.map(k=>this.counts[k]*Math.max(0,tilt[k]??1));
      let den=weights.reduce((a,b)=>a+b,0);
      if(den<=EPS){for(let i=0;i<available.length;i++)weights[i]=this.counts[available[i]];den=weights.reduce((a,b)=>a+b,0);}
      const pass=left;
      for(let i=0;i<available.length;i++){
        const k=available[i],take=Math.min(this.counts[k],pass*weights[i]/den);
        this.counts[k]-=take;taken[k]+=take;left-=take;
      }
    }
    if(left>1e-6)throw Error("expected_card_conservation_failed");
    return taken;
  }
  updateRound(outcome,observation={}){
    if(!["B","P","T"].includes(outcome))throw Error("outcome must be B/P/T");
    if(!observation||typeof observation!=="object")throw Error("invalid_observation");
    const cards=observation.cards;
    if(cards!==undefined&&(!Array.isArray(cards)||cards.length<4||cards.length>6||!cards.every(x=>Number.isInteger(x)&&x>=0&&x<=9)))throw Error("cards must contain 4..6 observed points");
    for(const k of ["bankerPair","playerPair","luckySix"]){
      if(observation[k]!==undefined&&typeof observation[k]!=="boolean")throw Error(k+" must be boolean");
    }
    if(outcome!=="T"){this.streak=outcome===this.lastSide?this.streak+1:1;this.lastSide=outcome;}
    let deducted,draws,source;
    if(cards){
      draws=cards.length;const occurrences=Object.fromEntries(GROUPS.map(k=>[k,0]));
      for(const rank of cards)occurrences[group(rank)]++;
      for(const k of GROUPS)if(occurrences[k]>this.counts[k]+EPS)throw Error("observed_count_exceeds_remaining_"+k);
      deducted={...occurrences};
      for(const k of GROUPS)this.counts[k]-=deducted[k];
      this.observedRounds++;source="observed";
    }else{
      draws=outcome==="T"?4.5:4.94;
      const factors=outcome==="B"?{low:1.12,mid:0.88,six:1.06,neutral:0.96,zero:1.04}:
        outcome==="P"?{low:0.90,mid:1.08,six:0.96,neutral:1.02,zero:1.00}:
        {zero:1,low:1,neutral:1,six:1,mid:1};
      const impact=outcome!=="T"&&this.streak>3?0.7:1;
      const tilt=Object.fromEntries(GROUPS.map(k=>[k,1+impact*(factors[k]-1)]));
      deducted=this._consume(draws,tilt);this.estimatedRounds++;source="estimated";
    }
    this.runningCount+=GROUPS.reduce((s,k)=>s+deducted[k]*TAG[k],0);
    if(typeof observation.bankerPair==="boolean"||typeof observation.playerPair==="boolean"){
      this.pairObserved++;if(observation.bankerPair)this.bankerPairs++;
      if(observation.playerPair)this.playerPairs++;
    }
    if(typeof observation.luckySix==="boolean"){
      this.luckySixObserved++;if(observation.luckySix)this.luckySixHits++;
    }
    this.history.push(outcome);this._ratios();
    const beforeReset=this.remainingCards;
    const update={outcome,source,estimated_only:source==="estimated",deducted,draws,remaining_after:beforeReset,
      true_count:this.trueCount,reset_triggered:beforeReset<=60};
    // A depletion threshold is NOT evidence that a real casino shuffled.
    // Require explicit reset at shuffle; refuse further card depletion past zero.
    this.lastUpdate=update;return update;
  }
  get countSource(){return this.estimatedRounds===0&&this.observedRounds>0?"observed":this.observedRounds?"mixed":"estimated";}
  generateShoeContext(){
    const tc=this.trueCount,rr=this.remainingCards/TOTAL,progress=1-rr,n=this.history.length;
    const v=[
      Math.pow(tc,2)*Math.sign(tc),tc/8,Math.tanh(tc/4),Math.tanh(this.runningCount/32),
      rr,progress,clamp(n/70,0,1),this.observedRounds/Math.max(1,n),
      this.countSource==="observed"?1:0,this.countSource==="estimated"?1:0,
      Math.tanh((this.bankerPairs-this.playerPairs)/6),Math.tanh(this.luckySixHits/4),
      this.history.at(-1)==="B"?1:this.history.at(-1)==="P"?-1:0,
      clamp(this.streak/12,0,1),this.pairObserved/Math.max(1,n),this.luckySixObserved/Math.max(1,n)
    ];
    for(const k of GROUPS){
      const r=this.cardRatios[k],d=this.cardDensities[k],initial=INITIAL[k]/TOTAL;
      v.push(r,d,d/initial,this.counts[k]/TOTAL,r*r,d*d,d-initial,(1-r));
    }
    for(const center of [-7,-5,-3,-1,1,3,5,7]){
      const z=(tc-center)/2;v.push(clamp(1-Math.abs(z),0,1),Math.exp(-.5*z*z),clamp((tc-center)/16,-1,1));
    }
    for(let i=0;i<8;i++){
      const center=(i+0.5)/8,z=(progress-center)*8;
      v.push(clamp(1-Math.abs(z),0,1),Math.exp(-.5*z*z),clamp((progress-center)*2,-1,1));
    }
    for(const w of [3,5,8,12,16,24,48,70]){
      const h=this.history.slice(-w);
      for(const side of ["B","P","T"])v.push(h.length?h.filter(x=>x===side).length/h.length:0);
    }
    if(v.length!==128||!v.every(finite))throw Error("quant_128_invalid");
    return v;
  }
  injectShoeContext(context256,{experimental=false}={}){
    if(!experimental)throw Error("quant V2 feature schema incompatible with frozen weights");
    if(!Array.isArray(context256)||context256.length!==256||!context256.every(finite))throw Error("expected 256D");
    return [...this.generateShoeContext(),...context256.slice(128)];
  }
  static fractionalKelly(win,loss,payout){
    if(![win,loss,payout].every(finite)||win<0||loss<0||win+loss>1+EPS||payout<=0)throw Error("invalid_kelly");
    const ev=payout*win-loss,raw=ev>0?0.25*ev/(payout*(win+loss)):0;
    return {ev,raw_fraction:raw,bet_fraction:ev>0?clamp(raw,0.05,0.30):0,
      minimum_clip_applied:ev>0&&raw<0.05};
  }
  _rate(events,known,prior){return known?{observed_events:events,observed_rounds:known,rate:(events+prior*16)/(known+16),calibrated:false}:{observed_events:0,observed_rounds:0,rate:null,calibrated:false};}
  predictNext(){
    const tc=this.trueCount,weight=Math.min(0.7,Math.abs(tc)*0.15);
    const shift=clamp(tc*ALPHA,-0.15,0.15);
    const countProbability={B:BASE.B+shift,P:BASE.P-shift,T:BASE.T};
    const probabilities={B:BASE.B*(1-weight)+countProbability.B*weight,
      P:BASE.P*(1-weight)+countProbability.P*weight,T:BASE.T};
    const banker=BBBQuantEngine.fractionalKelly(probabilities.B,probabilities.P,0.95);
    const player=BBBQuantEngine.fractionalKelly(probabilities.P,probabilities.B,1);
    const choice=banker.ev>=player.ev?"B":"P",selected=choice==="B"?banker:player;
    const gateTC=Math.abs(tc)>=2,gateEV=selected.ev>0.005;
    const completeObserved=this.countSource==="observed";
    const allowed=gateTC&&gateEV&&completeObserved,reason=!gateTC?"true_count_below_2":
      !gateEV?"ev_below_0_005":!completeObserved?"unobserved_card_composition":"observed_gate_passed";
    const theoretical=gateTC&&gateEV?choice:"Skip";
    const direction=allowed?choice:"Skip";
    return {
      version:"BBB_QUANT_V2",context_schema:SCHEMA,experimental:true,validated:false,
      alpha:ALPHA,count_weight:weight,count_source:this.countSource,
      probabilities,count_probability:countProbability,baseline_probabilities:{...BASE},
      ev_banker:banker.ev,ev_player:player.ev,source_reliability:completeObserved?"observed_card_points":"uncalibrated_outcome_proxy",
      direction,direction_text:direction==="Skip"?"觀望":direction==="B"?"莊":"閒",
      theoretical_direction:theoretical,reason,bet_fraction:allowed?selected.bet_fraction:0,
      hypothetical_fraction:theoretical!=="Skip"?selected.bet_fraction:0,
      kelly_raw_fraction:allowed?selected.raw_fraction:0,minimum_clip_applied:allowed&&selected.minimum_clip_applied,
      side_bets:{
        banker_pair:this._rate(this.bankerPairs,this.pairObserved,0.074),
        player_pair:this._rate(this.playerPairs,this.pairObserved,0.074),
        tie:this._rate(this.history.filter(x=>x==="T").length,this.history.length,BASE.T),
        lucky_six:this._rate(this.luckySixHits,this.luckySixObserved,0.054)
      },
      side_bet_stake_fraction:0,
      estimated_shoe:{remaining_cards:this.remainingCards,cardRatios:{...this.cardRatios},
        cardDensities:{...this.cardDensities},runningCount:this.runningCount,trueCount:tc,
        observedRounds:this.observedRounds,estimatedRounds:this.estimatedRounds,estimated_only:!completeObserved},
      gates:{true_count_abs_min:2,ev_min_exclusive:0.005,observed_cards_required:true},
      notice:"No rank/side-bet inference is possible from B/P/T alone. Model coefficients are uncalibrated and not evidence of positive expected value."
    };
  }
  static fromHistory(history){const e=new BBBQuantEngine();for(const x of history)e.updateRound(x);return e;}
}
return {BBBQuantEngine,QUANT_CONTEXT_SCHEMA:SCHEMA,QUANT_INITIAL_COUNTS:INITIAL};
});
