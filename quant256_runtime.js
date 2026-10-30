/**
 * BBBShoeQuantEngine V6 — isolated, deterministic 256D experimental engine.
 *
 * This models an *assumed* distribution when only B/P/T outcomes are provided.
 * Actual deck ranks, shoe penetration, pairs and Lucky Six are unobservable
 * from B/P/T. Inferred counts, EV and Kelly fractions are NOT calibrated
 * probabilities or evidence of a wagering edge.
 *
 * The production Bayesian Markov estimate() API and Frozen Core are untouched.
 */
(function(root,factory){
  "use strict";
  // The strict research-only veto is supplied by the existing compatibility
  // runtime. In browsers auth-client loads it before this experimental engine.
  const countApi=(root&&root.__BBB_BAYES_MARKOV__)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./bayes_markov_runtime"):null);
  const pfApi=(root&&root.__BBB_RUN_SURVIVAL_PF__)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./run_survival_pf_runtime"):null);
  const api=factory(countApi,pfApi);
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root){
    root.BBBShoeQuantEngine=api.BBBShoeQuantEngine;
    root.BBBQuantEngine=api.BBBShoeQuantEngine; // previous experimental adapter
  }
})(typeof window!=="undefined"?window:null,function(countApi,pfApi){
  "use strict";

  // 8 decks: each point 1..9 has 32 cards; 0 (10/J/Q/K) has 128.
  // "mid" is specifically 5,7,8. Nine is tracked separately: it accounts
  // for the 32 otherwise missing cards in the requested four-group scheme.
  const INITIAL=Object.freeze({low:128,mid:96,six:32,zero:128,nine:32});
  const GROUPS=Object.freeze(Object.keys(INITIAL));
  const TOTAL=416;
  const BASE=Object.freeze({B:0.4586,P:0.4462,T:0.0952});
  const COUNTS=Object.freeze({low:1,mid:-1,six:0,zero:0,nine:0});
  const ALPHA=0.0045;
  const SCHEMA="BBB_SHOE_QUANT_128_V6_CPSI";
  const SMALL=1e-9,clamp=(x,min,max)=>Math.max(min,Math.min(max,x));
  const finite=Number.isFinite;
  function groupOf(point){
    if(!Number.isInteger(point)||point<0||point>9)throw Error("card point must be 0..9");
    if(point===0)return "zero";
    if(point<=4)return "low";
    if(point===6)return "six";
    if(point===9)return "nine";
    return "mid";
  }
  function assertOutcome(outcome){
    if(!["B","P","T"].includes(outcome))throw Error("outcome must be B, P or T");
  }
  function own(obj,key){return Object.prototype.hasOwnProperty.call(obj,key);}
  function validateBoolean(observation,key){
    if(own(observation,key)&&observation[key]!==undefined&&typeof observation[key]!=="boolean")
      throw Error(key+" must be a boolean when observed");
  }
  class BBBShoeQuantEngine {
    constructor(){this.shoeNumber=0;this.reset();}
    reset(){
      this.shoeNumber++;
      this.estimatedCounts={...INITIAL};this.remainingCards=TOTAL;
      // Point-value counts are retained ONLY from explicitly observed hands.
      // B/P/T inferred fractional consumption never mutates these ranks.
      this.observedPointCounts=[128,32,32,32,32,32,32,32,32,32];
      this.runningCount=0;this.trueCount=0;this.pairCount=0;this.l6Index=0;
      this.history=[];this.lastSide=null;this.streak=0;this.estimatedRounds=0;this.observedRounds=0;
      this.pairKnownRounds=0;this.bankerPairKnownRounds=0;this.playerPairKnownRounds=0;this.bankerPairs=0;this.playerPairs=0;
      this.l6KnownRounds=0;this.l6ObservedHits=0;this.lastUpdate=null;
      this._refresh();return this;
    }
    get countSource(){
      return this.estimatedRounds===0&&this.observedRounds>0?"observed":
        this.observedRounds>0?"mixed":"estimated";
    }
    _refresh(){
      this.remainingCards=GROUPS.reduce((s,k)=>s+this.estimatedCounts[k],0);
      this.cardRatios=Object.fromEntries(GROUPS.map(k=>[k,this.estimatedCounts[k]/INITIAL[k]]));
      this.cardDensities=Object.fromEntries(GROUPS.map(k=>[k,
        this.remainingCards>0?this.estimatedCounts[k]/this.remainingCards:0]));
      this.trueCount=clamp(this.runningCount/Math.max(this.remainingCards/52,SMALL),-8,8);
    }
    /** Expected fractional removal with capacity reallocation, no negative decks. */
    _consume(draws,distribution){
      const removed=Object.fromEntries(GROUPS.map(k=>[k,0]));
      let left=Math.min(draws,this.remainingCards);
      for(let step=0;step<32&&left>SMALL;step++){
        const avail=GROUPS.filter(k=>this.estimatedCounts[k]>SMALL);
        if(!avail.length)break;
        let weights=avail.map(k=>Math.max(0,distribution[k]||0));
        let weightSum=weights.reduce((a,b)=>a+b,0);
        if(weightSum<SMALL){
          weights=avail.map(k=>this.estimatedCounts[k]);
          weightSum=weights.reduce((a,b)=>a+b,0);
        }
        const allocation=left;
        for(let i=0;i<avail.length;i++){
          const key=avail[i],take=Math.min(this.estimatedCounts[key],allocation*weights[i]/weightSum);
          this.estimatedCounts[key]-=take;removed[key]+=take;left-=take;
        }
      }
      if(left>1e-6)throw Error("card_mass_conservation_failed");
      return removed;
    }
    /**
     * One round, with optional actual observation. B/P/T-only deduction is a
     * heuristic—not card counting. Annealing cuts estimated DRAW size by 30%
     * from the fourth identical non-tie result, as requested. This is a model
     * assumption, not a change in actual physical cards dealt by the casino.
     */
    updateRound(outcome,observation={}){
      assertOutcome(outcome);
      if(!observation||typeof observation!=="object")throw Error("invalid observation");
      for(const key of ["bankerPair","playerPair","luckySix"])validateBoolean(observation,key);
      const cards=observation.cards;
      if(cards!==undefined&&(!Array.isArray(cards)||cards.length<4||cards.length>6||
        !cards.every(x=>Number.isInteger(x)&&x>=0&&x<=9)))throw Error("cards must contain 4..6 observed card points");
      if(outcome!=="T"){
        this.streak=outcome===this.lastSide?this.streak+1:1;
        this.lastSide=outcome;
      }
      const anneal=outcome!=="T"&&this.streak>3&&!cards?0.7:1;
      const nominal=outcome==="T"?4.5:4.94,draws=cards?cards.length:nominal*anneal;
      let removed,source;
      if(cards){
        removed=Object.fromEntries(GROUPS.map(k=>[k,0]));
        for(const point of cards){
          removed[groupOf(point)]++;
          if(this.observedPointCounts[point]<=0)throw Error("observed_point_exceeds_shoe");
        }
        for(const k of GROUPS)if(removed[k]>this.estimatedCounts[k]+SMALL)throw Error("observed_count_exceeds_remaining_"+k);
        for(const k of GROUPS)this.estimatedCounts[k]-=removed[k];
        for(const point of cards)this.observedPointCounts[point]--;
        this.observedRounds++;source="observed";
      }else{
        const distribution=outcome==="B"?
          {low:.45,mid:.35,zero:.20,six:0,nine:0}:
          outcome==="P"?
          {low:.40,mid:.45,zero:.15,six:0,nine:0}:
          {low:.30,mid:.30,zero:.40,six:0,nine:0};
        removed=this._consume(draws,distribution);
        this.estimatedRounds++;source="estimated";
      }
      // Exact requested +/- low vs mid proxy. With a P round at 4.94 cards:
      // 4.94*.40 - 4.94*.45 = -0.247. With B it is +0.494.
      this.runningCount+=GROUPS.reduce((s,k)=>s+removed[k]*COUNTS[k],0);
      // Pair events are distinct markets: do not infer the player pair rate from
      // banker-pair observations or vice versa. Missing is not a false event.
      if(typeof observation.bankerPair==="boolean"){
        this.bankerPairKnownRounds++;
        if(observation.bankerPair)this.bankerPairs++;
      }
      if(typeof observation.playerPair==="boolean"){
        this.playerPairKnownRounds++;
        if(observation.playerPair)this.playerPairs++;
      }
      if(typeof observation.bankerPair==="boolean"||typeof observation.playerPair==="boolean")
        this.pairKnownRounds++;
      this.pairCount=this.bankerPairs+this.playerPairs;
      if(typeof observation.luckySix==="boolean"){
        this.l6KnownRounds++;
        if(observation.luckySix)this.l6ObservedHits++;
      }
      // l6Index without card totals is only an opaque diagnostic heuristic.
      this.l6Index+=(outcome==="B"?.10:outcome==="P"?-.05:0);
      this.history.push(outcome);
      this._refresh();
      const update={outcome,source,estimated_only:source!=="observed",nominal_draws:nominal,
        expected_draws:draws,anneal_factor:anneal,deducted:removed,
        remaining_after:this.remainingCards,true_count:this.trueCount,
        reset_triggered:this.remainingCards<=60,reset_reason:null};
      if(update.reset_triggered){
        update.reset_reason="estimated_penetration_threshold_not_confirmed_shuffle";
        this.reset(); // requested auto reset; user must also reset on a real shuffle.
      }
      this.lastUpdate=update;
      return update;
    }
    survivalSignal(){
      if(!countApi||typeof countApi.countSurvivalHazard!=="function")
        throw Error("count_survival_hazard_unavailable");
      return countApi.countSurvivalHazard({
        history:this.history,trueCount:this.trueCount,
        zeroRatio:this.cardRatios.zero,countSource:this.countSource
      });
    }
    survivalFeatures(){
      const s=this.survivalSignal(),last=s.run_side;
      // Slot definitions shared with run_length_hazard.py.
      return [s.cpsi,s.hazard_rate,clamp(s.run_length/16,0,1),
        last==="B"?1:last==="P"?-1:0,
        clamp(s.aligned_tc/8,-1,1),clamp(s.zero_ratio/2,0,1),
        +s.observed_card_evidence,+s.active,+(s.direction!=="Skip"),
        +(s.direction===last&&last!==null),
        +(s.direction!=="Skip"&&s.direction!==last),
        s.follow_multiplier-1,s.break_multiplier-1,
        clamp(s.simulated_bet_fraction/.25,0,1),
        +s.pf_weighting_allowed,clamp(s.run_length/70,0,1)];
    }
    survivalParticleFilter(){
      const s=this.survivalSignal();
      if(!pfApi||typeof pfApi.estimate!=="function")
        throw Error("survival_pf_module_unavailable");
      return pfApi.estimate({history:this.history,count_source:this.countSource,
        pointCounts:this.observedPointCounts.slice(),run_side:s.run_side,
        run_length:s.run_length,follow_multiplier:s.follow_multiplier,
        break_multiplier:s.break_multiplier});
    }
    /** Exact 128-slot schema; V6 registers CPSI slots 16..31. */
    generateShoeContext(){
      const tc=this.trueCount,n=this.history.length,rr=this.remainingCards/TOTAL,progress=1-rr;
      // The first four coordinates are intentionally NONLINEAR, signed
      // count-only features; road data never enters these slots. Scaling
      // alone does not create predictive information from unobserved ranks.
      const sign=Math.sign(tc),square=tc*tc;
      const v=[
        square*sign,sign*square/(1+Math.abs(tc)),
        sign*Math.pow(Math.tanh(tc/3),2),sign*Math.log1p(square),
        rr,progress,clamp(n/80,0,1),Math.tanh(this.runningCount/32),
        Math.tanh(this.pairCount/8),Math.tanh(this.l6Index/8),
        this.observedRounds/Math.max(1,n),this.estimatedRounds/Math.max(1,n),
        clamp(this.streak/12,0,1),this.history.at(-1)==="B"?1:this.history.at(-1)==="P"?-1:0,
        this.cardDensities.low-this.cardDensities.mid,
        this.history.length?this.history.slice(-8).filter(x=>x==="T").length/Math.min(8,n):0
      ];
      for(const k of GROUPS){
        const ratio=this.cardRatios[k],density=this.cardDensities[k],initial=INITIAL[k]/TOTAL;
        v.push(ratio,density,density/initial,this.estimatedCounts[k]/TOTAL,
          ratio*ratio,density*density,density-initial,1-ratio);
      }
      for(const center of [-7,-5,-3,-1,1,3,5,7]){
        const z=(tc-center)/2;
        v.push(clamp(1-Math.abs(z),0,1),Math.exp(-.5*z*z),clamp((tc-center)/16,-1,1));
      }
      for(let i=0;i<8;i++){
        const center=(i+.5)/8,z=(progress-center)*8;
        v.push(clamp(1-Math.abs(z),0,1),Math.exp(-.5*z*z),clamp((progress-center)*2,-1,1));
      }
      for(const win of [3,5,8,12,16,24,48,70]){
        const h=this.history.slice(-win);
        for(const side of ["B","P","T"])
          v.push(h.length?h.filter(x=>x===side).length/h.length:0);
      }
      // Replace, not append: fixed count-conditioned CPSI/hazard slots
      // (17..32 by one-based indexing), preserving exactly 128D.
      const survival=this.survivalFeatures();
      for(let i=0;i<16;i++)v[16+i]=survival[i];
      if(v.length!==128||!v.every(finite))throw Error("invalid_quant_128D");
      return v;
    }
    injectShoeContext(context256,{experimental=false,allowUncalibrated=false,maskRoad=true}={}){
      if(!experimental&&!allowUncalibrated)throw Error("quant schema incompatible with frozen weights; opt in to research adapter");
      if(!Array.isArray(context256)||context256.length!==256||!context256.every(finite))throw Error("expected finite 256D");
      const vector=context256.slice();
      const shoe=this.generateShoeContext();
      for(let i=0;i<128;i++)vector[i]=shoe[i];
      // 129..144 are CPSI/HSMM diagnostics, NOT original road features.
      // 145..256 remain zero. Experimental LinUCB scores only count heads.
      if(maskRoad){
        for(let i=128;i<256;i++)vector[i]=0;
        const h=this.survivalFeatures();
        for(let i=0;i<16;i++)vector[128+i]=h[i];
      }
      return vector;
    }
    static fractionalKelly(win,loss,payout){
      if(![win,loss,payout].every(finite)||win<0||loss<0||win+loss>1+SMALL||payout<=0)throw Error("invalid Kelly");
      const ev=win*payout-loss;
      const quarter=ev>0?0.25*ev/(payout*(win+loss)):0;
      return {ev,raw_fraction:quarter,bet_fraction:ev>0?clamp(quarter,.05,.30):0,
        minimum_clip_applied:ev>0&&quarter<.05};
    }
    _sideIndicator(name,signal,events,known,reason){
      return {name,signal:!!signal,known_rounds:known,observed_events:events,
        bet_allowed:false,bet_fraction:0,calibrated:false,reason};
    }
    predictNext(){
      const tc=this.trueCount;
      const w_count=Math.min(.75,Math.abs(tc)*.18);
      const adjustment=clamp(tc*ALPHA,-.16,.16);
      const count_probability={B:BASE.B+adjustment,P:BASE.P-adjustment,T:BASE.T};
      const probabilities={
        B:BASE.B*(1-w_count)+count_probability.B*w_count,
        P:BASE.P*(1-w_count)+count_probability.P*w_count,
        T:BASE.T
      };
      const b=BBBShoeQuantEngine.fractionalKelly(probabilities.B,probabilities.P,.95);
      const p=BBBShoeQuantEngine.fractionalKelly(probabilities.P,probabilities.B,1);
      if(!countApi||typeof countApi.absoluteCountVeto!=="function")
        throw Error("absolute_count_veto_unavailable");
      // Single-source-of-truth: bayes_markov_runtime's experimental veto.
      // Existing estimate() remains intact for production compatibility.
      const veto=countApi.absoluteCountVeto(tc),direction=veto.direction;
      const value=direction==="B"?b:direction==="P"?p:(b.ev>=p.ev?b:p);
      const survival=this.survivalSignal();
      const runPF=this.survivalParticleFilter();
      // During a long run the survival gate takes precedence over V5's
      // absolute TC veto. No observed card points => abstain, not fabricate
      // an edge from the B/P/T streak. In other rounds V5 is unchanged.
      const inRun=survival.active;
      const finalDirection=inRun?survival.direction:direction;
      const gateCount=inRun?finalDirection!=="Skip":veto.count_passed;
      const gateEV=value.ev>0;
      const source=this.countSource;
      const tieRecent=this.history.slice(-12),tieHits=tieRecent.filter(x=>x==="T").length;
      const tieSignal=tieRecent.length>=8&&tieHits/tieRecent.length>.20;
      const bankerPairSignal=this.bankerPairKnownRounds>=10&&this.bankerPairs/this.bankerPairKnownRounds>.085;
      const playerPairSignal=this.playerPairKnownRounds>=10&&this.playerPairs/this.playerPairKnownRounds>.085;
      const l6Signal=this.l6KnownRounds>=10&&this.l6ObservedHits/this.l6KnownRounds>.12;
      return {
        version:"BBB_SHOE_QUANT_V6",context_schema:SCHEMA,experimental:true,validated:false,
        alpha:ALPHA,count_weight:w_count,count_source:source,
        baseline_probabilities:{...BASE},count_probability,probabilities,
        ev_banker:b.ev,ev_player:p.ev,ev_selected:value.ev,direction:finalDirection,
        direction_text:finalDirection==="Skip"?"觀望":finalDirection==="B"?"莊":"閒",
        reason:inRun?survival.reason:veto.reason,
        survival,particle_survival:runPF,
        // Absolute ±2.5 veto is the SOLE experimental direction authority;
        // EV is still calculated as a diagnostic but cannot force a bet.
        count_gate_passed:gateCount,ev_gate_passed:gateEV,
        research_direction_allowed:gateCount,
        bet_fraction:0,simulated_bet_fraction:inRun?
          survival.simulated_bet_fraction:veto.simulated_bet_fraction,
        kelly_raw_fraction:0,
        minimum_clip_applied:false,
        side_bets:{
          banker_pair:this._sideIndicator("banker_pair",bankerPairSignal,this.bankerPairs,this.bankerPairKnownRounds,"observed_banker_pair_rate_not_calibrated"),
          player_pair:this._sideIndicator("player_pair",playerPairSignal,this.playerPairs,this.playerPairKnownRounds,"observed_player_pair_rate_not_calibrated"),
          tie:this._sideIndicator("tie",tieSignal,tieHits,tieRecent.length,"outcome_window_only_not_calibrated"),
          lucky_six:this._sideIndicator("lucky_six",l6Signal,this.l6ObservedHits,this.l6KnownRounds,"observed_lucky_six_rate_not_calibrated")
        },
        estimated_shoe:{
          remaining_cards:this.remainingCards,runningCount:this.runningCount,trueCount:tc,
          cardRatios:{...this.cardRatios},cardDensities:{...this.cardDensities},
          pairCount:this.pairCount,l6Index:this.l6Index,
          estimatedRounds:this.estimatedRounds,observedRounds:this.observedRounds,
          estimated_only:source!=="observed"
        },
        gates:{true_count_abs_strictly_greater_than:2.5,count_passed:gateCount,
          ev_passed:gateEV,ev_is_diagnostic_only:true,neutral_forced_skip:!gateCount,
          direction_owner:inRun?"countSurvivalHazard":"absoluteCountVeto",
          physical_count_evidence_required_for_run:true,
          road_override_allowed:false,bet_fraction:0,simulated:true},
        notice:"V6 experimental survival only: B/P/T-only count has NO observed card support. CPSI, PF stress weights and forced follow/break scenarios are uncalibrated and NOT true hazard probabilities or a wagering advantage."
      };
    }
    static fromHistory(history){
      if(!Array.isArray(history))throw Error("history must be array");
      const engine=new BBBShoeQuantEngine();
      for(const row of history){
        if(typeof row==="string")engine.updateRound(row);
        else if(row&&typeof row==="object")engine.updateRound(row.outcome,row.observation||{});
        else throw Error("invalid history row");
      }
      return engine;
    }
  }
  return {BBBShoeQuantEngine,BBBQuantEngine:BBBShoeQuantEngine,
    QUANT_CONTEXT_SCHEMA:SCHEMA,QUANT_INITIAL_COUNTS:INITIAL,
    BASE_PROBABILITIES:BASE};
});
