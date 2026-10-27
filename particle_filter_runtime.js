(() => {
"use strict";

/*
Fixed-500 stateless baccarat particle posterior for the browser runtime.

Important semantics:
- Every prediction call starts from 500 fresh eight-deck particles.
- No particle array, weights, ESS, or resampling state is carried from the
  previous prediction. The current B/P/T history is replayed from scratch.
- B/P/T is the only observation. Hidden ranks/points are conditional
  hypotheses, never claimed as reconstructed casino cards.
- The particle layer is a bounded correction to Direct Physics and is not
  allowed to flip the Direct Physics B/P side by itself.
*/

const VERSION="PARTICLE500_STATELESS_V1";
const PARTICLE_COUNT=500;
const LIKELIHOOD_DRAWS=2;
const FALLBACK_DRAWS=2;
const FORECAST_DRAWS=1;
const ESS_RESAMPLE_THRESHOLD=.50;
const TIE_LIKELIHOOD_INFLUENCE=.30;
const MIN_BLEND=.06;
const MAX_BLEND=.16;
const RANKS=13;
const INITIAL_PER_RANK=32;
const OUTCOMES=new Set(["B","P","T"]);
const BLEND_MODES=new Set(["quality","off","fixed_0.06","fixed_0.10","fixed_0.16","uncertainty_only"]);
const clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));

function normalizeBlendMode(mode){return BLEND_MODES.has(String(mode||"quality"))?String(mode||"quality"):"quality";}
function fixedBlendWeight(mode){
  if(mode==="fixed_0.06")return .06;
  if(mode==="fixed_0.10")return .10;
  if(mode==="fixed_0.16")return .16;
  return null;
}

function seedFromHistory(history){
  let h=2166136261>>>0;
  for(const token of history){
    h^=token==="B"?66:token==="P"?80:84;
    h=Math.imul(h,16777619)>>>0;
  }
  h^=history.length+0x9e3779b9;
  return h>>>0;
}
function rng32(seed){
  let x=(seed||0x6d2b79f5)>>>0;
  return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};
}
function rankValue(index){const rank=index+1;return rank<=9?rank:0;}
function bankerDraws(total,playerThird){
  if(playerThird===null)return total<=5;
  if(total<=2)return true;
  if(total===3)return playerThird!==8;
  if(total===4)return playerThird>=2&&playerThird<=7;
  if(total===5)return playerThird>=4&&playerThird<=7;
  if(total===6)return playerThird>=6&&playerThird<=7;
  return false;
}
function drawRank(counts,rng){
  let total=0;for(let i=0;i<RANKS;i++)total+=counts[i];
  if(total<=0)throw new Error("particle shoe exhausted");
  let pick=Math.floor(rng()*total),sum=0;
  for(let i=0;i<RANKS;i++){sum+=counts[i];if(pick<sum){counts[i]--;return i;}}
  return RANKS-1;
}
function deal(base,rng){
  const counts=base.slice();
  const p1=drawRank(counts,rng),b1=drawRank(counts,rng),p2=drawRank(counts,rng),b2=drawRank(counts,rng);
  const ranks=[p1,b1,p2,b2];
  let pt=(rankValue(p1)+rankValue(p2))%10,bt=(rankValue(b1)+rankValue(b2))%10;
  let p3=null;
  const natural=pt>=8||bt>=8;
  if(!natural){
    if(pt<=5){const r=drawRank(counts,rng);ranks.push(r);p3=rankValue(r);pt=(pt+p3)%10;}
    if(bankerDraws(bt,p3)){const r=drawRank(counts,rng);ranks.push(r);bt=(bt+rankValue(r))%10;}
  }
  return {outcome:bt>pt?"B":pt>bt?"P":"T",playerPoint:pt,bankerPoint:bt,ranks,counts,cardCount:ranks.length};
}
function normaliseWeights(values){
  let sum=0;for(const v of values)sum+=Math.max(0,+v||0);
  if(sum<=1e-300)return Array(values.length).fill(1/Math.max(1,values.length));
  return values.map(v=>Math.max(0,+v||0)/sum);
}
function systematicResample(particles,consumed,weights,rng){
  const n=particles.length,cdf=Array(n);let s=0;
  for(let i=0;i<n;i++){s+=weights[i];cdf[i]=s;}
  const start=rng()/n,next=Array(n),nextConsumed=Array(n);let j=0;
  for(let i=0;i<n;i++){
    const pos=start+i/n;while(j<n-1&&pos>cdf[j])j++;
    next[i]=particles[j].slice();nextConsumed[i]=consumed[j];
  }
  return {particles:next,consumed:nextConsumed,weights:Array(n).fill(1/n)};
}
function entropy3(block){
  let h=0;
  for(const p of block){if(p>1e-12)h-=p*Math.log(p);}
  return clip(h/Math.log(3));
}
function normalizeBlock(a){
  const x=a.map(v=>Math.max(0,+v||0)),sum=x.reduce((p,c)=>p+c,0);
  return sum>1e-12?x.map(v=>v/sum):Array(a.length).fill(1/a.length);
}
function blendBlock(direct,posterior,start,n,w){
  const d=normalizeBlock(direct.slice(start,start+n)),p=normalizeBlock(posterior.slice(start,start+n));
  const b=d.map((v,i)=>(1-w)*v+w*p[i]);
  const z=normalizeBlock(b);for(let i=0;i<n;i++)direct[start+i]=z[i];
}
function preserveWinnerDirection(corrected,direct){
  const dB=clip(direct[23]),dP=clip(direct[24]),dT=clip(direct[25]),dMass=Math.max(1e-9,dB+dP),dDir=dB/dMass;
  let b=clip(corrected[23]),p=clip(corrected[24]),t=clip(corrected[25]),mass=Math.max(1e-9,b+p),dir=b/mass;
  let blocked=false;
  if(dDir>.5&&dir<.5){dir=.5;blocked=true;}
  if(dDir<.5&&dir>.5){dir=.5;blocked=true;}
  const directionalMass=clip(1-t);
  corrected[23]=directionalMass*dir;corrected[24]=directionalMass*(1-dir);corrected[25]=t;
  return blocked;
}
function posteriorForecast(particles,consumed,weights,rng,draws=FORECAST_DRAWS){
  const out=Array(48).fill(0);let mass=0;
  let consumedMean=0,consumedSq=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i];consumedMean+=w*consumed[i];consumedSq+=w*consumed[i]*consumed[i];
    for(let d=0;d<draws;d++){
      const hand=deal(particles[i],rng),m=w/draws;mass+=m;
      out[{4:0,5:1,6:2}[hand.cardCount]]+=m;
      out[3+hand.playerPoint]+=m;out[13+hand.bankerPoint]+=m;
      out[23+(hand.outcome==="B"?0:hand.outcome==="P"?1:2)]+=m;
      for(const rank of hand.ranks)out[26+rank]+=m;
      const diff=hand.bankerPoint-hand.playerPoint;
      out[46]+=m*diff/9;out[47]+=m*Math.abs(diff)/9;
    }
  }
  if(mass<=1e-12)throw new Error("particle forecast empty");
  for(const [start,n] of [[0,3],[3,10],[13,10],[23,3]])for(let i=0;i<n;i++)out[start+i]/=mass;
  for(let i=26;i<39;i++)out[i]/=mass;
  out[43]=consumedMean;
  let low=0,high=0,total=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i],counts=particles[i];let localTotal=0,localLow=0,localHigh=0;
    for(let r=0;r<RANKS;r++){localTotal+=counts[r];if(r<5)localLow+=counts[r];if(r>=8)localHigh+=counts[r];}
    total+=w*localTotal;low+=w*localLow;high+=w*localHigh;
  }
  out[44]=total>0?low/total:0;out[45]=total>0?high/total:0;
  out[46]/=mass;out[47]/=mass;
  const variance=Math.max(0,consumedSq-consumedMean*consumedMean);
  return {physics:out,consumedStd:Math.sqrt(variance)};
}
function estimate(history,directPhysics,mode="quality",config={}){
  const seq=(Array.isArray(history)?history:[]).map(x=>String(x||"").toUpperCase()).filter(x=>OUTCOMES.has(x));
  if(!Array.isArray(directPhysics)||directPhysics.length!==48)throw new Error("particle500 requires Direct Physics 48D");
  const blendMode=normalizeBlendMode(mode);
  const likelihoodDraws=Math.round(clip(config.likelihood_draws??LIKELIHOOD_DRAWS,2,8));
  const fallbackDraws=Math.round(clip(config.fallback_draws??FALLBACK_DRAWS,2,8));
  if(blendMode==="off")return {
    physics:directPhysics.map(v=>+v||0),posterior:null,
    diagnostics:{version:VERSION,enabled:false,particle_filter_enabled:false,particle_count:0,
      policy:"direct_physics_only",persistent_state:false,rebuild_from_scratch:false,
      history_rounds:seq.length,history_fingerprint:seq.join(""),particle_blend_mode:blendMode,blend_weight:0}
  };
  const rng=rng32(seedFromHistory(seq));
  let particles=Array.from({length:PARTICLE_COUNT},()=>Array(RANKS).fill(INITIAL_PER_RANK));
  let consumed=Array(PARTICLE_COUNT).fill(0),weights=Array(PARTICLE_COUNT).fill(1/PARTICLE_COUNT);
  let resamplingCount=0,lastEssRatio=1,tieUpdates=0;

  for(const actual of seq){
    const next=Array(PARTICLE_COUNT),nextConsumed=Array(PARTICLE_COUNT),rawWeights=Array(PARTICLE_COUNT);
    for(let i=0;i<PARTICLE_COUNT;i++){
      const base=particles[i],proposals=[],matches=[];
      for(let d=0;d<likelihoodDraws;d++){const hand=deal(base,rng);proposals.push(hand);if(hand.outcome===actual)matches.push(hand);}
      const matchCount=matches.length;
      let likelihood=(matchCount+.20)/(likelihoodDraws+.60);
      if(!matches.length){
        for(let d=0;d<fallbackDraws;d++){const hand=deal(base,rng);if(hand.outcome===actual){matches.push(hand);break;}}
      }
      const chosen=matches.length?matches[Math.floor(rng()*matches.length)]:proposals[Math.floor(rng()*proposals.length)];
      if(!matches.length)likelihood=Math.max(likelihood,.015);
      if(actual==="T"){likelihood=(1-TIE_LIKELIHOOD_INFLUENCE)+TIE_LIKELIHOOD_INFLUENCE*likelihood;tieUpdates++;}
      next[i]=chosen.counts;nextConsumed[i]=consumed[i]+chosen.cardCount;rawWeights[i]=weights[i]*Math.max(1e-8,likelihood);
    }
    weights=normaliseWeights(rawWeights);particles=next;consumed=nextConsumed;
    let sq=0;for(const w of weights)sq+=w*w;lastEssRatio=clip((1/Math.max(1e-12,sq))/PARTICLE_COUNT);
    if(lastEssRatio<ESS_RESAMPLE_THRESHOLD){
      const r=systematicResample(particles,consumed,weights,rng);particles=r.particles;consumed=r.consumed;weights=r.weights;resamplingCount++;lastEssRatio=1;
    }
  }

  const forecastDraws=Math.round(clip(config.forecast_draws??FORECAST_DRAWS,1,5));
  const forecast=posteriorForecast(particles,consumed,weights,rng,forecastDraws),posterior=forecast.physics;
  const winner=normalizeBlock(posterior.slice(23,26));
  const spread=clip(forecast.consumedStd/18);
  const posteriorUncertainty=clip(.55*entropy3(winner)+.30*(1-lastEssRatio)+.15*spread);
  const quality=clip(.45*lastEssRatio+.35*(1-posteriorUncertainty)+.20*(1-spread));
  const minimum=clip(config.min_blend??MIN_BLEND,0,MAX_BLEND),maximum=clip(config.max_blend??MAX_BLEND,minimum,MAX_BLEND);
  const sameSide=(directPhysics[23]-directPhysics[24])*(posterior[23]-posterior[24])>0;
  const signalGate=config.signal_gate===true?(sameSide?clip(lastEssRatio)*clip(1-posteriorUncertainty):0):1;
  const qualityBlendWeight=clip(minimum+(maximum-minimum)*quality*signalGate,minimum,maximum);
  const forcedBlendWeight=fixedBlendWeight(blendMode);
  const blendWeight=forcedBlendWeight===null?qualityBlendWeight:forcedBlendWeight;

  const corrected=directPhysics.map(v=>+v||0);
  let directionFlipBlocked=false;
  if(blendMode==="uncertainty_only"){
    corrected[42]=clip(.65*clip(directPhysics[42])+.35*posteriorUncertainty);
    corrected[44]=(1-blendWeight)*corrected[44]+blendWeight*posterior[44];
    corrected[45]=(1-blendWeight)*corrected[45]+blendWeight*posterior[45];
  }else{
    blendBlock(corrected,posterior,0,3,blendWeight);
    blendBlock(corrected,posterior,3,10,blendWeight);
    blendBlock(corrected,posterior,13,10,blendWeight);
    blendBlock(corrected,posterior,23,3,blendWeight);
    directionFlipBlocked=preserveWinnerDirection(corrected,directPhysics);
    for(let i=26;i<39;i++)corrected[i]=(1-blendWeight)*corrected[i]+blendWeight*posterior[i];
    corrected[43]=(1-blendWeight)*corrected[43]+blendWeight*posterior[43];
    corrected[44]=(1-blendWeight)*corrected[44]+blendWeight*posterior[44];
    corrected[45]=(1-blendWeight)*corrected[45]+blendWeight*posterior[45];
    corrected[46]=(1-blendWeight)*corrected[46]+blendWeight*posterior[46];
    corrected[47]=(1-blendWeight)*corrected[47]+blendWeight*posterior[47];
    corrected[42]=clip(.65*clip(directPhysics[42])+.35*posteriorUncertainty);
    const pB=clip(corrected[23]),pP=clip(corrected[24]);
    corrected[39]=pB*.95-pP;corrected[40]=pP-pB;corrected[41]=corrected[39]-corrected[40];
  }

  return {
    physics:corrected,
    posterior,
    diagnostics:{
      version:VERSION,enabled:true,particle_filter_enabled:true,particle_count:PARTICLE_COUNT,
      policy:"stateless_rebuild_each_prediction",persistent_state:false,rebuild_from_scratch:true,
      history_rounds:seq.length,history_fingerprint:seq.join(""),likelihood_draws:likelihoodDraws,
      fallback_draws:fallbackDraws,forecast_draws:forecastDraws,ess_resample_threshold:ESS_RESAMPLE_THRESHOLD,
      posterior_update_method:"observed_bpt_likelihood_and_resampling",observations_processed:seq.length,
      resampling_count:resamplingCount,recent_ess_ratio:lastEssRatio,posterior_uncertainty:posteriorUncertainty,
      particle_quality:quality,signal_gate:signalGate,same_direction:sameSide,blend_min:minimum,blend_max:maximum,blend_weight:blendWeight,quality_blend_weight:qualityBlendWeight,particle_blend_mode:blendMode,
      tie_soft_update:true,tie_likelihood_influence:TIE_LIKELIHOOD_INFLUENCE,
      tie_observation_count:seq.filter(x=>x==="T").length,direction_flip_blocked:directionFlipBlocked,
      direct_directional_pb:clip(directPhysics[23]/Math.max(1e-9,directPhysics[23]+directPhysics[24])),
      corrected_directional_pb:clip(corrected[23]/Math.max(1e-9,corrected[23]+corrected[24])),
      physics_uncertainty:corrected[42],expected_consumed_cards:corrected[43],
      physical_ev_banker:corrected[39],physical_ev_player:corrected[40],physical_ev_gap:corrected[41],
    }
  };
}

// Score-conditioned PF. The legacy 13-rank PF above remains frozen and is
// still used whenever the user selects B/P/T compatibility mode or no score
// observations are available. This path tracks the ten baccarat point classes
// (0..9). Its transition likelihood is exact for independent draws and a
// high-quality approximation to a finite shoe; hidden cards are sampled and
// removed from each particle without replacement.
const POINT_CLASSES=10;
const INITIAL_POINT_COUNTS=[128,32,32,32,32,32,32,32,32,32];
const POINT_UPDATE_VERSION="POINT_SCORE_CONDITIONED_PF500_V1";
function seedFromPointHistory(history,observations){
  let h=seedFromHistory(history);
  for(let i=0;i<history.length;i++){
    const o=observations?.[i]||{};
    for(const v of [o.player_score,o.banker_score,o.card_count]){
      const n=v===null||v===undefined||v===""?255:Number(v);
      h^=(Number.isFinite(n)?n:254)&255;h=Math.imul(h,16777619)>>>0;
    }
  }
  return h>>>0;
}
function pointProbabilities(counts){
  const total=counts.reduce((a,b)=>a+b,0);
  return total>0?counts.map(v=>v/total):Array(POINT_CLASSES).fill(.1);
}
function pointPairTotals(probabilities){
  const out=Array(10).fill(0);
  for(let a=0;a<10;a++)for(let b=0;b<10;b++)out[(a+b)%10]+=probabilities[a]*probabilities[b];
  return out;
}
function pointOutcome(player,banker){return banker>player?"B":player>banker?"P":"T";}
function canonicalPointObservation(actual,raw={}){
  const asInteger=(v,lo,hi)=>{
    if(v===null||v===undefined||v==="")return null;
    const n=Number(v);return Number.isInteger(n)&&n>=lo&&n<=hi?n:null;
  };
  const player=asInteger(raw.player_score??raw.playerPoint,0,9);
  const banker=asInteger(raw.banker_score??raw.bankerPoint,0,9);
  const cardCount=asInteger(raw.card_count??raw.cardCount,4,6);
  const hasPlayer=player!==null,hasBanker=banker!==null;
  if(hasPlayer!==hasBanker)return {outcome:actual,card_count:cardCount,invalid_partial_score:true};
  if(hasPlayer&&pointOutcome(player,banker)!==actual)return {outcome:actual,card_count:cardCount,invalid_score_outcome:true};
  return {outcome:actual,player_score:hasPlayer?player:null,banker_score:hasBanker?banker:null,card_count:cardCount,
    score_observed:hasPlayer&&hasBanker};
}
function enumeratePointPaths(counts,observation=null){
  const total=counts.reduce((a,b)=>a+b,0),probabilities=pointProbabilities(counts),pair=pointPairTotals(probabilities);
  const paths=[];let likelihood=0;
  const cardFilter=observation?.card_count??null;
  const add=(p,b,cards,p0,b0,p3,b3,weight)=>{
    if(weight<=0)return;
    if(cardFilter!==null&&cards!==cardFilter)return;
    if(observation?.score_observed){
      if(p!==observation.player_score||b!==observation.banker_score)return;
    }else if(observation?.outcome&&pointOutcome(p,b)!==observation.outcome)return;
    likelihood+=weight;paths.push({p,b,cards,p0,b0,p3,b3,weight});
  };
  if(total<6)return {likelihood:0,paths,probabilities,pair};
  for(let p0=0;p0<10;p0++){
    if(pair[p0]<=0)continue;
    for(let b0=0;b0<10;b0++){
      const start=pair[p0]*pair[b0];if(start<=0)continue;
      if(p0>=8||b0>=8){add(p0,b0,4,p0,b0,null,null,start);continue;}
      if(p0<=5){
        for(let p3=0;p3<10;p3++){
          const player=(p0+p3)%10,afterPlayer=start*probabilities[p3];if(afterPlayer<=0)continue;
          if(bankerDraws(b0,p3)){
            for(let b3=0;b3<10;b3++)add(player,(b0+b3)%10,6,p0,b0,p3,b3,afterPlayer*probabilities[b3]);
          }else add(player,b0,5,p0,b0,p3,null,afterPlayer);
        }
      }else if(bankerDraws(b0,null)){
        for(let b3=0;b3<10;b3++)add(p0,(b0+b3)%10,5,p0,b0,null,b3,start*probabilities[b3]);
      }else add(p0,b0,4,p0,b0,null,null,start);
    }
  }
  return {likelihood,paths,probabilities,pair};
}
function samplePairForPointTotal(counts,target,rng){
  const total=counts.reduce((a,b)=>a+b,0);if(total<2)return null;
  const options=[];let mass=0;
  for(let a=0;a<10;a++){
    if(counts[a]<=0)continue;
    for(let b=0;b<10;b++){
      const remaining=counts[b]-(a===b?1:0);
      if((a+b)%10!==target||remaining<=0)continue;
      const weight=counts[a]*remaining;mass+=weight;options.push({a,b,weight});
    }
  }
  if(mass<=0)return null;
  let pick=rng()*mass;
  for(const item of options){pick-=item.weight;if(pick<0){counts[item.a]--;counts[item.b]--;return true;}}
  const last=options.at(-1);counts[last.a]--;counts[last.b]--;return true;
}
function weightedChoice(items,rng){
  let mass=0;for(const item of items)mass+=item.weight;
  if(mass<=0)return null;
  let pick=rng()*mass;
  for(const item of items){pick-=item.weight;if(pick<0)return item;}
  return items.at(-1)||null;
}
function samplePointHand(counts,paths,rng){
  if(!paths.length)return null;
  // A few retries cover the small finite-deck correction to the with-replacement
  // path weights. Point classes are aggregated, so no rank-level inference is made.
  for(let attempt=0;attempt<24;attempt++){
    const path=weightedChoice(paths,rng);if(!path)return null;
    const remaining=counts.slice();
    if(!samplePairForPointTotal(remaining,path.p0,rng)||!samplePairForPointTotal(remaining,path.b0,rng))continue;
    if(path.p3!==null){if(remaining[path.p3]<=0)continue;remaining[path.p3]--;}
    if(path.b3!==null){if(remaining[path.b3]<=0)continue;remaining[path.b3]--;}
    return {counts:remaining,card_count:path.cards,player_score:path.p,banker_score:path.b};
  }
  return null;
}
function pointPosteriorForecast(particles,consumed,weights){
  const out=Array(48).fill(0),meanCounts=Array(10).fill(0);let consumedMean=0,consumedSq=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i],counts=particles[i],total=counts.reduce((a,b)=>a+b,0);
    if(total<6)continue;
    consumedMean+=w*consumed[i];consumedSq+=w*consumed[i]*consumed[i];
    for(let c=0;c<10;c++)meanCounts[c]+=w*counts[c];
    const {paths}=enumeratePointPaths(counts,null);
    for(const hand of paths){
      const m=w*hand.weight;
      out[{4:0,5:1,6:2}[hand.cards]]+=m;
      out[3+hand.p]+=m;out[13+hand.b]+=m;
      out[23+(hand.b>hand.p?0:hand.p>hand.b?1:2)]+=m;
      out[46]+=m*(hand.b-hand.p)/9;out[47]+=m*Math.abs(hand.b-hand.p)/9;
    }
    const probs=pointProbabilities(counts);
    // A=point 1, 2..9 map directly, and zero points split evenly across 10/J/Q/K.
    for(let value=1;value<=9;value++)out[25+value]+=w*probs[value];
    for(let rank=9;rank<13;rank++)out[26+rank]+=w*probs[0]/4;
    out[44]+=w*(counts.slice(1,6).reduce((a,b)=>a+b,0)/total);
    out[45]+=w*((counts[0]+counts[9])/total);
  }
  for(const [start,n] of [[0,3],[3,10],[13,10],[23,3]]){
    let sum=0;for(let i=0;i<n;i++)sum+=out[start+i];
    if(sum>0)for(let i=0;i<n;i++)out[start+i]/=sum;
  }
  out[43]=consumedMean;out[46]=clip(out[46]);out[47]=clip(out[47]);
  const variance=Math.max(0,consumedSq-consumedMean*consumedMean);
  return {physics:out,consumedStd:Math.sqrt(variance),mean_point_counts:meanCounts};
}
function estimatePoints(history,directPhysics,mode="quality",config={},observations=[]){
  const seq=(Array.isArray(history)?history:[]).map(x=>String(x||"").toUpperCase()).filter(x=>OUTCOMES.has(x));
  if(!Array.isArray(directPhysics)||directPhysics.length!==48)throw new Error("point PF requires Direct Physics 48D");
  if(!Array.isArray(observations))throw new Error("point PF observations must be an array");
  const blendMode=normalizeBlendMode(mode);
  const entries=seq.map((outcome,i)=>canonicalPointObservation(outcome,observations[i]||{}));
  const informative=entries.filter(x=>x.score_observed||x.card_count!==null).length;
  if(!informative){
    const fallback=estimate(seq,directPhysics,blendMode,{...config,score_update_enabled:false});
    return {...fallback,diagnostics:{...fallback.diagnostics,score_update_fallback_reason:"no_point_or_card_count_observations"}};
  }
  if(blendMode==="off")return {
    physics:directPhysics.map(v=>+v||0),posterior:null,
    diagnostics:{version:POINT_UPDATE_VERSION,enabled:false,particle_filter_enabled:false,particle_count:0,
      policy:"direct_physics_only",persistent_state:false,rebuild_from_scratch:false,score_update_enabled:false,
      history_rounds:seq.length,history_fingerprint:seq.join(""),point_observations_processed:0,particle_blend_mode:blendMode,blend_weight:0}
  };
  const rng=rng32(seedFromPointHistory(seq,entries));
  let particles=Array.from({length:PARTICLE_COUNT},()=>INITIAL_POINT_COUNTS.slice());
  let consumed=Array(PARTICLE_COUNT).fill(0),weights=Array(PARTICLE_COUNT).fill(1/PARTICLE_COUNT);
  let resamplingCount=0,lastEssRatio=1,lastPostResampleEssRatio=1;
  for(const observation of entries){
    const next=new Array(PARTICLE_COUNT),nextConsumed=new Array(PARTICLE_COUNT),rawWeights=new Array(PARTICLE_COUNT);
    for(let i=0;i<PARTICLE_COUNT;i++){
      const prior=particles[i],likelihood=enumeratePointPaths(prior,observation);
      rawWeights[i]=weights[i]*likelihood.likelihood;
      const sampled=samplePointHand(prior,likelihood.paths,rng);
      next[i]=sampled?.counts||prior.slice();
      nextConsumed[i]=consumed[i]+(sampled?.card_count||0);
    }
    const weightMass=rawWeights.reduce((a,b)=>a+b,0);
    if(!(weightMass>1e-280)){
      const fallback=estimate(seq,directPhysics,blendMode,{...config,score_update_enabled:false});
      return {...fallback,diagnostics:{...fallback.diagnostics,score_update_enabled:false,
        score_update_fallback_reason:"point_conditioned_likelihood_collapsed",point_observations_processed:entries.filter(x=>x.score_observed).length}};
    }
    weights=normaliseWeights(rawWeights);particles=next;consumed=nextConsumed;
    let sq=0;for(const w of weights)sq+=w*w;
    lastEssRatio=clip((1/Math.max(1e-12,sq))/PARTICLE_COUNT);lastPostResampleEssRatio=lastEssRatio;
    if(lastEssRatio<ESS_RESAMPLE_THRESHOLD){
      const sampled=systematicResample(particles,consumed,weights,rng);
      particles=sampled.particles;consumed=sampled.consumed;weights=sampled.weights;resamplingCount++;lastPostResampleEssRatio=1;
    }
  }
  const forecast=pointPosteriorForecast(particles,consumed,weights),posterior=forecast.physics;
  if(!posterior.every(Number.isFinite))throw new Error("invalid point posterior forecast");
  const winner=normalizeBlock(posterior.slice(23,26));
  const spread=clip(forecast.consumedStd/18);
  const posteriorUncertainty=clip(.55*entropy3(winner)+.30*(1-lastEssRatio)+.15*spread);
  const quality=clip(.45*lastEssRatio+.35*(1-posteriorUncertainty)+.20*(1-spread));
  posterior[42]=posteriorUncertainty;
  const pB=posterior[23],pP=posterior[24];
  posterior[39]=.95*pB-pP;posterior[40]=pP-pB;posterior[41]=posterior[39]-posterior[40];
  const bptOnly=estimate(seq,directPhysics,blendMode,{...config,score_update_enabled:false});
  const bptProb=bptOnly.posterior?.slice(23,26)||[];
  const pointProb=posterior.slice(23,26),delta=bptProb.length===3?pointProb.map((v,i)=>v-bptProb[i]):null;
  const result={physics:posterior,posterior,
    diagnostics:{version:POINT_UPDATE_VERSION,enabled:true,particle_filter_enabled:true,particle_count:PARTICLE_COUNT,
      policy:"stateless_point_conditioned_rebuild_each_prediction",persistent_state:false,rebuild_from_scratch:true,
      history_rounds:seq.length,history_fingerprint:seq.join(""),observations_processed:seq.length,
      point_observations_processed:entries.filter(x=>x.score_observed).length,
      card_count_only_observations:entries.filter(x=>!x.score_observed&&x.card_count!==null).length,
      bpt_only_observations:entries.filter(x=>!x.score_observed&&x.card_count===null).length,
      posterior_update_method:"ten_class_score_pair_likelihood_and_conditional_hand_sampling",
      likelihood_approximation:"exact_with_replacement_baccarat_transition_dp",
      conditional_card_sampling:"score_conditioned_point_classes_without_replacement",
      total_cards_conditioning_used:entries.filter(x=>x.card_count!==null).length>0,
      observed_point_history:entries.map(x=>({player_score:x.player_score,banker_score:x.banker_score,card_count:x.card_count,outcome:x.outcome})),
      remaining_point_counts:forecast.mean_point_counts.map(v=>Number(v.toFixed(3))),
      resampling_count:resamplingCount,recent_ess_ratio:lastEssRatio,post_resample_ess_ratio:lastPostResampleEssRatio,posterior_uncertainty:posteriorUncertainty,
      particle_quality:quality,blend_min:0,blend_max:0,blend_weight:1,particle_blend_mode:"point_posterior_primary",
      point_vs_bpt_diagnostic:{bpt_only_method:"legacy_13_rank_pf500_bpt_likelihood",bpt_only_probabilities:bptProb.length===3?{B:bptProb[0],P:bptProb[1],T:bptProb[2]}:null,
        point_conditioned_probabilities:{B:pointProb[0],P:pointProb[1],T:pointProb[2]},probability_delta:delta,
        max_absolute_probability_delta:delta?Math.max(...delta.map(Math.abs)):null,
        winner_probability_l1_delta:delta?delta.reduce((a,v)=>a+Math.abs(v),0):null,
        direction_changed:!!delta&&((bptProb[0]-bptProb[1])*(pointProb[0]-pointProb[1])<0)},
      direct_directional_pb:clip(directPhysics[23]/Math.max(1e-9,directPhysics[23]+directPhysics[24])),
      corrected_directional_pb:clip(pointProb[0]/Math.max(1e-9,pointProb[0]+pointProb[1])),
      physics_uncertainty:posteriorUncertainty,expected_consumed_cards:posterior[43],
      physical_ev_banker:posterior[39],physical_ev_player:posterior[40],physical_ev_gap:posterior[41]}
  };
  return result;
}
function validatePointObservation(playerScore,bankerScore,cardCount=null){
  const p=Number(playerScore),b=Number(bankerScore),n=cardCount===null||cardCount===""?null:Number(cardCount);
  if(!Number.isInteger(p)||p<0||p>9||!Number.isInteger(b)||b<0||b>9)return {valid:false,reason:"scores_must_be_0_to_9"};
  if(n!==null&&(!Number.isInteger(n)||![4,5,6].includes(n)))return {valid:false,reason:"card_count_must_be_4_5_or_6"};
  const outcome=pointOutcome(p,b),obs={outcome,player_score:p,banker_score:b,card_count:n,score_observed:true};
  const valid=enumeratePointPaths(INITIAL_POINT_COUNTS,obs).likelihood>0;
  return {valid,reason:valid?null:"score_pair_card_count_impossible_under_baccarat_rules",outcome};
}

if(typeof window!=="undefined")window.__BGS_PARTICLE500__={version:VERSION,particleCount:PARTICLE_COUNT,estimate,estimatePoints,validatePointObservation};
})();
