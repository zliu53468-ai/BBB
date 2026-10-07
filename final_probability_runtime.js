(() => {
"use strict";

const CORE=(typeof window!=="undefined")?window.__BGS256_CONTINUATION_TEST__:null;
if(!CORE||typeof CORE.hazardChoose!=="function")return;

const VERSION="PHYSICS_57D_FINAL_PROBABILITY_V10_EARLY35_GATE";
const PHYSICS_URL="physics_multitask_model.json";
const FINAL56_URL="final_probability_model.json";
const ORIGINAL7_NAMES=["core_p_b","round_index","estimated_total_hands","remaining_ratio","sx_markov_p_same","stage","depth"];
const PHYSICS_NAMES=[
"cards_p4","cards_p5","cards_p6",
...Array.from({length:10},(_,i)=>"player_point_p"+i),
...Array.from({length:10},(_,i)=>"banker_point_p"+i),
"winner_p_b","winner_p_p","winner_p_t",
..."A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(x=>"next_rank_expected_"+x),
"physical_ev_banker","physical_ev_player","physical_ev_gap","particle_uncertainty",
"shoe_consumed_cards","remaining_low_rank_density","remaining_high_rank_density",
"expected_point_diff_norm","expected_abs_point_diff_norm"];
const LEGACY_PHYSICS_NAMES=[
...PHYSICS_NAMES.slice(0,39),
"next_suit_ratio_spades","next_suit_ratio_hearts","next_suit_ratio_diamonds","next_suit_ratio_clubs",
...PHYSICS_NAMES.slice(43)];
const PHYSICS_INDEX=Object.fromEntries(PHYSICS_NAMES.map((name,index)=>[name,index]));
const LEGACY_PHYSICS_INDEX=Object.fromEntries(LEGACY_PHYSICS_NAMES.map((name,index)=>[name,index]));
const EXTENDED_NAMES=["core_p_b_external","shoe_progress_weight",...ORIGINAL7_NAMES.slice(1).map(x=>"original7_"+x),...PHYSICS_NAMES,"physics_noise_score"];
const HISTORY_WINDOW=64,HISTORY_INPUT_DIM=213,PHYSICS_DIM=48,FEATURE_DIM=57;
const ESTIMATED_TOTAL_HANDS_MIN=50,ESTIMATED_TOTAL_HANDS_MAX=70,ESTIMATED_PLAYABLE_CARDS=416-60;
const DEFAULT_BOUNDS=[.40,.60],EARLY_BOUNDS=[.45,.55],LATE_CLEAN_BOUNDS=[.35,.65],PHYSICS_NOISE_LOW_THRESHOLD=.78;
const SHOE_ERROR_CORRECTION_VERSION=1,SHOE_ERROR_CORRECTION_MEMORY=10;
const SNAPSHOT_SCHEMA_VERSION=7;
const TRAINING_KEY="bgs_xgb_final_training_v5",PENDING_KEY="bgs_xgb_final_pending_v5";
const SHOE_KEY="bgs_xgb_final_shoe_id_v5",CUT_KEY="bgs_xgb_estimated_total_hands_v1";
const STORAGE_KEY="bgs256d_short_x_dynamic_v23",MAX_TRAINING_ROWS=15000;
const EARLY35_VERSION=1,EARLY35_BRIDGE_START=35,EARLY35_BRIDGE_END=40;
const EARLY35_EVIDENCE_ANCHORS=[[1,.15],[5,.22],[10,.32],[15,.45],[20,.55],[25,.64],[30,.72],[35,.78]];
const EARLY35_EV_RELIABILITY_ANCHORS=[[1,.20],[5,.28],[10,.38],[15,.48],[20,.58],[25,.66],[30,.73],[35,.79]];
const clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));
const bp=seq=>seq.filter(x=>x==="B"||x==="P");
const directionalRoundCount=seq=>bp(seq).length;
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));

let physicsBundle=null,final56Bundle=null;
let status={physics:false,final56:false,errors:[]};

function transitionSequence(seq){const a=bp(seq),out=[];for(let i=1;i<a.length;i++)out.push(a[i]===a[i-1]?"S":"X");return out;}
function sxMarkovPSame(seq,window=24,prior=1){
  const t=transitionSequence(seq);if(!t.length)return .5;
  const current=t.at(-1),start=Math.max(0,t.length-1-Math.max(2,window));let same=0,sw=0;
  for(let i=start;i<t.length-1;i++){if(t[i]!==current)continue;if(t[i+1]==="S")same++;else if(t[i+1]==="X")sw++;}
  return clip((same+prior)/(same+sw+2*prior));
}
function currentStage(seq){const a=bp(seq);if(!a.length)return 0;const side=a.at(-1);let n=1;for(let i=a.length-2;i>=0&&a[i]===side;i--)n++;return n;}
function currentDepth(seq){const t=transitionSequence(seq);if(!t.length)return 0;const token=t.at(-1);let n=1;for(let i=t.length-2;i>=0&&t[i]===token;i--)n++;return n;}
function getEstimatedTotalHands(){
  try{const c=+(window.__BGS_FINAL_CONFIG__?.estimatedTotalHands||0);if(c>=ESTIMATED_TOTAL_HANDS_MIN&&c<=ESTIMATED_TOTAL_HANDS_MAX)return c;const s=+localStorage.getItem(CUT_KEY);if(s>=ESTIMATED_TOTAL_HANDS_MIN&&s<=ESTIMATED_TOTAL_HANDS_MAX)return s;}catch(_){}
  return 60;
}
function setEstimatedTotalHands(v){const n=Math.round(+v||0);if(n<ESTIMATED_TOTAL_HANDS_MIN||n>ESTIMATED_TOTAL_HANDS_MAX)throw new Error("estimatedTotalHands must be 50..70");try{localStorage.setItem(CUT_KEY,String(n));}catch(_){}return n;}
function buildOriginal7(seq,corePrediction){
  const corePB=clip(+corePrediction?.probabilities?.B||.5),roundIndex=Math.max(1,Math.min(70,seq.length+1));
  const estimatedTotalHands=getEstimatedTotalHands(),remainingRatio=clip((estimatedTotalHands-(roundIndex-1))/Math.max(1,estimatedTotalHands));
  const signal=corePrediction?.singleHazard||null;
  return {core_p_b:corePB,round_index:roundIndex,estimated_total_hands:estimatedTotalHands,remaining_ratio:remainingRatio,
    sx_markov_p_same:sxMarkovPSame(seq),stage:Number.isFinite(+signal?.state?.length)?+signal.state.length:currentStage(seq),
    depth:Number.isFinite(+signal?.depth?.depth)?+signal.depth.depth:currentDepth(seq)};
}
function original7Vector(f){return ORIGINAL7_NAMES.map(n=>Number.isFinite(+f[n])?+f[n]:0);}

function entropy3(p){let s=0;for(const v of p)if(v>0)s-=v*Math.log(v);return s/Math.log(3);}
function ratios(seq,n){const b=n?seq.slice(-n):seq;if(!b.length)return[0,0,0];return[b.filter(x=>x==="B").length/b.length,b.filter(x=>x==="P").length/b.length,b.filter(x=>x==="T").length/b.length];}
function historyVector(seq){
  const one=Array(HISTORY_WINDOW*3).fill(0),tail=seq.slice(-HISTORY_WINDOW),offset=HISTORY_WINDOW-tail.length,map={B:0,P:1,T:2};
  tail.forEach((token,j)=>{one[(offset+j)*3+map[token]]=1;});
  const bps=bp(seq);let turns=0;for(let i=1;i<bps.length;i++)if(bps[i]!==bps[i-1])turns++;
  const r8=ratios(seq,8),r16=ratios(seq,16),r32=ratios(seq,32),rall=ratios(seq,Math.max(1,seq.length));
  const summary=[Math.min(seq.length,90)/90,Math.min(bps.length,90)/90,Math.min(currentStage(seq),12)/12,turns/Math.max(1,bps.length-1),
    ...r8,...r16,...r32,...rall,entropy3(r8),entropy3(r16),entropy3(r32),bps.length&&bps.at(-1)==="B"?1:0,bps.length&&bps.at(-1)==="P"?1:0];
  const out=[...one,...summary];if(out.length!==HISTORY_INPUT_DIM)throw new Error("history vector "+out.length);return out;
}
function dense(input,w,b,relu){
  const out=Array(b.length).fill(0);
  for(let j=0;j<b.length;j++){let s=+b[j]||0;for(let i=0;i<input.length;i++)s+=(+input[i]||0)*(+w[i][j]||0);out[j]=relu?Math.max(0,s):s;}
  return out;
}
function normalise(block,fallback){const a=block.map(v=>Math.max(0,Number.isFinite(+v)?+v:0)),s=a.reduce((x,y)=>x+y,0);return s>1e-12?a.map(v=>v/s):fallback.slice();}
function temperatureNorm(block,fallback,t=1){const p=normalise(block,fallback).map(v=>Math.max(1e-8,v)),z=p.map(v=>Math.log(v)/Math.max(.25,+t||1)),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function physicsBundleUsesPhysicalEv(){return Array.isArray(physicsBundle?.feature_names)&&physicsBundle.feature_names.includes("physical_ev_banker");}
function finalModelUsesPhysicalEv(){return (+final56Bundle?.training?.particle_physics_version||0)>=2;}
function adaptiveStageEnabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=2;}
function stageProgressV3Enabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=3;}
function stageProgressV4Enabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=4;}
function early35Version(){return Math.max(0,Math.floor(+final56Bundle?.training?.early35_version||+final56Bundle?.early35_version||0));}
function early35Enabled(){return early35Version()>=EARLY35_VERSION;}
function shoeErrorCorrectionVersion(){return Math.max(0,Math.floor(+final56Bundle?.training?.shoe_error_correction_version||+final56Bundle?.shoe_error_correction_version||0));}
function shoeErrorCorrectionEnabled(){return shoeErrorCorrectionVersion()>=SHOE_ERROR_CORRECTION_VERSION;}
function modelSemanticsCompatible(){return finalModelUsesPhysicalEv()===physicsBundleUsesPhysicalEv();}
function sanitizePhysics(raw,temperatures={},physicalEvSemantics=physicsBundleUsesPhysicalEv()){
  if(raw.length!==PHYSICS_DIM)throw new Error("physics dim mismatch");
  const out=Array(PHYSICS_DIM).fill(0);let src=0,dst=0;
  const copyNorm=(n,fb,key)=>{const a=temperatureNorm(raw.slice(src,src+n),fb,temperatures[key]);src+=n;for(const v of a)out[dst++]=v;};
  copyNorm(3,[.58,.34,.08],"card_count");copyNorm(10,Array(10).fill(.1),"player_points");copyNorm(10,Array(10).fill(.1),"banker_points");copyNorm(3,[.4586,.4462,.0952],"winner");
  for(let i=0;i<13;i++)out[dst++]=clip(raw[src++],0,6);
  if(physicalEvSemantics){
    out[dst++]=clip(raw[src++],-1,.95);out[dst++]=clip(raw[src++],-1,1);out[dst++]=clip(raw[src++],-2,2);out[dst++]=clip(raw[src++],0,1);
  }else copyNorm(4,Array(4).fill(.25),"suit");
  out[dst++]=clip(raw[src++],0,416);out[dst++]=clip(raw[src++],0,1);out[dst++]=clip(raw[src++],0,1);
  out[dst++]=clip(raw[src++],-1,1);out[dst++]=clip(raw[src++],0,1);
  return out;
}
const PARTICLE_COUNT=64,PARTICLE_LIKELIHOOD_DRAWS=4,PARTICLE_FALLBACK_DRAWS=12,PARTICLE_FORECAST_DRAWS=2;
let lastParticleDiagnostics=null;
function particleRng(seed=20261005){let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
function particleValue(rank){const r=rank+1;return r<=9?r:0;}
function particleBankerDraws(total,third){if(third===null)return total<=5;if(total<=2)return true;if(total===3)return third!==8;if(total===4)return third>=2&&third<=7;if(total===5)return third>=4&&third<=7;if(total===6)return third>=6&&third<=7;return false;}
function particleDraw(counts,rng){let total=counts.reduce((a,b)=>a+b,0);if(total<=0)throw new Error("empty particle shoe");let pick=Math.floor(rng()*total),sum=0;for(let i=0;i<counts.length;i++){sum+=counts[i];if(pick<sum){counts[i]--;return i;}}throw new Error("particle draw overflow");}
function particleDeal(base,rng){
  const counts=base.slice();if(counts.reduce((a,b)=>a+b,0)<6)throw new Error("particle shoe exhausted");
  const p1=particleDraw(counts,rng),b1=particleDraw(counts,rng),p2=particleDraw(counts,rng),b2=particleDraw(counts,rng);
  const ranks=[p1,b1,p2,b2];let pt=(particleValue(p1)+particleValue(p2))%10,bt=(particleValue(b1)+particleValue(b2))%10;
  const playerInitial=pt,bankerInitial=bt,natural=[8,9].includes(pt)||[8,9].includes(bt);let playerThird=null,bankerThird=null;
  if(!natural){if(pt<=5){const p3=particleDraw(counts,rng);ranks.push(p3);playerThird=particleValue(p3);pt=(pt+playerThird)%10;}if(particleBankerDraws(bt,playerThird)){const b3=particleDraw(counts,rng);ranks.push(b3);bankerThird=particleValue(b3);bt=(bt+bankerThird)%10;}}
  return {outcome:bt>pt?"B":pt>bt?"P":"T",playerPoint:pt,bankerPoint:bt,ranks,counts,cardCount:ranks.length,playerInitial,bankerInitial,natural,playerThird,bankerThird};
}
function particleResample(particles,consumed,weights,rng){
  const n=particles.length,cdf=[];let sum=0;for(const w of weights){sum+=w;cdf.push(sum);}
  const start=rng()/n,next=[],nextConsumed=[];let j=0;
  for(let i=0;i<n;i++){const pos=start+i/n;while(j<n-1&&pos>cdf[j])j++;next.push(particles[j].slice());nextConsumed.push(consumed[j]);}
  return {particles:next,consumed:nextConsumed};
}
function handStateAccumulator(){return {weight:0,natural:0,playerDraw:0,bankerDraw:0,cards:Array(3).fill(0),playerPoints:Array(10).fill(0),bankerPoints:Array(10).fill(0),winner:Array(3).fill(0),cardsConsumed:0};}
function addHandState(acc,hand,weight){
  const w=Math.max(0,+weight||0),outcomeIndex={B:0,P:1,T:2}[hand?.outcome];if(!(w>0)||outcomeIndex===undefined)return;
  const cardIndex=(+hand.cardCount||4)-4;if(cardIndex<0||cardIndex>2)return;
  acc.weight+=w;acc.natural+=w*(hand.natural?1:0);acc.playerDraw+=w*(hand.playerThird===null?0:1);acc.bankerDraw+=w*(hand.bankerThird===null?0:1);acc.cards[cardIndex]+=w;
  acc.playerPoints[(+hand.playerPoint||0)%10]+=w;acc.bankerPoints[(+hand.bankerPoint||0)%10]+=w;acc.winner[outcomeIndex]+=w;acc.cardsConsumed+=w*(+hand.cardCount||4);
}
function finishHandState(acc){
  const total=+acc?.weight||0;if(!(total>1e-12))return {available:false,cards_distribution:[0,0,0],player_final_point_distribution:Array(10).fill(0),banker_final_point_distribution:Array(10).fill(0),winner_distribution:[0,0,0],expected_cards_consumed:0};
  return {available:true,cards_distribution:normalise(acc.cards,[1/3,1/3,1/3]),player_final_point_distribution:normalise(acc.playerPoints,Array(10).fill(.1)),banker_final_point_distribution:normalise(acc.bankerPoints,Array(10).fill(.1)),winner_distribution:normalise(acc.winner,[1/3,1/3,1/3]),expected_cards_consumed:acc.cardsConsumed/total,natural_probability:clip(acc.natural/total),player_draw_probability:clip(acc.playerDraw/total),banker_draw_probability:clip(acc.bankerDraw/total)};
}
function distributionShift(a,b){const left=normalise(Array.isArray(a)?a:[],Array.isArray(a)&&a.length?Array(a.length).fill(1/a.length):[]),right=normalise(Array.isArray(b)?b:[],Array.isArray(b)&&b.length?Array(b.length).fill(1/b.length):[]);if(left.length!==right.length||!left.length)return 1;return clip(.5*left.reduce((sum,value,index)=>sum+Math.abs(value-right[index]),0));}
function snapshotFromHandState(state,{particleUncertainty=0,recentEssRatio=1,physicalEv=null}={}){
  const winner=normalise(state?.winner_distribution||[],[1/3,1/3,1/3]),rawEv=physicalEv||{banker:winner[0]*.95-winner[1],player:winner[1]-winner[0]};const gap=Number.isFinite(+rawEv.gap)?+rawEv.gap:(+rawEv.banker||0)-(+rawEv.player||0);
  return {available:!!state?.available,winner_probabilities:{B:winner[0],P:winner[1],T:winner[2]},winner_distribution:winner,cards_distribution:(state?.cards_distribution||[0,0,0]).slice(),player_final_point_distribution:(state?.player_final_point_distribution||Array(10).fill(0)).slice(),banker_final_point_distribution:(state?.banker_final_point_distribution||Array(10).fill(0)).slice(),physical_ev_banker:+rawEv.banker||0,physical_ev_player:+rawEv.player||0,physical_ev_gap:gap,particle_uncertainty:clip(particleUncertainty),recent_ess_ratio:clip(recentEssRatio),expected_consumed_cards:+state?.expected_cards_consumed||0};
}
function emptyShoeErrorCorrection(){return {shoe_error_correction_version:SHOE_ERROR_CORRECTION_VERSION,shoe_error_correction_policy:"posterior_health_only_no_observation_reweight",available:false,shoe_posterior_health:1,draw_state_health:1,physical_ev_health:1,posterior_reliability_multiplier:1,draw_reliability_multiplier:1,physical_ev_reliability_multiplier:1,recent_miscalibration_score:0,particle_observation_reused:true,particle_posterior_reweighted:false};}
function updateHealth(health,badEvidence){const current=clip(health),bad=clip(badEvidence);return bad>.04?clip(.80*current+.20*(1-.75*bad)):clip(current+.06*(1-current));}
function updateShoeErrorCorrection(state,memory,prior,posterior,actual,{recentEssRatio=1,particleUncertainty=1}={}){
  const expected=normalise(prior?.winner_distribution||[],[1/3,1/3,1/3]),index={B:0,P:1,T:2}[actual]??2,actualProbability=clip(expected[index],.01,.99),predictionSurprise=-Math.log(actualProbability),oneHot=[0,0,0];oneHot[index]=1;
  const brierResidual=expected.reduce((sum,value,i)=>sum+(value-oneHot[i])**2,0)/3,drawCountShift=distributionShift(prior?.cards_distribution,posterior?.cards_distribution),pointDistributionShift=.5*(distributionShift(prior?.player_final_point_distribution,posterior?.player_final_point_distribution)+distributionShift(prior?.banker_final_point_distribution,posterior?.banker_final_point_distribution)),consumptionShift=clip(Math.abs((+prior?.expected_consumed_cards||0)-(+posterior?.expected_cards_consumed||0))/2),ess=clip(recentEssRatio),uncertainty=clip(particleUncertainty),surpriseSeverity=clip((predictionSurprise-.65)/1.65),history=(memory||[]).slice(-SHOE_ERROR_CORRECTION_MEMORY),recent=history.length?history.reduce((sum,item)=>sum+(+item.surprise_severity||0),0)/history.length:0,latentShift=Math.max(drawCountShift,pointDistributionShift,consumptionShift),posteriorProblem=clip(.55*surpriseSeverity+.25*(1-ess)+.20*uncertainty),drawProblem=clip(.50*surpriseSeverity+.35*latentShift+.15*uncertainty),evStrength=clip(Math.abs(+prior?.physical_ev_gap||0)/.10),evOverconfidence=clip(evStrength*(.60*surpriseSeverity+.25*recent+.15*(1-ess)),0,1);
  const next={...(state||emptyShoeErrorCorrection())};next.shoe_posterior_health=updateHealth(next.shoe_posterior_health,posteriorProblem);next.draw_state_health=updateHealth(next.draw_state_health,drawProblem);next.physical_ev_health=updateHealth(next.physical_ev_health,evOverconfidence);Object.assign(next,{shoe_error_correction_version:SHOE_ERROR_CORRECTION_VERSION,shoe_error_correction_policy:"posterior_health_only_no_observation_reweight",available:!!prior?.available&&!!posterior?.available,actual_outcome:actual,actual_probability:actualProbability,prediction_surprise:predictionSurprise,brier_residual:brierResidual,draw_count_shift:drawCountShift,point_distribution_shift:pointDistributionShift,consumption_shift:consumptionShift,recent_ess_ratio:ess,particle_uncertainty:uncertainty,recent_miscalibration_score:recent,posterior_problem:posteriorProblem,draw_point_problem:drawProblem,physical_ev_overconfidence:evOverconfidence,posterior_reliability_multiplier:clip(.90+.10*next.shoe_posterior_health,.90,1),draw_reliability_multiplier:clip(.90+.10*next.draw_state_health,.90,1),physical_ev_reliability_multiplier:clip(.80+.20*next.physical_ev_health,.80,1),particle_observation_reused:true,particle_posterior_reweighted:false});
  const nextMemory=[...history,{surprise:predictionSurprise,surprise_severity:surpriseSeverity,brier_residual:brierResidual,draw_count_shift:drawCountShift,point_distribution_shift:pointDistributionShift,consumption_shift:consumptionShift,recent_ess_ratio:ess,particle_uncertainty:uncertainty}].slice(-SHOE_ERROR_CORRECTION_MEMORY);return {state:next,memory:nextMemory};
}
function settledErrorCorrectionAudit(snapshot,actual){if(!snapshot?.available)return null;const expected=normalise(snapshot.winner_distribution||[],[1/3,1/3,1/3]),index={B:0,P:1,T:2}[actual]??2,probability=clip(expected[index],.01,.99),oneHot=[0,0,0];oneHot[index]=1;return {actual_probability:probability,prediction_surprise:-Math.log(probability),brier_residual:expected.reduce((sum,value,i)=>sum+(value-oneHot[i])**2,0)/3,particle_observation_reused:true,particle_posterior_reweighted:false};}
function estimateParticlePhysics(seq){
  const history=seq.filter(x=>x==="B"||x==="P"||x==="T"),rng=particleRng(),n=PARTICLE_COUNT;
  const correctionEnabled=shoeErrorCorrectionEnabled();let particles=Array.from({length:n},()=>Array(13).fill(32)),consumed=Array(n).fill(0),essHistory=[],correction=emptyShoeErrorCorrection(),correctionMemory=[];
  for(const actual of history){
    const next=[],nextConsumed=Array(n).fill(0),rawWeights=Array(n).fill(0),priorState=correctionEnabled?handStateAccumulator():null,posteriorState=correctionEnabled?handStateAccumulator():null;
    for(let i=0;i<n;i++){
      const base=particles[i],proposals=[],matches=[];
      for(let k=0;k<PARTICLE_LIKELIHOOD_DRAWS;k++){const hand=particleDeal(base,rng);proposals.push(hand);if(correctionEnabled)addHandState(priorState,hand,1/(n*PARTICLE_LIKELIHOOD_DRAWS));if(hand.outcome===actual)matches.push(hand);}
      const initialMatches=matches.length;let likelihood=(initialMatches+.15)/(PARTICLE_LIKELIHOOD_DRAWS+.45);
      if(!matches.length){for(let k=0;k<PARTICLE_FALLBACK_DRAWS;k++){const hand=particleDeal(base,rng);if(hand.outcome===actual){matches.push(hand);break;}}}
      let chosen;if(matches.length){chosen=matches[Math.floor(rng()*matches.length)];if(initialMatches===0)likelihood=Math.max(likelihood,.02);if(correctionEnabled)for(const matching of matches)addHandState(posteriorState,matching,likelihood/(n*matches.length));}else{chosen=proposals[Math.floor(rng()*proposals.length)];likelihood=1e-4;}
      next.push(chosen.counts);nextConsumed[i]=consumed[i]+chosen.cardCount;rawWeights[i]=likelihood;
    }
    const total=rawWeights.reduce((a,b)=>a+b,0),weights=total>1e-12?rawWeights.map(w=>w/total):Array(n).fill(1/n);
    const ess=1/Math.max(1e-12,weights.reduce((a,w)=>a+w*w,0)),essRatio=clip(ess/n);essHistory.push(essRatio);
    if(correctionEnabled){const prior=snapshotFromHandState(finishHandState(priorState),{particleUncertainty:clip(1-essRatio),recentEssRatio:essRatio}),posterior=finishHandState(posteriorState),update=updateShoeErrorCorrection(correction,correctionMemory,prior,posterior,actual,{recentEssRatio:essRatio,particleUncertainty:prior.particle_uncertainty});correction=update.state;correctionMemory=update.memory;}
    const resampled=particleResample(next,nextConsumed,weights,rng);particles=resampled.particles;consumed=resampled.consumed;
  }
  const out=Array(PHYSICS_DIM).fill(0),forecastRng=particleRng(20261005+104729);let samples=0;
  for(const counts of particles)for(let k=0;k<PARTICLE_FORECAST_DRAWS;k++){
    const hand=particleDeal(counts,forecastRng);samples++;out[{4:0,5:1,6:2}[hand.cardCount]]++;out[3+hand.playerPoint]++;out[13+hand.bankerPoint]++;out[23+({B:0,P:1,T:2}[hand.outcome])]++;
    for(const rank of hand.ranks)out[26+rank]++;const diff=hand.bankerPoint-hand.playerPoint;out[46]+=diff/9;out[47]+=Math.abs(diff)/9;
  }
  for(const [a,b] of [[0,3],[3,13],[13,23],[23,26],[26,39]])for(let i=a;i<b;i++)out[i]/=samples;
  const pB=out[23],pP=out[24],physicalEvB=pB*.95-pP,physicalEvP=pP-pB;
  out[39]=physicalEvB;out[40]=physicalEvP;out[41]=physicalEvB-physicalEvP;out[43]=consumed.reduce((a,b)=>a+b,0)/n;
  const means=Array(13).fill(0);for(const counts of particles)for(let j=0;j<13;j++)means[j]+=counts[j]/n;
  const remaining=Math.max(1,means.reduce((a,b)=>a+b,0));out[44]=means.slice(0,5).reduce((a,b)=>a+b,0)/remaining;out[45]=means.slice(8).reduce((a,b)=>a+b,0)/remaining;out[46]/=samples;out[47]/=samples;
  let spread=0;for(let j=0;j<13;j++){let v=0;for(const counts of particles)v+=(counts[j]-means[j])**2/n;spread+=Math.sqrt(v)/32;}spread/=13;
  const recent=essHistory.length?essHistory.slice(-8).reduce((a,b)=>a+b,0)/Math.min(8,essHistory.length):1;
  const consumedMean=out[43],consumedStd=Math.sqrt(consumed.reduce((a,v)=>a+(v-consumedMean)**2,0)/n),uncertainty=clip(.60*Math.min(1,spread*4)+.40*(1-recent));
  out[42]=uncertainty;
  const forecastState={available:true,cards_distribution:out.slice(0,3),player_final_point_distribution:out.slice(3,13),banker_final_point_distribution:out.slice(13,23),winner_distribution:out.slice(23,26),expected_cards_consumed:4*out[0]+5*out[1]+6*out[2]},preHandSnapshot=snapshotFromHandState(forecastState,{particleUncertainty:uncertainty,recentEssRatio:recent,physicalEv:{banker:physicalEvB,player:physicalEvP,gap:physicalEvB-physicalEvP}});
  return {physics:out,diagnostics:{particle_count:n,history_rounds:history.length,expected_consumed_cards:consumedMean,consumed_cards_std:consumedStd,recent_ess_ratio:recent,composition_spread:spread,posterior_uncertainty:uncertainty,physical_ev_banker:physicalEvB,physical_ev_player:physicalEvP,physical_ev_gap:physicalEvB-physicalEvP,pre_hand_snapshot:preHandSnapshot,shoe_error_correction_version:correctionEnabled?SHOE_ERROR_CORRECTION_VERSION:0,shoe_error_correction:correctionEnabled?correction:null,shoe_posterior_health:correctionEnabled?correction.shoe_posterior_health:1,draw_state_health:correctionEnabled?correction.draw_state_health:1,physical_ev_health:correctionEnabled?correction.physical_ev_health:1,posterior_reliability_multiplier:correctionEnabled?correction.posterior_reliability_multiplier:1,draw_reliability_multiplier:correctionEnabled?correction.draw_reliability_multiplier:1,physical_ev_reliability_multiplier:correctionEnabled?correction.physical_ev_reliability_multiplier:1,recent_miscalibration_score:correctionEnabled?correction.recent_miscalibration_score:0,error_memory_size:correctionEnabled?correctionMemory.length:0}};
}
function interpAnchors(value,anchors){if(value<=anchors[0][0])return anchors[0][1];for(let i=1;i<anchors.length;i++){if(value<=anchors[i][0]){const [x0,y0]=anchors[i-1],[x1,y1]=anchors[i],t=(value-x0)/Math.max(1e-12,x1-x0);return y0+t*(y1-y0);}}return anchors.at(-1)[1];}
function effectiveParticleProgress(rounds,d,estimatedTotalHands=60){
  const uncertainty=clip(+d?.posterior_uncertainty||1);
  if(!stageProgressV4Enabled()){
    const roundProgress=clip(rounds/70),cardProgress=clip((+d?.expected_consumed_cards||0)/ESTIMATED_PLAYABLE_CARDS),roundWeight=.55+.25*uncertainty;
    return clip(roundWeight*roundProgress+(1-roundWeight)*cardProgress);
  }
  const totalHands=clip(estimatedTotalHands,ESTIMATED_TOTAL_HANDS_MIN,ESTIMATED_TOTAL_HANDS_MAX),roundProgress=clip(rounds/totalHands),cardProgress=clip((+d?.expected_consumed_cards||0)/ESTIMATED_PLAYABLE_CARDS),roundWeight=.45+.30*uncertainty;
  return clip(roundWeight*roundProgress+(1-roundWeight)*cardProgress);
}
function baselineParticleFusionWeight(rounds,d,estimatedTotalHands=60){
  if(!adaptiveStageEnabled()){const base=rounds<12?.18:rounds<20?.25:rounds<=40?.34:rounds<=50?.40:.46,reliability=.75+.25*clip(d?.recent_ess_ratio??1);return clip(base*reliability,.10,.46);}
  const progress=effectiveParticleProgress(rounds,d,estimatedTotalHands),base=stageProgressV4Enabled()?interpAnchors(progress,[[0,.18],[.15,.22],[.30,.29],[.45,.36],[.60,.43],[.70,.48],[.80,.52],[.90,.55],[1,.58]]):interpAnchors(70*progress,[[0,.18],[10,.20],[20,.27],[30,.33],[40,.39],[50,.45],[55,.48],[60,.52],[65,.55],[70,.57]]);
  const ess=clip(d?.recent_ess_ratio??1),uncertainty=clip(d?.posterior_uncertainty??1),reliability=(.70+.30*ess)*(1-.20*uncertainty);
  return clip(base*reliability,.10,.58);
}
function particleReliability(d){return clip(d?.recent_ess_ratio??1)*(1-clip(d?.posterior_uncertainty??1));}
function early35EvidenceWeight(rounds,d,estimatedTotalHands=60){
  const stage=interpAnchors(rounds,EARLY35_EVIDENCE_ANCHORS),early=clip(stage*(.65+.35*particleReliability(d)),.10,.80);
  if(rounds<=EARLY35_BRIDGE_START)return early;
  const baseline=baselineParticleFusionWeight(rounds,d,estimatedTotalHands);
  if(rounds>=EARLY35_BRIDGE_END)return baseline;
  const t=clip((rounds-EARLY35_BRIDGE_START)/(EARLY35_BRIDGE_END-EARLY35_BRIDGE_START));return early+(baseline-early)*t;
}
function particleFusionWeight(rounds,d,estimatedTotalHands=60){return baselineParticleFusionWeight(rounds,d,estimatedTotalHands);}
function baselinePhysicalEvReliability(rounds,d,estimatedTotalHands=60){
  if(!adaptiveStageEnabled())return 1;
  const progress=effectiveParticleProgress(rounds,d,estimatedTotalHands),stage=stageProgressV4Enabled()?interpAnchors(progress,[[0,.30],[.15,.36],[.30,.45],[.45,.56],[.60,.68],[.70,.75],[.80,.81],[.90,.85],[1,.88]]):interpAnchors(70*progress,[[0,.32],[10,.36],[20,.45],[30,.53],[40,.61],[50,.70],[55,.75],[60,.80],[65,.84],[70,.87]]);
  const ess=clip(d?.recent_ess_ratio??1),uncertainty=clip(d?.posterior_uncertainty??1);
  return clip(stage*(.72+.28*ess)*(1-.30*uncertainty),.20,.90);
}
function early35PhysicalEvReliability(rounds,d,estimatedTotalHands=60){
  const stage=interpAnchors(rounds,EARLY35_EV_RELIABILITY_ANCHORS),ess=clip(d?.recent_ess_ratio??1),uncertainty=clip(d?.posterior_uncertainty??1),early=clip(stage*(.70+.30*ess)*(1-.35*uncertainty),0,.85);
  if(rounds<=EARLY35_BRIDGE_START)return early;
  const baseline=baselinePhysicalEvReliability(rounds,d,estimatedTotalHands);
  if(rounds>=EARLY35_BRIDGE_END)return baseline;
  const t=clip((rounds-EARLY35_BRIDGE_START)/(EARLY35_BRIDGE_END-EARLY35_BRIDGE_START));return early+(baseline-early)*t;
}
function physicalEvReliability(rounds,d,estimatedTotalHands=60){return early35Enabled()?early35PhysicalEvReliability(rounds,d,estimatedTotalHands):baselinePhysicalEvReliability(rounds,d,estimatedTotalHands);}
function fuseParticlePhysics(mlp,seq){
  const estimate=estimateParticlePhysics(seq),p=estimate.physics.slice(),estimatedTotalHands=getEstimatedTotalHands(),w=particleFusionWeight(seq.length,estimate.diagnostics,estimatedTotalHands),evidenceWeight=early35Enabled()?early35EvidenceWeight(seq.length,estimate.diagnostics,estimatedTotalHands):w,evReliability=physicalEvReliability(seq.length,estimate.diagnostics,estimatedTotalHands),out=mlp.slice(),correctionEnabled=shoeErrorCorrectionEnabled(),correction=correctionEnabled?(estimate.diagnostics.shoe_error_correction||{}):{},posteriorMultiplier=clip(correction.posterior_reliability_multiplier??1,.90,1),drawMultiplier=clip(correction.draw_reliability_multiplier??1,.90,1),physicalEvMultiplier=clip(correction.physical_ev_reliability_multiplier??1,.80,1);
  const rawEv=p.slice(39,42);if(adaptiveStageEnabled()||correctionEnabled)for(let i=39;i<42;i++)p[i]*=(adaptiveStageEnabled()?evReliability:1)*physicalEvMultiplier;
  for(const [a,b] of [[0,3],[3,13],[13,23],[23,26]]){const blockWeight=evidenceWeight*(a===23?posteriorMultiplier:drawMultiplier),mixed=mlp.slice(a,b).map((v,i)=>(1-blockWeight)*v+blockWeight*p[a+i]),norm=normalise(mixed,Array(b-a).fill(1/(b-a)));for(let i=a;i<b;i++)out[i]=norm[i-a];}
  const rankWeight=w*drawMultiplier;for(let i=26;i<39;i++)out[i]=(1-rankWeight)*mlp[i]+rankWeight*p[i];
  for(let i=39;i<43;i++)out[i]=p[i];
  for(let i=43;i<48;i++)out[i]=(1-rankWeight)*mlp[i]+rankWeight*p[i];
  const expected=4*out[0]+5*out[1]+6*out[2],rankTotal=out.slice(26,39).reduce((a,b)=>a+b,0);if(rankTotal>1e-12)for(let i=26;i<39;i++)out[i]*=expected/rankTotal;
  const effectiveProgress=effectiveParticleProgress(seq.length,estimate.diagnostics,estimatedTotalHands);estimate.diagnostics.estimated_total_hands=estimatedTotalHands;estimate.diagnostics.round_progress=clip(seq.length/estimatedTotalHands);estimate.diagnostics.card_progress=clip((+estimate.diagnostics.expected_consumed_cards||0)/ESTIMATED_PLAYABLE_CARDS);estimate.diagnostics.effective_progress=effectiveProgress;estimate.diagnostics.effective_progress_round=70*effectiveProgress;
  estimate.diagnostics.fusion_weight=w;estimate.diagnostics.particle_evidence_weight=evidenceWeight;estimate.diagnostics.early35_version=early35Enabled()?EARLY35_VERSION:0;estimate.diagnostics.physical_ev_reliability=evReliability;estimate.diagnostics.shoe_error_correction_version=correctionEnabled?SHOE_ERROR_CORRECTION_VERSION:0;estimate.diagnostics.posterior_reliability_multiplier=posteriorMultiplier;estimate.diagnostics.draw_reliability_multiplier=drawMultiplier;estimate.diagnostics.physical_ev_reliability_multiplier=physicalEvMultiplier;estimate.diagnostics.effective_particle_evidence_weight=evidenceWeight*posteriorMultiplier;estimate.diagnostics.effective_draw_evidence_weight=evidenceWeight*drawMultiplier;estimate.diagnostics.raw_physical_ev_banker=rawEv[0];estimate.diagnostics.raw_physical_ev_player=rawEv[1];
  estimate.diagnostics.physical_ev_banker=out[39];estimate.diagnostics.physical_ev_player=out[40];estimate.diagnostics.physical_ev_gap=out[41];estimate.diagnostics.expected_next_card_count=expected;lastParticleDiagnostics=estimate.diagnostics;
  return sanitizePhysics(out,{},true);
}

function predictPhysics(seq){
  if(!physicsBundle?.trained)return null;
  let x=historyVector(seq),mean=physicsBundle.scaler?.mean||[],scale=physicsBundle.scaler?.scale||[];
  x=x.map((v,i)=>(v-(+mean[i]||0))/Math.max(1e-12,+scale[i]||1));
  const coefs=physicsBundle.coefs||[],intercepts=physicsBundle.intercepts||[];
  if(!coefs.length||coefs.length!==intercepts.length)throw new Error("invalid physics bundle");
  let h=x;for(let layer=0;layer<coefs.length;layer++)h=dense(h,coefs[layer],intercepts[layer],layer<coefs.length-1);
  const lossScale=physicsBundle.loss_scale||[];h=h.map((v,i)=>(+v||0)/Math.max(1e-12,+lossScale[i]||1));
  const tMean=physicsBundle.target_scaler?.mean||[],tScale=physicsBundle.target_scaler?.scale||[];
  h=h.map((v,i)=>(+v||0)*Math.max(1e-12,+tScale[i]||1)+(+tMean[i]||0));
  const outputSlope=physicsBundle.output_calibration?.slope||[],outputIntercept=physicsBundle.output_calibration?.intercept||[];
  h=h.map((v,i)=>v*(Number.isFinite(+outputSlope[i])?+outputSlope[i]:1)+(Number.isFinite(+outputIntercept[i])?+outputIntercept[i]:0));
  const physicalSemantics=physicsBundleUsesPhysicalEv();
  const mlp=sanitizePhysics(h,physicsBundle.calibration_temperatures||{},physicalSemantics);
  const particleCompatible=finalModelUsesPhysicalEv()&&physicalSemantics;
  if(!particleCompatible){
    lastParticleDiagnostics={enabled:false,reason:"legacy_model_semantics",particle_physics_version:+final56Bundle?.training?.particle_physics_version||0};
    return mlp;
  }
  const fused=fuseParticlePhysics(mlp,seq);
  lastParticleDiagnostics={enabled:true,particle_physics_version:2,...lastParticleDiagnostics};
  return fused;
}

function findChild(node,id){return(node?.children||[]).find(c=>+c.nodeid===+id)||null;}
function splitIndex(split,names){const t=String(split??"");if(/^f\d+$/.test(t))return+t.slice(1);return names.indexOf(t);}
function evaluateTree(tree,vector,names){
  let node=tree,guard=0;while(node&&guard++<256){if(Object.prototype.hasOwnProperty.call(node,"leaf"))return+node.leaf||0;
    const idx=splitIndex(node.split,names),value=idx>=0?Math.fround(vector[idx]):NaN,threshold=Math.fround(+node.split_condition);
    node=findChild(node,!Number.isFinite(value)?node.missing:(value<threshold?node.yes:node.no));}
  return 0;
}
function predictFinalProbability(bundle,vector,names){
  if(!bundle?.trained||!Array.isArray(bundle.trees))return .5;
  let margin=+bundle.base_margin||0;
  for(const tree of bundle.trees)margin+=evaluateTree(tree,vector,names);
  let probability=clip(sigmoid(margin),1e-7,1-1e-7),calibration=bundle.calibration||{};
  if(calibration.method==="platt"){
    const logit=Math.log(probability/(1-probability));
    const slope=Number.isFinite(+calibration.slope)?+calibration.slope:1,intercept=Number.isFinite(+calibration.intercept)?+calibration.intercept:0;
    probability=sigmoid(slope*logit+intercept);
  }else if(calibration.method==="isotonic"&&calibration.x_thresholds?.length>1&&calibration.y_thresholds?.length===calibration.x_thresholds.length){
    const xs=calibration.x_thresholds,ys=calibration.y_thresholds;if(probability<=xs[0])probability=ys[0];else if(probability>=xs.at(-1))probability=ys.at(-1);else{
      let lo=0,hi=xs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(probability<xs[mid])hi=mid;else lo=mid;}
      const ratio=(probability-xs[lo])/Math.max(1e-12,xs[hi]-xs[lo]);probability=ys[lo]+ratio*(ys[hi]-ys[lo]);
    }
  }
  return clip(probability);
}
function normalisedEntropy(block){const p=normalise(block,Array(block.length).fill(1/block.length));return-p.reduce((s,v)=>s+(v>0?v*Math.log(v):0),0)/Math.log(block.length);}
function calibrateNoise(value,calibration){
  const v=clip(value),xs=calibration?.x_thresholds||[],ys=calibration?.y_thresholds||[];
  if(calibration?.method!=="isotonic"||xs.length<2||ys.length!==xs.length)return v;
  if(v<=xs[0])return clip(ys[0]);if(v>=xs.at(-1))return clip(ys.at(-1));
  let lo=0,hi=xs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(v<xs[mid])hi=mid;else lo=mid;}
  const t=(v-xs[lo])/Math.max(1e-12,xs[hi]-xs[lo]);return clip(ys[lo]+t*(ys[hi]-ys[lo]));
}
function physicsNoiseScore(physics,roundIndex=70,effectiveProgress=null){
  const version=final56Bundle?.training?.noise_score_version||1;
  if(version<2){const r=physicsIntegrity(physics);return r?Math.abs(r.cardCountProbabilitySum-1):1;}
  const winner=physics.slice(23,26).sort((a,b)=>b-a),gap=(winner[0]||0)-(winner[1]||0);
  if(version<3)return clip(.20*normalisedEntropy(physics.slice(0,3))+.15*normalisedEntropy(physics.slice(3,13))+.15*normalisedEntropy(physics.slice(13,23))+.35*normalisedEntropy(physics.slice(23,26))+.15*(1-gap));
  const densityGap=Math.abs(clip(physics[44])-clip(physics[45])),densityAmbiguity=1-Math.min(1,densityGap/.25);
  const uncertaintyTerm=version>=5?clip(physics[42]):normalisedEntropy(physics.slice(39,43));
  const raw=clip(.10*normalisedEntropy(physics.slice(0,3))+.075*normalisedEntropy(physics.slice(3,13))+.075*normalisedEntropy(physics.slice(13,23))+.15*normalisedEntropy(physics.slice(23,26))+.25*normalisedEntropy(physics.slice(26,39))+.15*uncertaintyTerm+.075*(1-gap)+.125*densityAmbiguity);
  const compressed=.50+.35*Math.tanh((raw-.75)/.20),uncertainty=clip(physics[42]),baseProxy=version>=5?clip(Math.max(.70*compressed+.30*uncertainty,.50+.40*uncertainty*uncertainty)):compressed;
  const influence=stageProgressV4Enabled()&&Number.isFinite(+effectiveProgress)?interpAnchors(clip(+effectiveProgress),[[0,.35],[.35,.35],[.70,.65],[1,1]]):adaptiveStageEnabled()?interpAnchors(roundIndex,[[1,.35],[40,.35],[50,.65],[70,1]]):(roundIndex<=40?.35:roundIndex<=50?.65:1);
  const proxy=clip(.50+(baseProxy-.50)*influence);
  if(version<4)return proxy;
  const calibration=final56Bundle?.training?.physics_noise_calibration||physicsBundle?.uncertainty_calibration||{};
  return calibrateNoise(proxy,calibration);
}
function buildExtended(corePB,o7,physics){
  const original=original7Vector(o7),roundProgress=clip(Number(original[1])/70),uncertainty=clip(physics[42]),cardProgress=clip((+physics[43]||0)/ESTIMATED_PLAYABLE_CARDS);
  const v4Diagnostics=lastParticleDiagnostics?.enabled===true?lastParticleDiagnostics:null;
  const effectiveProgress=stageProgressV4Enabled()&&v4Diagnostics?effectiveParticleProgress(original[1],v4Diagnostics,original[2]):adaptiveStageEnabled()?clip((.55+.25*uncertainty)*roundProgress+(.45-.25*uncertainty)*cardProgress):roundProgress;
  const progress=effectiveProgress**3,noise=physicsNoiseScore(physics,original[1],effectiveProgress);
  const out=[corePB,progress,...original.slice(1),...physics,noise];if(out.length!==FEATURE_DIM)throw new Error("extended dim "+out.length);return out;
}
function unpackPhysicsForecast(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)throw new Error("physics forecast dim mismatch");
  const physicalSemantics=physicsBundleUsesPhysicalEv(),index=physicalSemantics?PHYSICS_INDEX:LEGACY_PHYSICS_INDEX;
  const value=name=>{const v=+physics[index[name]];return Number.isFinite(v)?v:0;};
  const cardCountProbabilities={"4_cards":value("cards_p4"),"5_cards":value("cards_p5"),"6_cards":value("cards_p6")};
  const expectedNextCardCount=4*cardCountProbabilities["4_cards"]+5*cardCountProbabilities["5_cards"]+6*cardCountProbabilities["6_cards"];
  const rankExpectedConsumption=Object.fromEntries("A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(rank=>[rank,value("next_rank_expected_"+rank)]));
  if(!physicalSemantics){
    const suitConsumptionRatios=Object.fromEntries(["spades","hearts","diamonds","clubs"].map(suit=>[suit,value("next_suit_ratio_"+suit)]));
    return {nextCardCountProbabilities:cardCountProbabilities,expectedNextCardCount,rankExpectedConsumption,suitConsumptionRatios,physicalEv:null};
  }
  const physicalEv={banker:value("physical_ev_banker"),player:value("physical_ev_player"),gap:value("physical_ev_gap"),uncertainty:value("particle_uncertainty")};
  return {nextCardCountProbabilities:cardCountProbabilities,expectedNextCardCount,rankExpectedConsumption,physicalEv};
}
function physicsIntegrity(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)return null;
  const forecast=unpackPhysicsForecast(physics),sum=values=>values.reduce((total,value)=>total+(+value||0),0);
  const cardCountProbabilitySum=sum(Object.values(forecast.nextCardCountProbabilities));
  const rankExpectedConsumptionTotal=sum(Object.values(forecast.rankExpectedConsumption));
  const rankExpectedTotalGap=Math.abs(rankExpectedConsumptionTotal-forecast.expectedNextCardCount);
  const rankExpectedTotalTolerance=Math.max(.50,forecast.expectedNextCardCount*.10);
  const physicalOk=!forecast.physicalEv||(forecast.physicalEv.banker>=-1&&forecast.physicalEv.banker<=.95&&forecast.physicalEv.player>=-1&&forecast.physicalEv.player<=1&&forecast.physicalEv.gap>=-2&&forecast.physicalEv.gap<=2&&forecast.physicalEv.uncertainty>=0&&forecast.physicalEv.uncertainty<=1);
  const checks={cardCountDistribution:Math.abs(cardCountProbabilitySum-1)<=1e-4,rankConsumptionTotal:rankExpectedTotalGap<=rankExpectedTotalTolerance,physicalEvRanges:physicalOk};
  return {valid:Object.values(checks).every(Boolean),checks,cardCountProbabilitySum,expectedNextCardCount:forecast.expectedNextCardCount,rankExpectedConsumptionTotal,rankExpectedTotalGap,rankExpectedTotalTolerance,physicalEv:forecast.physicalEv||null};
}
function dataQuality(seq){const directionalRounds=directionalRoundCount(seq);return {directionalRounds,stage:directionalRounds<12?"cold":directionalRounds<20?"warm":"ready",entryEligible:directionalRounds>=12,preferredEntry:directionalRounds>=20};}
function effectiveProgressRound(roundIndex,progressWeight){
  if(!stageProgressV3Enabled()||!Number.isFinite(+progressWeight))return clip(+roundIndex||1,1,70);
  return 70*Math.cbrt(clip(+progressWeight||0));
}
function applyProbabilityBounds(rawPB,roundIndex,noiseScore,progressWeight=null){
  if(!stageProgressV3Enabled()){
    const pair=roundIndex<=40?EARLY_BOUNDS:(roundIndex>50&&noiseScore<=PHYSICS_NOISE_LOW_THRESHOLD?LATE_CLEAN_BOUNDS:DEFAULT_BOUNDS);
    return {value:clip(rawPB,pair[0],pair[1]),low:pair[0],high:pair[1]};
  }
  if(stageProgressV4Enabled()){
    const progress=Math.cbrt(clip(+progressWeight||0)),middle=clip((progress-.35)/.35),late=clip((progress-.70)/.30),clean=clip((PHYSICS_NOISE_LOW_THRESHOLD-clip(noiseScore))/PHYSICS_NOISE_LOW_THRESHOLD);
    const low=EARLY_BOUNDS[0]-.05*middle-.05*late*clean,high=EARLY_BOUNDS[1]+.05*middle+.05*late*clean;
    return {value:clip(rawPB,low,high),low,high};
  }
  const progressRound=effectiveProgressRound(roundIndex,progressWeight);
  let low=interpAnchors(progressRound,[[1,EARLY_BOUNDS[0]],[40,EARLY_BOUNDS[0]],[50,DEFAULT_BOUNDS[0]]]);
  let high=interpAnchors(progressRound,[[1,EARLY_BOUNDS[1]],[40,EARLY_BOUNDS[1]],[50,DEFAULT_BOUNDS[1]]]);
  if(progressRound>50&&noiseScore<=PHYSICS_NOISE_LOW_THRESHOLD){
    low=interpAnchors(progressRound,[[50,DEFAULT_BOUNDS[0]],[70,LATE_CLEAN_BOUNDS[0]]]);
    high=interpAnchors(progressRound,[[50,DEFAULT_BOUNDS[1]],[70,LATE_CLEAN_BOUNDS[1]]]);
  }
  return {value:clip(rawPB,low,high),low,high};
}
function dynamicEmaAlpha(roundIndex,noiseScore,config,progressWeight=null){
  if(!adaptiveStageEnabled()){
    const stage=roundIndex<=40?"early":roundIndex>50?"late":"middle",range=stage==="early"?[.35,.45]:stage==="middle"?[.50,.60]:[.65,.75];
    const base=Number.isFinite(+config[stage+"_alpha"])?+config[stage+"_alpha"]:(range[0]+range[1])/2,gain=Math.max(0,Number.isFinite(+config.noise_gain)?+config.noise_gain:.05);
    return clip(base+gain*(.5-clip(+noiseScore||0)),range[0],range[1]);
  }
  if(stageProgressV4Enabled()){
    const progress=Math.cbrt(clip(+progressWeight||0)),base=interpAnchors(progress,[[0,.38],[.20,.41],[.35,.46],[.50,.53],[.65,.59],[.75,.63],[.85,.67],[.95,.70],[1,.72]]),noise=clip(noiseScore),adjustment=.03*Math.max(0,.5-noise)/.5-.05*Math.max(0,noise-.5)/.5;
    return clip(base+adjustment,.35,.75);
  }
  const early=Number.isFinite(+config.early_alpha)?+config.early_alpha:.40,middle=Number.isFinite(+config.middle_alpha)?+config.middle_alpha:.55,late=Number.isFinite(+config.late_alpha)?+config.late_alpha:.70;
  const progressRound=stageProgressV3Enabled()?effectiveProgressRound(roundIndex,progressWeight):roundIndex;
  const base=interpAnchors(progressRound,[[1,early],[20,early],[40,(early+middle)/2],[50,middle],[60,(middle+late)/2],[70,late]]);
  const gain=Math.max(0,Number.isFinite(+config.noise_gain)?+config.noise_gain:.05);
  return clip(base+gain*(.5-clip(+noiseScore||0)),.35,.75);
}
function applyDynamicSmoothing(clippedPB,roundIndex,noiseScore,bounds,progressWeight=null){
  const config=final56Bundle?.smoothing||{},dynamic=config.method==="dynamic_post_clip_ema"&&config.enabled===true;
  const legacy=config.method==="causal_ema"&&(+config.strength||0)>0;
  if(!dynamic&&!legacy)return {value:clippedPB,alpha:1,strength:0,profile:"off",applied:false};
  const shoeId=getShoeId(),rows=readRows();let previous=null;
  for(let i=rows.length-1;i>=0;i--){const row=rows[i];if(row?.shoe_id!==shoeId||+row.round_index>=roundIndex)continue;
    const value=Number.isFinite(+row.final_p_b)?+row.final_p_b:+row.smoothed_p_b;if(Number.isFinite(value)){previous=value;break;}}
  if(previous===null)return {value:clippedPB,alpha:1,strength:0,profile:config.profile||"first_round",applied:false};
  const alpha=dynamic?dynamicEmaAlpha(roundIndex,noiseScore,config,progressWeight):1-clip(+config.strength||0,0,.15);
  const value=clip(alpha*clippedPB+(1-alpha)*previous,bounds.low,bounds.high);
  return {value,alpha,strength:1-alpha,profile:config.profile||(legacy?"legacy":"custom"),applied:true};
}
function decisionPolicy(roundIndex,noiseScore){
  const threshold=final56Bundle?.ev_thresholds||{},pick=(key,fallback)=>Number.isFinite(+threshold[key])?+threshold[key]:fallback;
  const base=roundIndex<=40?pick("early",.020):roundIndex>50?pick("late",.005):pick("middle",.010);
  const config=final56Bundle?.decision_policy||{};
  const profile=String(config.profile||final56Bundle?.decision_policy_profile||(config.enabled===true?"custom":"hard_ev"));
  if(config.enabled!==true)return {enabled:false,profile,minEv:base,activationEv:base,softBand:0,minConfidence:0,confidenceBand:0,strongMargin:0,bandMinimum:0,volumeGuard:{}};
  const value=(key,fallback)=>Number.isFinite(+config[key])?+config[key]:fallback;
  const noiseThreshold=value("noise_threshold",.78),noise=clip(+noiseScore||0),lowNoise=noise<=noiseThreshold;
  const middle=roundIndex>40&&roundIndex<=50,late=roundIndex>50;
  const relief=lowNoise?(middle?value("middle_relief",.001):late?value("late_relief",.001):0):0;
  const maxPenalty=Math.max(0,value("max_noise_ev_penalty",.001));
  const penalty=Math.min(maxPenalty,Math.max(0,noise-noiseThreshold)*(maxPenalty/Math.max(1e-9,1-noiseThreshold)));
  const minEv=Math.max(0,base-relief+penalty);
  const softBand=lowNoise?(middle?value("middle_soft_band",.001):late?value("late_soft_band",.0015):0):0;
  const bandConfig=config.confidence_band&&typeof config.confidence_band==="object"?config.confidence_band:{};
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late";
  const bandBase=Math.max(0,Number.isFinite(+bandConfig[stage])?+bandConfig[stage]:0);
  const bandReference=clip(Number.isFinite(+bandConfig.noise_reference)?+bandConfig.noise_reference:.50);
  const bandGain=Math.max(0,Number.isFinite(+bandConfig.noise_gain)?+bandConfig.noise_gain:0);
  const cleanLate=stage==="late"&&noise<=PHYSICS_NOISE_LOW_THRESHOLD;
  const cleanLateRelief=cleanLate?Math.max(0,Number.isFinite(+bandConfig.clean_late_relief)?+bandConfig.clean_late_relief:0):0;
  const bandMinimum=Math.max(0,Number.isFinite(+bandConfig.minimum)?+bandConfig.minimum:0);
  const bandMaximum=Math.max(bandMinimum,Number.isFinite(+bandConfig.maximum)?+bandConfig.maximum:.5);
  const confidenceBand=clip(bandBase+bandGain*Math.max(0,noise-bandReference)-cleanLateRelief,bandMinimum,bandMaximum);
  const strong=bandConfig.strong_margin&&typeof bandConfig.strong_margin==="object"?bandConfig.strong_margin:{};
  const strongMargin=Math.max(0,Number.isFinite(+strong[stage])?+strong[stage]:0);
  return {enabled:true,profile,minEv,activationEv:Math.max(0,minEv-softBand),softBand,minConfidence:Math.max(0,value("min_confidence",.0005)),confidenceBand,strongMargin,bandMinimum,volumeGuard:config.volume_guard||{}};
}
function softConfidence(edge,policy){
  if(edge<=policy.activationEv)return 0;
  const premium=Math.max(0,edge-policy.minEv);
  return policy.softBand<=0?premium:Math.max(policy.minConfidence,premium,.5*Math.min(policy.softBand,edge-policy.activationEv));
}
function volumeGuardState(roundIndex,probabilityB,policy){
  const config=policy?.volumeGuard||{};
  if(policy?.enabled!==true||config.enabled!==true)return {active:false,bandRelief:0,evRelief:0};
  const window=Math.max(1,Math.floor(+config.window||16)),minimum=Math.max(0,Math.floor(+config.min_history||8));
  const shoeId=getShoeId(),rows=readRows().filter(row=>row?.shoe_id===shoeId&&+row.round_index<roundIndex&&Number.isFinite(+row.final_p_b)).slice(-window);
  if(rows.length<minimum)return {active:false,bandRelief:0,evRelief:0};
  const p=clip(probabilityB),expectedProbability=value=>Math.max(clip(value),1-clip(value));
  let actions=0,expected=0,baseline=0;
  for(const row of rows){
    const priorP=clip(+row.final_p_b),priorPlayer=1-priorP;
    const priorActivation=Math.max(0,Number.isFinite(+row.activation_ev)?+row.activation_ev:policy.activationEv);
    const baseAction=(priorP*.95-priorPlayer)>priorActivation||(priorPlayer-priorP)>priorActivation;
    const action=String(row.predicted_direction||"").includes("莊 B")||String(row.predicted_direction||"").includes("閒 P");
    if(action){actions++;expected+=expectedProbability(priorP);}
    if(baseAction)baseline+=expectedProbability(priorP);
  }
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late",rates=config.target_action_rate&&typeof config.target_action_rate==="object"?config.target_action_rate:{};
  const target=Math.max(0,Number.isFinite(+rates[stage])?+rates[stage]:0),floor=clip(Number.isFinite(+config.expected_correct_floor)?+config.expected_correct_floor:.95);
  const active=actions/rows.length<target||(baseline>0&&expected+1e-12<floor*baseline);
  return {active,bandRelief:active?Math.max(0,+config.band_relief||0):0,evRelief:active?Math.max(0,+config.ev_relief||0):0};
}

function applyFinalPrediction(seq,corePrediction=null){
  // Production order is intentional:
  // Physics/Particle -> Physical EV -> Frozen Core -> Final XGB -> Final EV -> #43 Volume Guard.
  let physics=null,physicsForecast=null,physicsIntegrityReport=null,physicalEv=null,error="";
  let resolvedCore=null,original7=null,corePB=.5,extended=null,rawPB=.5,clippedPB=.5,smoothedPB=.5,smoothingAlpha=1,smoothingStrength=0,smoothingProfile="off",finalPB=.5,bounds=null,mode="core",evDecision=null;
  const executionOrder=[];

  try{
    if(physicsBundle?.trained){
      executionOrder.push(finalModelUsesPhysicalEv()&&physicsBundleUsesPhysicalEv()?"particle_physics":"legacy_physics");
      physics=predictPhysics(seq);
      physicsForecast=unpackPhysicsForecast(physics);
      physicsIntegrityReport=physicsIntegrity(physics);
      physicalEv=physicsForecast?.physicalEv||null;
      if(physicalEv)executionOrder.push("physical_ev");
    }
  }catch(e){error="physics:"+String(e?.message||e||"runtime_error");physics=null;physicalEv=null;}

  try{
    executionOrder.push("frozen_core");
    resolvedCore=corePrediction||CORE.hazardChoose(seq);
    original7=buildOriginal7(seq,resolvedCore);
    corePB=original7.core_p_b;
    rawPB=corePB;clippedPB=corePB;smoothedPB=corePB;finalPB=corePB;

    if(physics&&final56Bundle?.trained&&modelSemanticsCompatible()){
      executionOrder.push("final_xgboost");
      extended=buildExtended(corePB,original7,physics);
      rawPB=predictFinalProbability(final56Bundle,extended,final56Bundle.feature_names||EXTENDED_NAMES);
      bounds=applyProbabilityBounds(rawPB,original7.round_index,extended.at(-1),extended[1]);clippedPB=bounds.value;
      const smoothing=applyDynamicSmoothing(clippedPB,original7.round_index,extended.at(-1),bounds,extended[1]);
      smoothedPB=smoothing.value;smoothingAlpha=smoothing.alpha;smoothingStrength=smoothing.strength;smoothingProfile=smoothing.profile;finalPB=smoothedPB;

      executionOrder.push("final_ev");
      const pTie=clip(+physics[23]),pPlayer=1-finalPB;
      const evBanker=finalPB*.95-pPlayer,evPlayer=pPlayer-finalPB;
      const policy=decisionPolicy(original7.round_index,extended.at(-1));

      executionOrder.push("volume_guard");
      const guard=volumeGuardState(original7.round_index,finalPB,policy);
      const effectiveBand=Math.max(policy.bandMinimum,policy.confidenceBand-guard.bandRelief),effectiveActivationEv=Math.max(0,policy.activationEv-guard.evRelief);
      const distance=Math.abs(finalPB-.5);
      const direction=distance>=effectiveBand&&evBanker>effectiveActivationEv&&evBanker>evPlayer?"B":distance>=effectiveBand&&evPlayer>effectiveActivationEv&&evPlayer>evBanker?"P":"Skip";
      const edge=direction==="B"?evBanker:direction==="P"?evPlayer:0;
      const entryTier=direction==="Skip"?"skip":distance>=effectiveBand+policy.strongMargin?"strong":"weak";
      evDecision={pTie,pPlayer,evBanker,evPlayer,minEv:policy.minEv,activationEv:policy.activationEv,effectiveActivationEv,softBand:policy.softBand,confidenceBand:policy.confidenceBand,effectiveConfidenceBand:effectiveBand,volumeGuardActive:guard.active,entryTier,stakeMultiplier:entryTier==="weak"?.5:entryTier==="strong"?1:0,policyEnabled:policy.enabled,policyProfile:policy.profile,direction,
        finalDirection:direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip",confidence:direction==="Skip"?0:softConfidence(edge,{...policy,activationEv:effectiveActivationEv})};
      mode="final56";
    }else if(physics&&final56Bundle?.trained&&!modelSemanticsCompatible()){
      error=error?error+";feature_semantics_mismatch":"feature_semantics_mismatch";
    }
  }catch(e){
    const message=String(e?.message||e||"runtime_error");error=error?error+";"+message:message;
    if(!resolvedCore)resolvedCore=corePrediction||CORE.hazardChoose(seq);
    if(!original7)original7=buildOriginal7(seq,resolvedCore);
    corePB=original7.core_p_b;rawPB=corePB;clippedPB=corePB;smoothedPB=corePB;finalPB=corePB;mode="core";
  }

  const direction=evDecision?.direction||resolvedCore.direction,finalPP=1-finalPB;
  const confidence=evDecision?.confidence??resolvedCore.confidence??0;
  const physicalEvBanker=physicalEv?.banker??null,physicalEvPlayer=physicalEv?.player??null,physicalEvGap=physicalEv?.gap??null,particleUncertainty=physicalEv?.uncertainty??null;
  return {...resolvedCore,direction,final_direction:evDecision?.finalDirection||(direction==="B"?"莊 B":"閒 P"),confidence,ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,
    physical_ev_banker:physicalEvBanker,physical_ev_player:physicalEvPlayer,physical_ev_gap:physicalEvGap,particle_uncertainty:particleUncertainty,
    min_ev:evDecision?.minEv??null,activation_ev:evDecision?.activationEv??null,effective_activation_ev:evDecision?.effectiveActivationEv??null,soft_band:evDecision?.softBand??0,confidence_band:evDecision?.confidenceBand??0,effective_confidence_band:evDecision?.effectiveConfidenceBand??0,volume_guard_active:evDecision?.volumeGuardActive??false,entry_tier:evDecision?.entryTier??"core",stake_multiplier:evDecision?.stakeMultiplier??1,decision_policy_enabled:evDecision?.policyEnabled??false,decision_policy_profile:evDecision?.policyProfile??"hard_ev",probabilities:{B:finalPB,P:finalPP},
    regime:mode==="final56"?(direction==="Skip"?"EV 觀望":direction!==resolvedCore.direction?"Final XGB換邊":"Final XGB裁決"):resolvedCore.regime,
    finalProbability:{version:VERSION,active:mode==="final56",mode,corePB,rawPB,clippedPB,smoothedPB,smoothingAlpha,smoothingStrength,smoothingProfile,finalPB,bounds,
      physical_ev_banker:physicalEvBanker,physical_ev_player:physicalEvPlayer,physical_ev_gap:physicalEvGap,particle_uncertainty:particleUncertainty,execution_order:executionOrder,
      p_tie:evDecision?.pTie??null,p_player:evDecision?.pPlayer??null,ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,min_ev:evDecision?.minEv??null,
      activation_ev:evDecision?.activationEv??null,effective_activation_ev:evDecision?.effectiveActivationEv??null,soft_band:evDecision?.softBand??0,confidence_band:evDecision?.confidenceBand??0,effective_confidence_band:evDecision?.effectiveConfidenceBand??0,volume_guard_active:evDecision?.volumeGuardActive??false,entry_tier:evDecision?.entryTier??"core",stake_multiplier:evDecision?.stakeMultiplier??1,decision_policy_enabled:evDecision?.policyEnabled??false,decision_policy_profile:evDecision?.policyProfile??"hard_ev",
      coreDirection:resolvedCore.direction,finalDirection:evDecision?.finalDirection||(direction==="B"?"莊 B":"閒 P"),flipped:direction!==resolvedCore.direction,
      original7,physics,physicsForecast,physicsIntegrity:physicsIntegrityReport,particleDiagnostics:lastParticleDiagnostics,dataQuality:dataQuality(seq),extended,error}};
}
function readHistory(){
  if(typeof localStorage==="undefined")return[];
  for(const key of["bgs256d_frozen_6x15_forward_v18","bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16"]){
    try{const r=JSON.parse(localStorage.getItem(key)||"null");if(r&&Array.isArray(r.history))return r.history.filter(x=>["B","P","T"].includes(x)).slice(-500);}catch(_){}
  }return[];
}
function getShoeId(){try{let id=localStorage.getItem(SHOE_KEY)||"";if(!id){id="shoe_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,8);localStorage.setItem(SHOE_KEY,id);}return id;}catch(_){return"browser_shoe";}}
function rotateShoeId(){try{localStorage.removeItem(SHOE_KEY);localStorage.removeItem(PENDING_KEY);}catch(_){}}
function readRows(){try{const r=JSON.parse(localStorage.getItem(TRAINING_KEY)||"[]");return Array.isArray(r)?r:[];}catch(_){return[];}}
function writeRows(rows){try{localStorage.setItem(TRAINING_KEY,JSON.stringify(rows.slice(-MAX_TRAINING_ROWS)));}catch(_){}}
function cloneFiniteVector(values,dimension){return Array.isArray(values)&&values.length===dimension&&values.every(value=>Number.isFinite(+value))?values.map(value=>+value):null;}
function registerPrediction(seq,prediction){
  const r=prediction.finalProbability||{},f=r.original7||buildOriginal7(seq,prediction);
  const createdAt=Date.now(),shoeId=getShoeId(),quality=r.dataQuality||dataQuality(seq),errorCorrectionSnapshot=r.particleDiagnostics?.pre_hand_snapshot||null;
  const pending={schema_version:SNAPSHOT_SCHEMA_VERSION,prediction_id:`${shoeId}:${createdAt}:${seq.join("")}`,shoe_id:shoeId,created_at:createdAt,
    history_fingerprint:seq.join(""),history_event_count:seq.length,directional_round_count:quality.directionalRounds,data_stage:quality.stage,
    entry_eligible:quality.entryEligible,preferred_entry:quality.preferredEntry,mode:r.mode||"core",model_versions:{final_probability:final56Bundle?.schema_version??null,physics:physicsBundle?.schema_version??null},
    core_p_b:f.core_p_b,round_index:f.round_index,estimated_total_hands:f.estimated_total_hands,remaining_ratio:f.remaining_ratio,
    sx_markov_p_same:f.sx_markov_p_same,stage:f.stage,depth:f.depth,original_7d:original7Vector(f),physics_48d:cloneFiniteVector(r.physics,PHYSICS_DIM),
    features_57d:cloneFiniteVector(r.extended,FEATURE_DIM),shoe_progress_weight:Number.isFinite(+r.extended?.[1])?+r.extended[1]:null,physics_noise_score:Number.isFinite(+r.extended?.at(-1))?+r.extended.at(-1):null,
    raw_p_b:Number.isFinite(+r.rawPB)?+r.rawPB:null,clipped_p_b:Number.isFinite(+r.clippedPB)?+r.clippedPB:null,smoothed_p_b:Number.isFinite(+r.smoothedPB)?+r.smoothedPB:null,
    smoothing_alpha:Number.isFinite(+r.smoothingAlpha)?+r.smoothingAlpha:1,smoothing_strength:Number.isFinite(+r.smoothingStrength)?+r.smoothingStrength:0,smoothing_profile:r.smoothingProfile||"off",final_p_b:Number.isFinite(+r.finalPB)?+r.finalPB:null,
    probability_bounds:r.bounds?[r.bounds.low,r.bounds.high]:null,min_ev:Number.isFinite(+r.min_ev)?+r.min_ev:null,
    activation_ev:Number.isFinite(+r.activation_ev)?+r.activation_ev:null,effective_activation_ev:Number.isFinite(+r.effective_activation_ev)?+r.effective_activation_ev:null,soft_band:Number.isFinite(+r.soft_band)?+r.soft_band:0,confidence_band:Number.isFinite(+r.confidence_band)?+r.confidence_band:0,effective_confidence_band:Number.isFinite(+r.effective_confidence_band)?+r.effective_confidence_band:0,volume_guard_active:r.volume_guard_active===true,entry_tier:r.entry_tier||"core",stake_multiplier:Number.isFinite(+r.stake_multiplier)?+r.stake_multiplier:1,decision_policy_enabled:r.decision_policy_enabled===true,decision_policy_profile:r.decision_policy_profile||"hard_ev",
    predicted_direction:r.finalDirection||prediction.direction||"",physics_integrity:r.physicsIntegrity||null,particle_physics:r.particleDiagnostics||null,
    physical_ev_banker:Number.isFinite(+r.physical_ev_banker)?+r.physical_ev_banker:null,physical_ev_player:Number.isFinite(+r.physical_ev_player)?+r.physical_ev_player:null,
    physical_ev_gap:Number.isFinite(+r.physical_ev_gap)?+r.physical_ev_gap:null,particle_uncertainty:Number.isFinite(+r.particle_uncertainty)?+r.particle_uncertainty:null,
    shoe_error_correction_version:Number.isFinite(+r.particleDiagnostics?.shoe_error_correction_version)?+r.particleDiagnostics.shoe_error_correction_version:0,pre_hand_snapshot:errorCorrectionSnapshot,shoe_error_correction:r.particleDiagnostics?.shoe_error_correction||null,
    execution_order:Array.isArray(r.execution_order)?r.execution_order.slice():[]};
  try{localStorage.setItem(PENDING_KEY,JSON.stringify(pending));}catch(_){}
}
function settlePending(actualOutcome){
  const actual=String(actualOutcome||"").toUpperCase();
  if(actual!=="B"&&actual!=="P"&&actual!=="T")return;
  let p=null;try{p=JSON.parse(localStorage.getItem(PENDING_KEY)||"null");}catch(_){}if(!p)return;
  const row={...p,settled_at:Date.now(),actual_outcome:actual,actual_b:actual==="B"?1:actual==="P"?0:null,is_directional_label:actual!=="T",shoe_error_correction_audit:settledErrorCorrectionAudit(p.pre_hand_snapshot,actual)},rows=readRows();
  if(!rows.length||rows.at(-1)?.shoe_id!==row.shoe_id||rows.at(-1)?.history_fingerprint!==row.history_fingerprint)rows.push(row);
  writeRows(rows);try{localStorage.removeItem(PENDING_KEY);}catch(_){}
}
function exportTrainingData(){
  const physicsNames=physicsBundleUsesPhysicalEv()?PHYSICS_NAMES:LEGACY_PHYSICS_NAMES;
  const featureNames=Array.isArray(final56Bundle?.feature_names)&&final56Bundle.feature_names.length===FEATURE_DIM?final56Bundle.feature_names:["core_p_b_external","shoe_progress_weight",...ORIGINAL7_NAMES.slice(1).map(x=>"original7_"+x),...physicsNames,"physics_noise_score"];
  return JSON.stringify({schema_version:SNAPSHOT_SCHEMA_VERSION,target:"actual_b_binary",feature_names:featureNames,original_7d_feature_names:ORIGINAL7_NAMES,physics_48d_feature_names:physicsNames,physics_semantics:physicsBundleUsesPhysicalEv()?"physical_ev_v2":"legacy_suit_v1",rows:readRows()},null,2);
}
function downloadTrainingData(){const blob=new Blob([exportTrainingData()],{type:"application/json;charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="bgs_final57_training_"+Date.now()+".json";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);}

async function fetchBundle(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw new Error(url+":HTTP"+r.status);return await r.json();}
async function loadModels(){
  status={physics:false,final56:false,errors:[]};
  const rs=await Promise.allSettled([fetchBundle(PHYSICS_URL),fetchBundle(FINAL56_URL)]);
  if(rs[0].status==="fulfilled"&&rs[0].value?.model_type==="baccarat_physics_multitask_mlp"){physicsBundle=rs[0].value;status.physics=!!physicsBundle.trained;}else status.errors.push("physics_model");
  if(rs[1].status==="fulfilled"&&rs[1].value?.model_type==="xgb_final_probability_classifier"&&rs[1].value?.feature_names?.length===FEATURE_DIM){final56Bundle=rs[1].value;status.final56=!!final56Bundle.trained;}else{final56Bundle=null;status.errors.push("final57_model");}
  if(status.physics&&status.final56&&!modelSemanticsCompatible()){status.final56=false;status.errors.push("feature_semantics_mismatch");}
  return status;
}
function saveSelection(direction){try{const old=JSON.parse(localStorage.getItem(STORAGE_KEY)||"null")||{},streak=old.last_selected===direction?Math.max(1,(+old.selection_streak||0)+1):1;localStorage.setItem(STORAGE_KEY,JSON.stringify({last_selected:direction,selection_streak:streak}));}catch(_){}}
function renderPrediction(p,n){
  const el=id=>document.getElementById(id),orb=el("directionOrb");if(!orb)return;const isSkip=p.direction==="Skip",isB=p.direction==="B";
  el("directionText").textContent=isSkip?"觀望":isB?"莊":"閒";el("directionCode").textContent=isSkip?"SKIP":isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";
  el("regime").textContent=p.regime;el("strength").textContent=isSkip?"觀望":p.strength>=.68?"穩定":p.strength>=.52?"中等":"保守";orb.className="direction-orb "+(isSkip?"":isB?"banker":"player");
  if(el("modePill"))el("modePill").textContent=p.finalProbability?.mode==="final56"?"Final 57D 完成":"Core 完成";
  if(el("roundCount"))el("roundCount").textContent=n;if(el("message"))el("message").textContent="第 "+(n+1)+" 局分析完成";
}
function installUI(){
  if(typeof document==="undefined")return;const old=document.getElementById("btnStart");if(!old)return;const btn=old.cloneNode(true);old.replaceWith(btn);
  btn.addEventListener("click",()=>{const history=readHistory();if(!history.length){const m=document.getElementById("message");if(m)m.textContent="請先輸入牌局紀錄";return;}
    const p=applyFinalPrediction(history);saveSelection(p.direction);registerPrediction(history,p);renderPrediction(p,history.length);});
  const b=document.getElementById("btnB"),p=document.getElementById("btnP"),t=document.getElementById("btnT");
  if(b)b.addEventListener("click",()=>settlePending("B"));if(p)p.addEventListener("click",()=>settlePending("P"));if(t)t.addEventListener("click",()=>settlePending("T"));
  const end=document.getElementById("btnEnd");if(end)end.addEventListener("click",rotateShoeId);
}
if(typeof window!=="undefined")window.__BGS_FINAL56__={version:VERSION,applyFinalPrediction,predictPhysics,estimateParticlePhysics,unpackPhysicsForecast,physicsIntegrity,dataQuality,buildOriginal7,historyVector,loadModels,setEstimatedTotalHands,getEstimatedTotalHands,effectiveParticleProgress,particleFusionWeight,physicalEvReliability,early35EvidenceWeight,early35PhysicalEvReliability,early35Enabled,shoeErrorCorrectionEnabled,updateShoeErrorCorrection,emptyShoeErrorCorrection,snapshotFromHandState,settledErrorCorrectionAudit,applyProbabilityBounds,dynamicEmaAlpha,
  registerPrediction,settlePending,exportTrainingData,downloadTrainingData,getTrainingRows:()=>readRows(),getTrainingCount:()=>readRows().length,getModelStatus:()=>({...status,mode:status.physics&&status.final56?"final56":"core"})};
loadModels();installUI();
})();
