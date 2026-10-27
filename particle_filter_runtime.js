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
  for(let i=0;i<particles.length;i++)