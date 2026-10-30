/**
 * Quant256 V6 observational run-survival particle filter (isolated shadow).
 * Deterministic, stateless, 512 particles. Each particle is an independently
 * dealt hypothetical NEXT hand from the observed 8-deck 0..9 point counts.
 * Exact observed card points are mandatory; B/P/T alone NEVER enables a
 * count-conditioned resampling claim. This is a scenario stress test, NOT
 * calibrated baccarat probabilities or an actionable betting signal.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.__BBB_RUN_SURVIVAL_PF__=api;
})(typeof window!=="undefined"?window:null,function(){
 "use strict";
 const N=512,clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 const TOTAL=416,INITIAL=[128,32,32,32,32,32,32,32,32,32];
 function seedFrom(snapshot){
  const txt=(snapshot.history||[]).join("")+"|"+(snapshot.pointCounts||[]).join(",");
  let h=2166136261;for(let i=0;i<txt.length;i++){h^=txt.charCodeAt(i);h=Math.imul(h,16777619);}
  return h>>>0||1;
 }
 function rng32(seed){let x=seed;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}
 function draw(counts,rng){
  let total=0;for(const x of counts)total+=x;
  if(total<=0)throw Error("empty_shoe");
  let target=rng()*total;
  for(let i=0;i<10;i++){target-=counts[i];if(target<0){counts[i]--;return i;}}
  for(let i=9;i>=0;i--)if(counts[i]>0){counts[i]--;return i;}
  throw Error("invalid_draw");
 }
 function simulate(base,rng){
  const c=base.slice(),d=()=>draw(c,rng);
  const p1=d(),b1=d(),p2=d(),b2=d();
  let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
   let third=null;
   if(p<=5){third=d();p=(p+third)%10;}
   let bankerDraw=false;
   if(third===null)bankerDraw=b<=5;
   else if(b<=2)bankerDraw=true;
   else if(b===3)bankerDraw=third!==8;
   else if(b===4)bankerDraw=third>=2&&third<=7;
   else if(b===5)bankerDraw=third>=4&&third<=7;
   else if(b===6)bankerDraw=third===6||third===7;
   if(bankerDraw)b=(b+d())%10;
  }
  return b>p?"B":p>b?"P":"T";
 }
 function neutral(reason,snapshot={}){
  return {version:"CPSI_PF_SHADOW_V6",active:false,reason,
   particle_count:0,prior:null,posterior:null,ess_ratio:null,
   resampled:false,weights_applied:false,observed_evidence:false,
   input_run_length:snapshot.run_length||0,
   simulated_only:true,validated:false};
 }
 function estimate(snapshot){
  if(!snapshot||typeof snapshot!=="object")return neutral("missing_snapshot");
  const {pointCounts,history,run_side,run_length,count_source,
    follow_multiplier=1,break_multiplier=1}=snapshot;
  if(count_source!=="observed")return neutral("unobserved_cards",snapshot);
  if(!Array.isArray(history)||!history.every(x=>["B","P","T"].includes(x)))
   throw Error("invalid_history");
  if(!Array.isArray(pointCounts)||pointCounts.length!==10||
    !pointCounts.every(x=>Number.isInteger(x)&&x>=0))
   throw Error("invalid_observed_point_counts");
  if(pointCounts.some((x,i)=>x>INITIAL[i]))throw Error("counts_exceed_initial_shoe");
  const remaining=pointCounts.reduce((a,b)=>a+b,0);
  if(remaining<6||remaining>TOTAL)throw Error("invalid_remaining_cards");
  if(run_length<4||!["B","P"].includes(run_side))
   return neutral("short_streak",snapshot);
  if(!Number.isFinite(follow_multiplier)||!Number.isFinite(break_multiplier)||
    follow_multiplier<=0||break_multiplier<=0)throw Error("invalid_particle_multipliers");
  const rng=rng32(seedFrom(snapshot)),outcomes=Array(N),weight=Array(N);
  const prior={B:0,P:0,T:0},posterior={B:0,P:0,T:0};
  let sum=0,sumSq=0;
  for(let i=0;i<N;i++){
   const outcome=simulate(pointCounts,rng);
   outcomes[i]=outcome;prior[outcome]++;
   // No likelihood from an unknown future hand is claimed: explicit
   // hypothesis stress weighting only, conditional on observed past cards.
   const w=outcome===run_side?follow_multiplier:
     outcome==="T"?1:break_multiplier;
   weight[i]=w;sum+=w;sumSq+=w*w;posterior[outcome]+=w;
  }
  for(const k of ["B","P","T"]){prior[k]/=N;posterior[k]/=sum;}
  const essRatio=(sum*sum)/(N*sumSq);
  // Systematic resampling: preserve posterior scenarios, no hidden state.
  const resampled={B:0,P:0,T:0};
  const step=sum/N,offset=rng()*step;
  let idx=0,cumulative=weight[0];
  for(let k=0;k<N;k++){
   const target=offset+k*step;
   while(target>cumulative&&idx<N-1)cumulative+=weight[++idx];
   resampled[outcomes[idx]]++;
  }
  for(const k of ["B","P","T"])resampled[k]/=N;
  return {version:"CPSI_PF_SHADOW_V6",active:true,particle_count:N,
   observed_evidence:true,weights_applied:true,resampled:true,
   run_side,run_length,prior,posterior,resampled_distribution:resampled,
   ess_ratio:clamp(essRatio,0,1),follow_multiplier,break_multiplier,
   model:"counterfactual_count_weight_stress_not_likelihood",
   simulated_only:true,validated:false};
 }
 return {estimate,simulate,INITIAL,N};
});
