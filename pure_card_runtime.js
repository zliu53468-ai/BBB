/** Stateless local-card model. Gaussian compositions are sensitivity scenarios,
 * not identified hidden-shoe posteriors. No LinUCB, road or context dependency. */
(function(root,factory){
 const node=typeof module==='object'&&module.exports;
 const api=factory(node?require('./pure_card_kernel.js'):root.BBBCardKernel,
                   node?require('./pure_card_eor_profile.json'):root.BBBCardEORProfile);
 if(node)module.exports=api;else root.BBBPureCards=api;
})(typeof self!=='undefined'?self:this,function(kernel,profile){
 'use strict';
 const VERSION='BBB_LOCAL_CARD_EOR_PF5000_MC10000_V1';
 const CONFIG=Object.freeze({particles:5000,sigma:2.5,alpha:.05,mcTrials:10000,
  depletion:true,evThreshold:.0025,mcSELimit:.0015,markovWindow:20,
  importanceBandwidth:.0001,markovPenalty:.0005});
 const FRESH=[128,32,32,32,32,32,32,32,32,32];
 if(!Array.isArray(kernel)||kernel.length!==6958||profile.coefficients.length!==10||
    !profile.coefficients.every(Number.isFinite))throw Error('model_assets_invalid');
 const terms=kernel.map(([key,n,b,p,t])=>{
  const rk=[];for(let r=0;r<10;r++){const k=(key>>>(r*3))&7;if(k)rk.push(r,k);}
  return {n,b,p,t,rk};
 });
 function seeded(seed){let x=seed>>>0||1;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
 function direction(x){return Math.abs(x)<1e-14?'SKIP':x>0?'B':'P';}
 function ev(q){return {B:.95*q.B-q.P,P:q.P-q.B};}
 function exact(counts){
  if(counts.length!==10||counts.some((n,i)=>!Number.isInteger(n)||n<0||n>FRESH[i])||counts.reduce((a,b)=>a+b,0)<6)throw Error('invalid_inventory');
  const total=counts.reduce((a,b)=>a+b,0),f=new Float64Array(70),den=[1,1,1];
  for(let r=0;r<10;r++){f[r*7]=1;for(let k=1;k<=6;k++)f[r*7+k]=f[r*7+k-1]*(counts[r]-k+1);}
  for(let n=4;n<=6;n++){let d=1;for(let k=0;k<n;k++)d*=total-k;den[n-4]=d;}
  let B=0,P=0,T=0;
  for(const term of terms){let w=1;for(let j=0;j<term.rk.length;j+=2)w*=f[term.rk[j]*7+term.rk[j+1]];
   w/=den[term.n-4];B+=w*term.b;P+=w*term.p;T+=w*term.t;}
  if(![B,P,T].every(Number.isFinite)||Math.abs(B+P+T-1)>1e-9)throw Error('probability_invalid');
  return {B,P,T};
 }
 const BASE=exact(FRESH),BASE_EV=ev(BASE);
 function parse(text){
  const parts=String(text).trim().toUpperCase().split(/[+＋,，\s]+/);
  if(parts.length<2||parts.length>3)throw Error('每邊須輸入 2～3 張實際牌面，不是最終總點');
  return parts.map(s=>{if(/^(10|J|Q|K|0)$/.test(s))return 0;if(s==='A')return 1;
   if(/^[1-9]$/.test(s))return Number(s);throw Error('牌面僅接受 A、2～10、J、Q、K 或 0');});
 }
 function bankerDraw(b,p3){return p3===null?b<=5:b<=2||b===3&&p3!==8||b===4&&p3>=2&&p3<=7||b===5&&p3>=4&&p3<=7||b===6&&(p3===6||p3===7);}
 function observation(player,banker){
  if(!Array.isArray(player)||!Array.isArray(banker)||![2,3].includes(player.length)||![2,3].includes(banker.length)||
   [...player,...banker].some(x=>!Number.isInteger(x)||x<0||x>9))throw Error('逐張牌面不合法');
  const p=(player[0]+player[1])%10,b=(banker[0]+banker[1])%10;
  if(p>=8||b>=8){if(player.length!==2||banker.length!==2)throw Error('天牌不可補牌');}
  else{
   if((p<=5)!==(player.length===3))throw Error('閒補牌不符合規則');
   if(bankerDraw(b,player.length===3?player[2]:null)!==(banker.length===3))throw Error('莊補牌不符合規則');
  }
  const used=Array(10).fill(0);for(const v of [...player,...banker])used[v]++;
  const pt=player.reduce((a,b)=>a+b,0)%10,bt=banker.reduce((a,b)=>a+b,0)%10;
  return {player:player.slice(),banker:banker.slice(),used,center:FRESH.map((n,i)=>n-used[i]),
   outcome:bt>pt?'B':pt>bt?'P':'T',features:{point_difference:bt-pt,draw_count:player.length+banker.length,
    player_cards:player.length,banker_cards:banker.length,card_type:p>=8||b>=8?'natural':player.length+banker.length===4?'stand':'draw'}};
 }
 function eor(used){const raw=used.reduce((s,n,i)=>s+n*profile.coefficients[i],0);
  return {raw,scaled:raw*CONFIG.alpha,direction:direction(raw),profile:profile.version,custom:true};}
 function sample(o,e){
  const rng=seeded(hash(o)^0x917ab),particles=[],logs=[];let max=-Infinity;
  function gaussian(){return Math.sqrt(-2*Math.log(Math.max(rng(),1e-15)))*Math.cos(2*Math.PI*rng());}
  for(let i=0;i<CONFIG.particles;i++){
   // Independent rounded normal, rejected above the known post-removal cap.
   // This is an explicit one-sided scenario proposal, not observed depletion.
   const counts=o.center.map(center=>{let x;do{x=Math.round(center+CONFIG.sigma*gaussian());}while(x<0||x>center);return x;});
   const hypothetical=eor(FRESH.map((n,j)=>n-counts[j])).scaled;
   const log=-.5*((hypothetical-e.scaled)/CONFIG.importanceBandwidth)**2;
   // Defined scenario target is proposal times this EOR compatibility kernel.
   // Proposal cancels in target/q; EOR is not a second data likelihood.
   particles.push({counts,weight:0});logs.push(log);max=Math.max(max,log);
  }
  let sum=0;for(let i=0;i<particles.length;i++)sum+=particles[i].weight=Math.exp(logs[i]-max);
  if(!Number.isFinite(sum)||sum<=0)throw Error('particle_weights_invalid');
  let sq=0;for(const p of particles){p.weight/=sum;sq+=p.weight*p.weight;}
  return {particles,ess:1/sq,prior:'truncated_gaussian_local_sensitivity_not_hidden_shoe_posterior'};
 }
 function hash(o){let h=2166136261;for(const x of [...o.player,11,...o.banker]){h^=x;h=Math.imul(h,16777619);}return h>>>0;}
 function composition(pf){let B=0,P=0,T=0;const cache=new Map();
  for(const p of pf.particles){const key=p.counts.join(',');let q=cache.get(key);if(!q){q=exact(p.counts);cache.set(key,q);}
   B+=p.weight*q.B;P+=p.weight*q.P;T+=p.weight*q.T;}
  return {probability:{B,P,T},ev:ev({B,P,T}),unique_compositions:cache.size};
 }
 function deal(counts,rng){
  const c=counts.slice();let total=c.reduce((a,b)=>a+b,0);
  function draw(){let x=rng()*total;for(let r=0;r<10;r++){x-=c[r];if(x<0){c[r]--;total--;return r;}}throw Error('draw_failed');}
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){let p3=null;if(p<=5){p3=draw();p=(p+p3)%10;}if(bankerDraw(b,p3))b=(b+draw())%10;}
  return b>p?'B':p>b?'P':'T';
 }
 function mc(pf,o){const rng=seeded(hash(o)^0xc00fee),cdf=[];let s=0;
  for(const p of pf.particles)cdf.push(s+=p.weight);cdf[cdf.length-1]=1;
  const tally={B:0,P:0,T:0};
  for(let i=0;i<CONFIG.mcTrials;i++){const x=rng();let lo=0,hi=cdf.length-1;
   while(lo<hi){const mid=(lo+hi)>>>1;if(x<cdf[mid])hi=mid;else lo=mid+1;}
   tally[deal(pf.particles[lo].counts,rng)]++;}
  const q=Object.fromEntries(Object.entries(tally).map(([k,n])=>[k,n/CONFIG.mcTrials]));
  return {probability:q,ev:ev(q),samples:CONFIG.mcTrials,depletion:true,
   win_probability_se:{B:Math.sqrt(q.B*(1-q.B)/CONFIG.mcTrials),P:Math.sqrt(q.P*(1-q.P)/CONFIG.mcTrials)}};
 }
 function diagnostic(rows){const recent=rows.slice(-20).filter(r=>['B','P'].includes(r.eor_direction)&&['B','P'].includes(r.actual));
  const transition={B:{B:0,P:0,T:0},P:{B:0,P:0,T:0}};
  for(const r of rows.slice(-20))if(transition[r.eor_direction]&&Object.hasOwn(transition[r.eor_direction],r.actual))transition[r.eor_direction][r.actual]++;
  const probabilities={};for(const side of ['B','P']){const n=Object.values(transition[side]).reduce((a,b)=>a+b,0);probabilities[side]=n?Object.fromEntries(Object.entries(transition[side]).map(([k,v])=>[k,v/n])):null;}
  const opposed=recent.length>=8&&recent.slice(-3).every(r=>r.eor_direction!==r.actual)&&recent.filter(r=>r.eor_direction!==r.actual).length/recent.length>=.6;
  return {window:20,support:rows.slice(-20).length,transition,probabilities,opposed,penalty:opposed?CONFIG.markovPenalty:0};
 }
 function decide(e,theory,m,diag){
  const td=theory.ev.B>=theory.ev.P?'B':'P',md=m.ev.B>=m.ev.P?'B':'P';
  const uplift=theory.ev[td]-BASE_EV[td],threshold=CONFIG.evThreshold+(diag?.penalty||0),se=m.win_probability_se[td];
  const reasons=[];if(e.direction==='SKIP'||e.direction!==td||td!==md)reasons.push('direction_disagreement');
  if(uplift<threshold)reasons.push('ev_uplift_below_threshold');
  if(se>CONFIG.mcSELimit)reasons.push('mc_uncertainty_above_limit');
  return {direction:reasons.length?'SKIP':td,reasons,uplift,threshold,mc_se:se,
   theoretical_direction:td,mc_direction:md,bet_allowed:false,validated_edge:false};
 }
 function run(player,banker,settledRows=[]){
  const o=observation(player,banker),e=eor(o.used),pf=sample(o,e),theory=composition(pf),m=mc(pf,o),diag=diagnostic(settledRows);
  return {version:VERSION,config:CONFIG,observation:o,eor:e,pf:{count:pf.particles.length,ess:pf.ess,prior:pf.prior},
   theory,mc:m,diagnostic:diag,decision:decide(e,theory,m,diag),baseline:BASE,
   validated_accuracy_gain:false,observed_full_shoe:false};
 }
 return Object.freeze({VERSION,CONFIG,BASE,BASE_EV,parse,observation,eor,sample,exact,composition,mc,diagnostic,decide,run,deal});
});
