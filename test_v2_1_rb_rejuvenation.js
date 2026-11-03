"use strict";
/* V2.1: unbiased Rao-Blackwell likelihood + valid conditional resample-move.
 * This is numeric-quality testing, NOT a claim of positive betting edge.
 */
const assert=require("node:assert/strict");
const {performance}=require("node:perf_hooks");
const {BlindPhysicsEstimator,conditionalHandProposal}=require("./quant256_online_runtime.js");
const {deal,PhysicsForecastFirstEngine}=require("./physics_forecast_first_runtime.js");
function rng(seed){
  let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
}
function observedFromSix(a){
  const p1=a[0],b1=a[1],p2=a[2],b2=a[3];
  let p=(p1+p2)%10,b=(b1+b2)%10,j=4,p3=-1;
  if(p<8&&b<8){
    if(p<=5){p3=a[j++];p=(p+p3)%10;}
    const bDraw=p3<0?b<=5:
      b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
      b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false;
    if(bDraw)b=(b+a[j++])%10;
  }
  return b>p?"B":p>b?"P":"T";
}
function exactFromSix(cards){
  const tally={B:0,P:0,T:0};
  function perm(prefix,remaining){
    if(!remaining.length){tally[observedFromSix(prefix)]++;return;}
    remaining.forEach((v,i)=>perm(prefix.concat(v),remaining.filter((_,j)=>j!==i)));
  }
  perm([],cards);
  return Object.fromEntries(Object.entries(tally).map(([k,v])=>[k,v/720]));
}
const counts=[0,1,1,1,1,1,1,0,0,0],exact=exactFromSix([1,2,3,4,5,6]);
for(const side of ["B","P","T"]){
  const next=rng(0xabc123+side.charCodeAt(0));
  let mean=0;
  const iterations=20000;
  for(let i=0;i<iterations;i++){
    const proposal=conditionalHandProposal(counts,next,side);
    assert(proposal.weight>=0&&proposal.weight<=1+1e-12);
    mean+=proposal.weight;
    if(proposal.weight>0){
      assert(proposal.hand&&proposal.hand.result===side);
      assert(proposal.hand.draws>=4&&proposal.hand.draws<=6);
      assert.equal(proposal.hand.counts.reduce((s,v)=>s+v,0),6-proposal.hand.draws);
      proposal.hand.counts.forEach((v,k)=>assert(Number.isInteger(v)&&v>=0&&v<=counts[k]));
    }else assert.equal(proposal.hand,null);
  }
  assert(Math.abs(mean/iterations-exact[side])<.025,
    side+" RB likelihood not close to exact 6-card combinatorial odds");
}
// For the SAME target probability, conditioning on the first four
// mathematically reduces or preserves trial variance vs Bernoulli draw.
const n=18000,ra=rng(1729),naive=rng(1934);
let sumRB=0,sqRB=0,sumRaw=0,sqRaw=0;
for(let i=0;i<n;i++){
  const w=conditionalHandProposal(counts,ra,"B").weight;
  const raw=Number(deal(counts,naive).outcome==="B");
  sumRB+=w;sqRB+=w*w;sumRaw+=raw;sqRaw+=raw*raw;
}
const varRB=sqRB/n-(sumRB/n)**2,varRaw=sqRaw/n-(sumRaw/n)**2;
assert(varRB<=varRaw+.01,JSON.stringify({varRB,varRaw}));
const a=new BlindPhysicsEstimator({particles:8,seed:777});
a.weights=[.99,...Array(7).fill(.01/7)];
a.observe("B");
assert(a.lastResampled&&a.lastESS<4);
assert(a.lastRejuvenated>0,"repeated ancestry should get fresh conditional deals");
assert.equal(a.weights.reduce((x,y)=>x+y,0),1);
assert(new Set(a.particles.map(p=>p.join(","))).size>1,
  "resample-move should restore legal latent composition diversity");
for(const shoe of a.particles){
  shoe.forEach((n,k)=>assert(Number.isInteger(n)&&n>=0&&n<=a.particles[0][k]+32));
  assert(shoe.reduce((s,v)=>s+v,0)>=410);
}
const t0=performance.now(),sequence="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const run=()=>{
  const e=new PhysicsForecastFirstEngine(),report=e.syncHistory(sequence);
  assert.equal(report.action,"SKIP");
  assert.equal(report.physical_ev.samples,96*96);
  assert.equal(report.feature_weights.physics,5);
  assert.equal(report.feature_weights.road,.1);
  assert.equal(report.dynamic_gate===0||report.dynamic_gate===.005||report.dynamic_gate===.035,true);
  assert.equal(report.physical_ev.ev_uncertainty_diagnostics.calibrated_hit_rate,false);
  assert.equal(report.physical_ev.ev_uncertainty_diagnostics.validated_edge,false);
  assert(Number.isFinite(report.physical_ev.mc_se_gap));
  assert(Number.isFinite(report.physical_ev.ev_uncertainty_diagnostics.likelihood_relative_se));
  assert.deepEqual(e.predict(),report);
  return report;
};
assert.deepEqual(run(),run(),"deterministic full V2.1 replay");
console.log("PASS V2.1 RB conditional third-card numerical validity / raw variance",{
  exact,likelihood_variance_ratio:varRB/varRaw,
  rejuvenated:a.lastRejuvenated,elapsed_ms:Math.round(performance.now()-t0)});
