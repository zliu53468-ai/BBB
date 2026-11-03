"use strict";
/** V2.3: point-conditioned, causal particle likelihood and deterministic replay. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {BlindPhysicsEstimator,conditionalHandProposal}=require("./quant256_online_runtime.js");
const {PhysicsForecastFirstEngine}=require("./physics_forecast_first_runtime.js");
const nextRng=seed=>{
  let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
};
function totalsFromCards(a){
  const p1=a[0],b1=a[1],p2=a[2],b2=a[3];
  let p=(p1+p2)%10,b=(b1+b2)%10,j=4,p3=-1;
  if(p<8&&b<8){
    if(p<=5){p3=a[j++];p=(p+p3)%10;}
    const bankerDraw=p3<0?b<=5:
      b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
      b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false;
    if(bankerDraw)b=(b+a[j++])%10;
  }
  return{player:p,banker:b};
}
const exact=new Map();
function perm(prefix,remaining){
  if(!remaining.length){
    const x=totalsFromCards(prefix),k=x.player+","+x.banker;
    exact.set(k,(exact.get(k)||0)+1);return;
  }
  remaining.forEach((v,i)=>perm([...prefix,v],remaining.filter((_,j)=>i!==j)));
}
perm([],[1,2,3,4,5,6]);
const counts=[0,1,1,1,1,1,1,0,0,0];
for(const [player,banker] of [[7,7],[7,8],[9,6]]){
  const outcome=banker>player?"B":player>banker?"P":"T";
  const goal=(exact.get(player+","+banker)||0)/720;
  assert(goal>0);
  const rng=nextRng(12345+player*20+banker);
  const n=16000;let total=0;
  for(let i=0;i<n;i++){
    const a=conditionalHandProposal(counts,rng,outcome,{player,banker});
    assert(a.weight>=0&&a.weight<=1+1e-10);
    total+=a.weight;
    if(a.hand){
      assert.equal(a.hand.result,outcome);
      assert(a.hand.draws>=4&&a.hand.draws<=6);
      assert.equal(a.hand.counts.reduce((t,v)=>t+v,0),6-a.hand.draws);
      assert(a.hand.counts.every((v,k)=>Number.isInteger(v)&&v>=0&&v<=counts[k]));
    }else assert.equal(a.weight,0);
  }
  assert(Math.abs(total/n-goal)<.013,
    "Conditional first-four Rao-Blackwell mean differs from legal 6-card exhaustive likelihood: "+outcome+" "+total/n+" vs "+goal);
}
assert.throws(()=>conditionalHandProposal(counts,nextRng(1),"B",{player:8,banker:1}),
  /invalid_point_observation/);
assert.throws(()=>conditionalHandProposal(counts,nextRng(1),"B",{player:-1,banker:8}),
  /invalid_point_observation/);
const pf=new BlindPhysicsEstimator({particles:16,seed:20261103});
const previous=pf.snapshot();
assert.equal(previous.point_conditioned_rounds,0);
pf.observe("B",{player:7,banker:8});
const snap=pf.snapshot();
assert.equal(snap.point_conditioned_rounds,1);
assert.equal(snap.point_conditioned_last_round,true);
assert(snap.point_scores_are_not_observed_card_ranks);
assert(pf.particles.every(c=>c.reduce((a,b)=>a+b,0)>=410));
const initial=["B","P","T","B","P"];
const points=[null,null,null,null,null,{player:7,banker:8},{player:8,banker:2},{player:7,banker:7}];
const sequence=[...initial,"B","P","T"];
function replay(){
  const engine=new PhysicsForecastFirstEngine();
  const init=engine.syncHistory(initial);
  assert.equal(init.point_score_observation_count,0);
  assert.equal(init.score_conditioned_particles,false);
  const result=engine.syncHistory(sequence,points);
  assert.equal(result.point_score_observation_count,3);
  assert.equal(result.score_conditioned_particles,true);
  assert.equal(result.score_observations_are_actual_card_ranks,false);
  assert.equal(engine.quant.physics.pointConditionedRounds,3);
  assert.equal(result.action,"SKIP","final scores do not qualify as true observed-card BET");
  assert.equal(result.observed_card_evidence,false);
  assert.equal(result.physical_ev.samples,256*160);
  assert.equal(result.feature_weights.physics,5);
  assert.equal(result.feature_weights.road,.1);
  assert.equal(result.research_signal.direction,
    result.ev_banker>=result.ev_player?"B":"P");
  assert([0,.005,.035].includes(result.dynamic_gate));
  assert.deepEqual(result,engine.predict(),"repeated predictions cannot consume a hand");
  engine.syncHistory(sequence.slice(0,-1),points.slice(0,-1));
  assert.deepEqual(engine.syncHistory(sequence,points),result,
    "undo then replay with score evidence must reproduce same output");
  assert.throws(()=>engine.syncHistory(sequence,[...points.slice(0,-1),
    {player:1,banker:4}]),/invalid_point_observation/);
  return result;
}
const a=replay(),b=replay();
assert.deepEqual(a,b,"all-score chronological playback must be deterministic");
const plain=new PhysicsForecastFirstEngine().syncHistory(sequence);
assert.equal(plain.point_score_observation_count,0);
assert.notDeepEqual(a.physical_ev.simulated_probability,plain.physical_ev.simulated_probability,
  "legal score observation should condition PF composition beyond only winner");
assert.deepEqual(new PhysicsForecastFirstEngine().syncHistory(initial),
  new PhysicsForecastFirstEngine().syncHistory(initial,Array(initial.length).fill(null)),
  "BPT-only prediction must not change");
const page=fs.readFileSync("index.html","utf8");
const css=fs.readFileSync("styles.css","utf8");
const panel=fs.readFileSync("quant256_online_panel.js","utf8");
for(const id of ["btnStart","btnB","btnP","btnT","btnBack","btnEnd",
  "pointPlayerFinal","pointBankerFinal","bbbPointEntry","pointInputStatus"])
  assert(page.includes('id="'+id+'"'),"existing and new control "+id);
assert(page.indexOf('id="pointPlayerFinal"')<page.indexOf('id="pointBankerFinal"'),
  "player input must be LEFT of banker");
assert(page.includes('id="bbbPointEntry" hidden'),"point controls hidden until analysis");
assert(css.includes(".bbb-point-mode .bbb-results-block{display:none}"),
  "history controls must be swapped, not stacked");
assert(panel.includes('button.click()'),"score entry must reuse the original result handler once");
console.log("PASS V2.3 exact final-total likelihood on 6-card combinatorial deck, point PF update, deterministic undo/replay, historical BPT invariance and compact two-field controls");
