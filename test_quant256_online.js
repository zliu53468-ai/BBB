"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm");
const {OnlineLinUCB256,BlindPhysicsEstimator,context256,windowStats,ACTIVE_INDICES,VERSION}=
  require("./quant256_online_runtime");
const close=(a,b,tol=1e-10)=>assert(Math.abs(a-b)<tol,"not close: "+a+" vs "+b);
assert.equal(ACTIVE_INDICES.length,36);
assert.equal(VERSION,"BBB_QUANT256_ONLINE_V1");
assert.throws(()=>new OnlineLinUCB256({lambda:.89}),/lambda/);
assert.throws(()=>new OnlineLinUCB256({lambda:.97}),/lambda/);
for(const lambda of [.90,.94,.96])assert.equal(new OnlineLinUCB256({lambda}).lambda,lambda);
const q=new OnlineLinUCB256({lambda:.94}),p0=q.predict();
assert.equal(p0.context.length,256);
assert.equal(p0.shoe_context.length,128);
assert.equal(p0.road_context.length,128);
assert.equal(p0.direction,"Skip");assert.equal(p0.reason,"warmup");
assert.equal(p0.bet_allowed,false);
assert.equal(p0.blind_physics.actual_draw_count_known,false);
assert.equal(p0.blind_physics.actual_remaining_known,false);
assert.equal(p0.blind_physics.source,"simulated_unobserved");
assert.equal(p0.blind_physics.observed_cards,0);
assert.equal(p0.calibrated_probability,false);
assert.throws(()=>q.observe("X"),/invalid_outcome/);
const x0=p0.context[ACTIVE_INDICES[0]];
q.observe("B");
assert.equal(q.trainedRounds,1);
close(q.arms.B.A[0][0],1+x0*x0);
close(q.arms.B.b[0],x0);
close(q.arms.P.b[0],0);
const previousA=q.arms.B.A[0][0],previousB=q.arms.B.b[0];
const x1=q.predict().context[ACTIVE_INDICES[0]];
q.observe("P");
close(q.arms.B.A[0][0],.94*previousA+.06+x1*x1);
close(q.arms.B.b[0],.94*previousB);
assert(q.arms.P.b[0]>0);
const train=q.trainedRounds;q.observe("T");
assert.equal(q.trainedRounds,train,"Tie must not pretend to label a B/P direction");
assert.equal(q.history.join(""),"BPT");
assert.equal(q.prequential.eligible,0,"warmup must not count as a wager");
assert.equal(q.syncHistory(["B","P","T"]).history_length,3);
assert.equal(q.trainedRounds,2,"idempotent history replay");
const q2=new OnlineLinUCB256();q2.syncHistory(["B","P","T"]);
close(q2.predict().ucb_banker,q.predict().ucb_banker);
assert.equal(q2.syncHistory(["B","P"]).history_length,2,"undo must rebuild");
assert.equal(q2.trainedRounds,2);
q2.syncHistory([]);assert.equal(q2.trainedRounds,0);
const sim=new BlindPhysicsEstimator({particles:12});
assert.equal(sim.snapshot().virtual_remaining,416);
for(const r of ["B","P","T","B","B","P"])sim.observe(r);
assert(sim.snapshot().virtual_remaining<416);
assert.equal(sim.snapshot().actual_draw_count_known,false);
assert(sim.snapshot().expected_draws_recent>=4);
assert(sim.snapshot().expected_draws_recent<=6);
const s1=new BlindPhysicsEstimator({particles:12}),s2=new BlindPhysicsEstimator({particles:12});
for(const r of ["B","T","P","B"]){s1.observe(r);s2.observe(r);}
close(s1.snapshot().virtual_high_ratio,s2.snapshot().virtual_high_ratio);
const h="BBBPBBPPTBPPBBTPBPBBPP".split("");
const stats=context256(h,sim);
assert.equal(stats.vector.length,256);
assert(stats.vector.slice(12,128).every(x=>x===0),"pruned shoe features");
assert(stats.vector.slice(128+24).every(x=>x===0),"pruned long road features");
assert.equal(stats.features.entropy_delta,windowStats(h,3).entropy-windowStats(h,8).entropy);
assert.equal(stats.features.regularity_delta,windowStats(h,3).regularity-windowStats(h,8).regularity);
const q3=new OnlineLinUCB256({confidenceThreshold:.025,minSamples:100});
const trained=q3.syncHistory(h);
assert.equal(trained.bet_allowed,false,"quality gate is mandatory in warmup");
assert(trained.confidence_threshold>=.025);
assert(trained.confidence_threshold<=.025*2.3);
assert(["B","P"].includes(trained.raw_direction));
assert(Number.isFinite(trained.ucb_banker)&&Number.isFinite(trained.ucb_player));
const q4=new OnlineLinUCB256({minSamples:1});
q4.syncHistory(h);
const response=q4.predict();
assert.equal(response.bet_allowed,response.delta>=response.confidence_threshold);
assert.equal(response.direction,response.bet_allowed?response.raw_direction:"Skip");
assert.equal(response.prequential.eligible,
  response.prequential.correct+response.prequential.wrong+response.prequential.ties);
const repeat=OnlineLinUCB256.fromHistory(h),again=OnlineLinUCB256.fromHistory(h);
close(repeat.ucb_banker,again.ucb_banker);
close(repeat.delta,again.delta);

// Interface simulation: B/P/T click first saves via existing Frozen app, then
// online listener settles a hand; Start is captured before Frozen Direct.
const listeners={},dom={};
const ids=["btnStart","btnB","btnP","btnT","btnBack","btnEnd","directionText",
  "directionCode","directionOrb","confidence","regime","strength","modePill","message"];
for(const id of ids){
  listeners[id]=[];
  dom[id]={textContent:"",className:"",
    addEventListener(type,fn,capture){listeners[id].push({type,fn,capture:!!capture});}};
}
let saved=[],frozenClicks=0;
const data=new Map([["bgs256d_frozen_6x15_forward_v18",JSON.stringify({history:saved})]]);
const storage={getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)};
for(const result of ["B","P","T"])dom["btn"+result].addEventListener("click",()=>{
  saved.push(result);storage.setItem("bgs256d_frozen_6x15_forward_v18",JSON.stringify({history:saved}));
});
dom.btnStart.addEventListener("click",()=>{frozenClicks++;});
dom.btnBack.addEventListener("click",()=>{
  saved.pop();storage.setItem("bgs256d_frozen_6x15_forward_v18",JSON.stringify({history:saved}));
});
function click(id){
  let stopped=false;
  const event={preventDefault(){},stopImmediatePropagation(){stopped=true;}};
  for(const item of [...listeners[id]].sort((a,b)=>Number(b.capture)-Number(a.capture))){
    if(stopped)break;item.fn(event);
  }
}
const win={document:{getElementById:id=>dom[id]||null},localStorage:storage,
  location:{search:""},BBBQuant256Online:require("./quant256_online_runtime")};
const sandbox={window:win,document:win.document,URLSearchParams,console};
vm.runInNewContext(fs.readFileSync("./quant256_online_panel.js","utf8"),sandbox);
const ui=win.__BBB_QUANT256_ONLINE_PANEL__;
assert(ui&&ui.version===VERSION);
click("btnB");click("btnP");click("btnT");
assert.equal(ui.model.history.join(""),"BPT");
assert.equal(ui.model.trainedRounds,2);
click("btnStart");
assert.equal(frozenClicks,0,"Frozen click must be suppressed");
assert.equal(dom.directionText.textContent,"觀望");
assert(dom.confidence.textContent.includes("非勝率"));
assert.equal(ui.model.trainedRounds,2);
click("btnBack");assert.equal(ui.model.history.join(""),"BP");
click("btnEnd");assert.equal(ui.model.trainedRounds,0);
console.log("PASS quant256 online: full A/b decay, 256D, blind B/P/T, strict skip, idempotence/undo/UI");