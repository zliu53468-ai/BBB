"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
function make(missing=false,opposed=false){
  const store=new Map(),nodes=new Map();
  const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async file=>({ok:true,json:async()=>{
    const m=JSON.parse(fs.readFileSync(file,"utf8"));
    // Historical winner-head mode remains tested for rollback; EV-led mode has a dedicated suite.
    if(file==="cardcount_profile.json")m.direction_selection={schema_version:1,mode:"physics_candidate_v1",ev_tie_tolerance:1e-12};
    if(missing&&file==="physics_multitask_model.json")m.trained=false;
    if(opposed&&file==="final_probability_model.json"){m.base_margin=Math.log(.001/.999);m.trees=[];m.calibration={method:"identity"};}
    return m;
  }})};ctx.window=ctx;

  vm.createContext(ctx);
  for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(f,"utf8"),ctx,{filename:f});
  const api=ctx.__BGS_FINAL56__;
  return {ctx,api,store,nodes};
}
const plain=v=>JSON.parse(JSON.stringify(v));
(async()=>{
  const a=make(),b=make(false,true);await Promise.all([a.api.loadModels(),b.api.loadModels()]);
  const h1="BBPPBTBP".split(""),h2="PBTBBBBPPPT".split("");
  const first=a.api.applyFinalPrediction(h1);
  a.api.applyFinalPrediction(h2);
  const repeat=a.api.applyFinalPrediction(h1);
  assert.deepEqual(plain(first.finalProbability.physics),plain(repeat.finalProbability.physics),"previous hand must not affect rebuilt PF");
  assert.equal(repeat.finalProbability.particleDiagnostics.history_fingerprint,h1.join(""));
  assert.equal(repeat.finalProbability.particleDiagnostics.rebuild_from_scratch,true);
  assert.notStrictEqual(first.finalProbability.physics,repeat.finalProbability.physics);
  assert.deepEqual(plain(repeat.next_hand_forecast.directional),plain(a.api.nextHandForecast(repeat.finalProbability.physics).directional));
  const v=Array(48).fill(0);v[23]=.46;v[24]=.44;v[25]=.10;v[42]=.2;
  const f=a.api.nextHandForecast(v);
  assert.equal(f.direction,"B");assert(Math.abs(f.directional.B-.46/.9)<1e-12);
  assert.equal(f.probabilities.T,.10);assert.equal(f.source,"direct_physics_winner_head");
  const equal=v.slice();equal[23]=equal[24]=.45;assert.equal(a.api.nextHandForecast(equal).direction,"Balanced");
  for(const bad of [null,[],Array(48).fill(0),v.map((x,i)=>i===23?NaN:x)])assert(!a.api.nextHandForecast(bad).available);
  for(const side of ["B","P"]){
    const advisory=a.api.auxiliaryCandidateFilter(side,side==="B"?.001:.999,side==="B"?.001:.999);
    assert.equal(advisory.decision,"keep");assert.equal(advisory.shrink,1);assert(advisory.advisoryOnly);
  }
  let predictions=0,skipsWithForecast=0;
  for(const pattern of ["BBBBPP","BP","BBPP","BPTBPBBPT"]){
    const history=[];
    for(let i=0;i<70;i++){
      history.push(pattern[i%pattern.length]);
      const core=a.ctx.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
      const low={...core,probabilities:{B:.001,P:.999}},high={...core,probabilities:{B:.999,P:.001}};
      const x=a.api.applyFinalPrediction(history,low),y=a.api.applyFinalPrediction(history,high),z=b.api.applyFinalPrediction(history,low);
      assert.deepEqual(plain(x.next_hand_forecast),plain(y.next_hand_forecast),"Core must not control forecast");
      assert.deepEqual(plain(x.next_hand_forecast),plain(z.next_hand_forecast),"XGB must not control forecast");
      assert.equal(x.direction,y.direction,"Core must not veto the bet");assert.equal(x.direction,z.direction,"XGB must not veto the bet");
      assert.equal(x.finalProbability.filteredPB,x.finalProbability.smoothedPB,"never erase probability on Skip");
      assert.equal(x.finalProbability.physics.length,48);assert.equal(x.finalProbability.extended.length,57);
      assert(x.finalProbability.execution_order.includes("particle_filter"));
      assert.equal(x.finalProbability.particleDiagnostics.particle_count,500);
      assert.equal(x.finalProbability.particleDiagnostics.persistent_state,false);
      assert(x.direction==="Skip"||x.direction===x.finalProbability.primary_candidate,"auxiliary flipped Physics");
      if(x.direction!=="Skip")assert((x.direction==="B"?x.ev_banker:x.ev_player)>0);
      if(x.direction==="Skip"&&x.next_hand_forecast.available)skipsWithForecast++;
      predictions++;
    }
  }
  assert(skipsWithForecast>0);
  const ids=["directionOrb","directionText","directionCode","confidence","regime","strength","modePill","roundCount","message"];
  for(const id of ids)a.nodes.set(id,{textContent:"",className:""});
  a.ctx.document.getElementById=id=>a.nodes.get(id)||null;
  const sample=a.api.applyFinalPrediction("BPBPBPTBBPPBP".split(""));
  a.api.renderPrediction({...sample,direction:"Skip",next_hand_forecast:f},13);
  assert.equal(a.nodes.get("directionText").textContent,"莊");
  assert.match(a.nodes.get("confidence").textContent,/莊 51\.11% \/ 閒 48\.89%/);
  assert.match(a.nodes.get("regime").textContent,/下注觀望/);
  const saved={...sample,direction:"Skip",finalProbability:{...sample.finalProbability,finalDirection:"觀望 Skip"}};
  a.api.registerPrediction(["B","P"],saved);a.api.settlePending("B",{outcome:"B"},["B","P"]);
  const row=a.api.getTrainingRows().at(-1);assert.equal(row.bet_direction,"Skip");assert.equal(row.predicted_direction,"觀望 Skip");assert(row.next_hand_forecast);
  const missing=make(true);await missing.api.loadModels();const unavailable=missing.api.applyFinalPrediction(["B","P"]);
  assert.equal(unavailable.direction,"Skip");assert(!unavailable.next_hand_forecast.available);
  a.api.renderPrediction(unavailable,2);assert.equal(a.nodes.get("confidence").textContent,"—");
  console.log(JSON.stringify({passed:true,predictions,skipsWithForecast,checks:["stateless PF500","Physics forecast only","Core/XGB advisory only","positive-EV bets","48D/57D","forecast separated from settlement","UI probabilities and tie","missing model"]}));
})().catch(e=>{console.error(e);process.exitCode=1;});
