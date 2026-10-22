"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const M=require("./macro_ema.js"),model=JSON.parse(fs.readFileSync("final_probability_model.json"));
const state=M.createState("bounded");
for(let i=0;i<20;i++)M.update(state,{outcome:"B",card_count:i<10?4:6,banker_score:i<10?9:3,player_score:0});
const features=M.snapshot(state),config={enabled:true,min_observations:12,max_delta:99,coefficients:{six_card:10,low_score:10,point_diff:10}};
for(const p of [0,.01,.49,.506,.52,.99,1])for(const progress of [0,.3,.7,1])for(const noise of [.2,.6,.75,1]){
  const out=M.adjustment(p,features,progress,noise,config);
  assert(Math.abs(out.delta)<=.05+1e-12,"hard adjustment cap");
  assert((p>=2/3.95&&out.value>=2/3.95)||(p<2/3.95&&out.value<=2/3.95),"EV preference flip");
  if(noise>=.75||progress<=.3)assert.equal(out.delta,0);
}
assert.equal(M.adjustment(.6,features,1,.2,{...config,coefficients:{}}).delta,0);
assert.equal(M.adjustment(.6,null,1,.2,config).delta,0);
for(let i=0;i<4;i++)M.update(state,{outcome:"B"});
assert.equal(M.adjustment(.6,M.snapshot(state),1,.2,config).delta,0,"stale observations");

// Under the v2 coefficient-unit policy, .01 denotes a .01 probability shift
// for a normalized spread of 1 BEFORE the unchanged progress/noise gates.
const signed={...config,max_delta:.03,coefficient_units:"probability_delta",
  coefficients:{six_card:.01,low_score:0,point_diff:0}};
const plus=M.adjustment(.60,features,1,.2,signed);
const minus=M.adjustment(.60,features,1,.2,{...signed,coefficients:{six_card:-.01,low_score:0,point_diff:0}});
assert(Math.abs(plus.delta-.01*features.Six_Card_Spread)<1e-12);
assert(Math.abs(minus.delta+.01*features.Six_Card_Spread)<1e-12);
assert(Math.abs(plus.delta)<=.03&&Math.abs(minus.delta)<=.03);
assert.equal(M.adjustment(.60,features,1,.2,{...signed,enabled:false}).delta,0);
assert.equal(M.adjustment(.60,features,1,.8,signed).delta,0);
assert.equal(M.adjustment(.60,features,.2,.2,signed).delta,0);

// New BBB score-only UI: total player/banker points arrive, but card count
// remains unknown. Only actual observed totals may update the two EMA heads.
const scoreOnly=M.createState("score_only");
for(let i=0;i<24;i++)M.update(scoreOnly,{outcome:"B",banker_score:i<12?9:2,player_score:0});
const so=M.snapshot(scoreOnly);
assert.equal(so.counts.Six_Card,0,"never infer card count from the result or scores");
assert.equal(so.Six_Card_Spread,null);
assert.equal(so.counts.Low_Score,24);
assert.equal(so.counts.Point_Diff,24);
assert(so.Low_Score_Spread>0&&so.Point_Diff_Spread<0);
const scorePolicy={enabled:true,coefficient_units:"probability_delta",max_delta:.03,
  min_observations:12,progress_start:.3,progress_power:2,max_age:3,
  coefficients:{six_card:.02,low_score:.015,point_diff:-.01}};
const scoreAdjusted=M.adjustment(.65,so,1,.2,scorePolicy);
const scoreSignal=.015*so.Low_Score_Spread-.01*so.Point_Diff_Spread/9;
assert.equal(scoreAdjusted.ready,true,"score-only EMA must be able to activate");
assert.deepEqual(scoreAdjusted.active_features,["low_score","point_diff"]);
assert.deepEqual(scoreAdjusted.ignored_configured_features,["six_card"]);
assert.equal(scoreAdjusted.feature_ready.six_card,false);
assert.equal(scoreAdjusted.feature_ready.low_score,true);
assert.equal(scoreAdjusted.feature_ready.point_diff,true);
assert(Math.abs(scoreAdjusted.delta-scoreSignal)<1e-12,"do not impute missing 6-card readings");
assert.equal(M.adjustment(.65,so,1,.2,{...scorePolicy,coefficients:{six_card:.02}}).delta,0,
  "a missing card count cannot activate a six-card coefficient");
assert.equal(M.adjustment(.65,so,1,.2,{...scorePolicy,coefficients:{}}).delta,0,
  "zero live coefficients must remain neutral");
assert.deepEqual(M.adjustment(.65,so,1,.2,{...scorePolicy,
  coefficients:{point_diff:-.01}}).active_features,["point_diff"],
  "point difference must activate on its own");
// Missing score observations after enough rounds make both heads stale.
for(let i=0;i<4;i++)M.update(scoreOnly,{outcome:"P"});
const stale=M.adjustment(.65,M.snapshot(scoreOnly),1,.2,scorePolicy);
assert.equal(stale.delta,0);
assert.equal(stale.ready,false);
assert.deepEqual(stale.ignored_configured_features,["six_card","low_score","point_diff"]);
assert.equal(stale.reason,"insufficient_observations");

function context(store=new Map(),withoutParticle=false,finalOverride=null){
  const sandbox={console,setTimeout,clearTimeout,localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    document:{getElementById:()=>null},fetch:async url=>({ok:true,json:async()=>finalOverride&&url==="final_probability_model.json"?finalOverride:JSON.parse(fs.readFileSync(url,"utf8"))})};
  sandbox.window=sandbox;
  vm.createContext(sandbox);
  for(const f of ["app256forward.js","app256continuation.js","macro_ema.js","final_probability_runtime.js"]){
    if(withoutParticle&&f==="particle_filter_runtime.js")continue;
    vm.runInContext(fs.readFileSync(f,"utf8"),sandbox,{filename:f});
  }
  return {sandbox,api:sandbox.__BGS_FINAL56__,store};
}
(async()=>{
  const {api,store}=context();await api.loadModels();
  const alphaConfig={progress_alpha_anchors:[[0,.6],[1,.6]]};
  assert.equal(api.dynamicEmaAlpha(30,.5,alphaConfig,.125,{physics_uncertainty:.5}),.6);
  const alphaDefault=api.dynamicEmaAlpha(30,.5,{},.125,{physics_uncertainty:.5});
  assert.equal(api.dynamicEmaAlpha(30,.5,{progress_alpha_anchors:[[0,0],[1,1]]},.125,{physics_uncertainty:.5}),alphaDefault);
  const poison=context();
  Object.defineProperty(poison.sandbox,"__BGS_PARTICLE500__",{get(){throw Error("PF accessed");}});
  await poison.api.loadModels();
  assert.equal(poison.api.applyFinalPrediction(["B","P"]).finalProbability.mode,"physics_primary");
  const absentModel=JSON.parse(JSON.stringify(model));absentModel.trained=false;
  const missing=context(new Map(),true,absentModel);await missing.api.loadModels();
  const unavailableModel=missing.api.applyFinalPrediction(["B","P"]);
  assert.equal(unavailableModel.direction,"Skip");assert.equal(unavailableModel.probabilities.B,.5);
  const seq=[];
  for(let i=0;i<16;i++){seq.push(i%2?"P":"B");api.recordCompletedHand(seq,{card_count:i%3?6:4,player_score:i%2?8:1,banker_score:i%2?0:3});}
  const original=JSON.stringify(api.getMacroFeatures(seq));
  assert.equal(JSON.stringify(api.getMacroFeatures(seq)),original,"repeated analysis changed EMA");
  const reloaded=context(store);await reloaded.api.loadModels();
  assert.equal(JSON.stringify(reloaded.api.getMacroFeatures(seq)),original,"reload changed EMA");
  const short=seq.slice(0,-1);api.getMacroFeatures(short);
  api.recordCompletedHand(seq,{card_count:4,player_score:8,banker_score:0});
  const recomputed=M.createState();
  for(let i=0;i<seq.length;i++)M.update(recomputed,{outcome:seq[i],card_count:i===15?4:i%3?6:4,player_score:i%2?8:1,banker_score:i%2?0:3});
  assert.equal(api.getMacroFeatures(seq).Six_Card_Spread,M.snapshot(recomputed).Six_Card_Spread,"undo replacement");
  const bands=model.decision_policy.confidence_band,dyn=model.decision_policy.dynamic_band;
  const clean=api.dynamicBand(60,.4,bands,dyn),dirty=api.dynamicBand(60,.95,bands,dyn);
  assert(clean.confidenceBand<dirty.confidenceBand);
  assert.equal(dirty.reliefScale,0);assert.equal(dirty.finalBandRelief,0);
  assert(dirty.confidenceBand>=bands.late);
  assert.equal(api.dynamicBand(60,NaN,bands,dyn).reliefScale,0);
  assert(api.dynamicBand(65,.4,bands,dyn).confidenceBand<=api.dynamicBand(51,.4,bands,dyn).confidenceBand);
  const policy=api.decisionPolicy(60,.95);
  const shoe=api.getMacroFeatures(seq).shoe_id;
  store.set("bgs_xgb_final_training_v5",JSON.stringify(Array.from({length:5},(_,i)=>({shoe_id:shoe,round_index:i+45,final_p_b:.49,activation_ev:.002,predicted_direction:"Skip"}))));
  assert.equal(api.volumeGuardState(60,.501,policy).active,false);
  const lowPolicy=api.decisionPolicy(60,.4),lowGuard=api.volumeGuardState(60,.49,lowPolicy);
  assert.equal(lowGuard.active,true);assert(lowGuard.bandRelief>0);assert(lowGuard.bandRelief<=lowPolicy.confidenceBand*.25);
  store.delete("bgs_xgb_final_training_v5");
  api.setEstimatedTotalHands(50);assert.equal(api.decisionPolicy(40,.4).progress,.8);api.setEstimatedTotalHands(60);
  // Real model integration, independent of auxiliary side and future labels.
  for(const n of [16,42,56]){
    const history=Array.from({length:n},(_,i)=>["B","P","B","T","P"][i%5]);
    const result=api.applyFinalPrediction(history),r=result.finalProbability;
    assert.equal(r.error,"");assert.equal(r.physics.length,48);assert.equal(r.extended.length,57);
    assert(!r.execution_order.includes("particle_filter"));
    assert.equal(r.xgb_role,"auxiliary_filter_no_flip");assert.equal(r.flipped_from_physics,false);
    assert(result.direction==="Skip"||result.direction===r.primary_candidate);
    if(result.direction!=="Skip")assert((result.direction==="B"?result.ev_banker:result.ev_player)>0,"negative EV forced action");
    assert.equal(r.macro_adjustment.delta,0,"unfitted coefficients must stay neutral");
  }
  api.rotateShoeId();assert.equal(api.getMacroFeatures([]).rounds,0);assert.equal(api.getMacroFeatures([]).Six_Card_Spread,null);
  const unavailable=context(new Map(),true);await unavailable.api.loadModels();
  const refused=unavailable.api.applyFinalPrediction(["B","P"]);
  assert.equal(refused.finalProbability.mode,"physics_primary","PF absence must not block inference");
  const tuned=JSON.parse(JSON.stringify(model));tuned.decision_policy.macro_ema.coefficients={six_card:1,low_score:1,point_diff:0};
  const controlled=context(new Map(),false,tuned);await controlled.api.loadModels();
  const physics=Array(48).fill(0);physics[0]=1;physics[3]=1;physics[16]=1;physics[23]=.6;physics[24]=.3;physics[25]=.1;
  physics[26]=1;physics[39]=.27;physics[40]=-.3;physics[41]=.57;physics[42]=.1;physics[43]=300;physics[44]=.8;physics[45]=.1;physics[46]=.3;
  const adjusted=controlled.api.directPhysicsPrimary(physics,60,features);
  assert(adjusted.macro.delta>0,"configured structure correction never reached Physical EV");
  // Real runtime integration: a score-only shoe needs no Six_Card observations.
  const scoreBasedModel=JSON.parse(JSON.stringify(model));
  scoreBasedModel.decision_policy.macro_ema.coefficients={six_card:.02,low_score:.015,point_diff:-.01};
  const scoreBased=context(new Map(),false,scoreBasedModel);
  await scoreBased.api.loadModels();
  const scoreOnlyRuntime=scoreBased.api.directPhysicsPrimary(physics,60,M.snapshot(scoreOnly));
  // The scoreOnly state has had four missing-score rounds; only a fresh snapshot can adjust.
  const freshScore=M.createState("real_ui_scores");
  for(let i=0;i<24;i++)M.update(freshScore,{outcome:"B",banker_score:i<12?9:2,player_score:0});
  const scoreResult=scoreBased.api.directPhysicsPrimary(physics,60,M.snapshot(freshScore));
  assert(scoreResult.macro.ready&&scoreResult.macro.delta!==0,
    "score-only independent EMA does not reach the actual Physical EV integration");
  assert.equal(scoreResult.macro.active_features.join(","),"low_score,point_diff");
  assert.equal(scoreOnlyRuntime.macro.delta,0,"stale score-only snapshot must not adjust Physical EV");
  assert(Math.abs(adjusted.evBanker-(adjusted.baseEvBanker+1.95*.9*adjusted.macro.delta))<1e-12);
  assert.equal(adjusted.pT,.1,"macro adjustment changed tie mass");
  console.log(JSON.stringify({ok:true,checks:["bounded_adjustment","Python_JS_parity_separate_test","missing_and_stale","reload","undo","noise_gate","dimensions","no_flip","positive_final_EV","shoe_reset"]}));
})().catch(e=>{console.error(e);process.exit(1);});
