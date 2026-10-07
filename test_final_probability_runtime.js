#!/usr/bin/env node
"use strict";

const fs=require("fs");
global.window=global;
const store=new Map();
global.localStorage={getItem:k=>store.has(k)?store.get(k):null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)};
global.document={getElementById:()=>null,createElement:()=>({click(){},remove(){}}),body:{appendChild(){}}};
global.URL={createObjectURL:()=>"",revokeObjectURL(){}};
global.Blob=function(){};

const finalBundle={
  schema_version:1,
  model_type:"xgb_final_probability_classifier",
  trained:true,
  link:"sigmoid",
  base_margin:Math.log(.90/.10),
  feature_names:Array.from({length:57},(_,i)=>"f"+i),
  probability_bounds:[.40,.60],
  training:{particle_physics_version:0,physics_direct_version:1,physics_primary_version:13,particle_filter_version:1},
  trees:[],
};
const useGeneratedBundle=process.env.BGS_USE_GENERATED_MODEL==="1";
global.fetch=async function(url){
  const name=String(url).replace(/^\.\//,"");
  try{
    const payload=JSON.parse(fs.readFileSync(name,"utf8"));
    if(name==="physics_multitask_model.json"&&!useGeneratedBundle){
      // Preflight runs before retraining, so reinterpret the checked-in legacy
      // 48D bundle as a direct-Physics ABI fixture only for runtime mechanics.
      const mock={...payload,feature_names:[...(payload.feature_names||[])]};
      const direct=["physical_ev_banker","physical_ev_player","physical_ev_gap","particle_uncertainty"];
      for(let i=0;i<4;i++)mock.feature_names[39+i]=direct[i];
      return {ok:true,status:200,json:async()=>mock};
    }
    return {ok:true,status:200,json:async()=>name==="final_probability_model.json"&& !useGeneratedBundle?finalBundle:payload};
  }catch(_){return {ok:false,status:404,json:async()=>({})};}
};

require("./app256forward.js");
require("./app256continuation.js");
require("./particle_filter_runtime.js");
require("./final_probability_runtime.js");

(async()=>{
  const api=global.__BGS_FINAL56__;
  if(!api)throw new Error("final probability runtime not installed");
  await api.loadModels();
  if(api.getModelStatus().mode!=="final56")throw new Error("direct model was not selected");

  const history="BPPBTBBPPTBBPPBTBP".split("");
  const core=global.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
  const out=api.applyFinalPrediction(history);
  const r=out.finalProbability;

  if(r?.mode!=="physics_primary")throw new Error("Physics-primary inference path was not used");
  if(r.primary_source!=="direct_physics_particle500_physical_ev"||r.xgb_role!=="auxiliary_filter_no_flip")throw new Error("primary/filter roles are incorrect");
  if(r.physics?.length!==48||r.extended?.length!==57)throw new Error("48D/57D contract changed");
  if(!r.physicsForecast||!r.physicsForecast.physicalEv)throw new Error("missing pre-Core Physical EV");
  if(!Number.isFinite(r.physics_raw_p_b)||!Number.isFinite(r.physics_smoothed_p_b)||!Number.isFinite(r.xgb_aux_p_b))throw new Error("primary/aux probabilities missing");
  if(!r.particleDiagnostics||r.physicsDiagnostics?.particle_filter_enabled!==true)throw new Error("Particle500 was not active");
  if(r.particleDiagnostics.particle_count!==500||r.particleDiagnostics.persistent_state!==false||r.particleDiagnostics.rebuild_from_scratch!==true)throw new Error("Particle500 stateless policy mismatch");
  if(r.particleDiagnostics.history_fingerprint!==history.join(""))throw new Error("Particle500 did not rebuild from current history");

  const expectedOrder=["direct_physics","particle_filter","physical_ev","physics_candidate","physics_ema","frozen_core","xgb_aux_filter","final_ev_guard","volume_guard"];
  if(JSON.stringify(r.execution_order)!==JSON.stringify(expectedOrder))throw new Error("production execution order mismatch");

  const candidate=r.primary_candidate;
  if(candidate==="B"&&!["B","Skip"].includes(out.direction))throw new Error("auxiliary filter flipped Banker candidate");
  if(candidate==="P"&&!["P","Skip"].includes(out.direction))throw new Error("auxiliary filter flipped Player candidate");
  if(candidate==="Skip"&&out.direction!=="Skip")throw new Error("auxiliary filter created action from Physics Skip");
  if(r.flipped_from_physics!==false)throw new Error("runtime reported a Physics direction flip");

  const hardB=api.auxiliaryCandidateFilter("B",.20,.10);
  const hardP=api.auxiliaryCandidateFilter("P",.80,.90);
  const softB=api.auxiliaryCandidateFilter("B",.44,.44);
  const xgbOnlyWeak=api.auxiliaryCandidateFilter("B",.60,.34);
  if(hardB.decision!=="skip"||hardP.decision!=="skip")throw new Error("severe joint Core/XGB conflict was not filtered");
  if(softB.decision!=="downgrade"||Math.abs(softB.shrink-.94)>1e-9)throw new Error("light auxiliary downgrade mismatch");
  if(xgbOnlyWeak.decision==="skip")throw new Error("XGB alone must not veto a Physics candidate");
  if(api.preserveCandidateDirection("B",.40)<.5||api.preserveCandidateDirection("P",.60)>.5)throw new Error("candidate-direction preservation failed");
  const lowUncertainty=api.calibratedPhysicsDirectionalPB(.60,10,.10,1),highUncertainty=api.calibratedPhysicsDirectionalPB(.60,10,.90,.45);
  if(!(lowUncertainty>highUncertainty&&lowUncertainty>.5&&highUncertainty>.5))throw new Error("Early-35 Physics calibration did not preserve direction/reduce overconfidence");
  if(!(api.physicsPrecisionPenalty(10,.90,.45)>api.physicsPrecisionPenalty(10,.10,1)))throw new Error("uncertainty-aware Physics EV penalty failed");

  const forecast=r.physicsForecast;
  if(Object.keys(forecast.nextCardCountProbabilities||{}).length!==3)throw new Error("missing 4/5/6-card forecast");
  if(Object.keys(forecast.rankExpectedConsumption||{}).length!==13)throw new Error("missing A-K conditional consumption");
  const cardExpectation=4*forecast.nextCardCountProbabilities["4_cards"]+5*forecast.nextCardCountProbabilities["5_cards"]+6*forecast.nextCardCountProbabilities["6_cards"];
  if(Math.abs(cardExpectation-forecast.expectedNextCardCount)>1e-9)throw new Error("incorrect next-hand card expectation");
  if(!r.physicsIntegrity||typeof r.physicsIntegrity.valid!=="boolean")throw new Error("missing Physics integrity audit");

  if(useGeneratedBundle){
    const generated=JSON.parse(fs.readFileSync("final_probability_model.json","utf8")),training=generated.training||{};
    if(training.runtime_role!=="auxiliary_candidate_filter_no_flip")throw new Error("generated XGB runtime role metadata missing");
    if(training.primary_predictor!=="direct_physics_physical_ev")throw new Error("generated primary predictor metadata missing");
    if((+training.physics_primary_version||0)<13)throw new Error("Physics Primary V13 metadata missing");
    if(JSON.stringify(training.execution_order)!==JSON.stringify(expectedOrder))throw new Error("generated execution-order metadata mismatch");
  }

  api.registerPrediction(history,out);
  api.settlePending("B");
  let rows=api.getTrainingRows();
  if(rows.length!==1)throw new Error("prediction snapshot was not retained");
  const snapshot=rows[0];
  if(snapshot.schema_version!==7||snapshot.actual_b!==1)throw new Error("invalid snapshot metadata");
  if(snapshot.physics_48d?.length!==48||snapshot.features_57d?.length!==57||snapshot.physics_direct_48d?.length!==48)throw new Error("snapshot feature dimensions changed");
  if(!Number.isFinite(snapshot.physics_smoothed_p_b)||!snapshot.physics_direct)throw new Error("Physics-primary EMA snapshot missing");
  if(!snapshot.particle_physics||snapshot.particle_physics.particle_count!==500||snapshot.particle_physics.persistent_state!==false)throw new Error("Particle500 snapshot missing");
  if(JSON.stringify(snapshot.execution_order)!==JSON.stringify(expectedOrder))throw new Error("snapshot execution order mismatch");
  if(snapshot.primary_candidate!==candidate)throw new Error("snapshot primary candidate mismatch");

  // Same B/P/T history must rebuild the same 500-particle posterior from scratch;
  // no previous particle state may leak into the next prediction.
  const repeat=api.applyFinalPrediction(history).finalProbability;
  if(repeat.particleDiagnostics?.history_fingerprint!==r.particleDiagnostics?.history_fingerprint)throw new Error("Particle500 history rebuild mismatch");
  if(Math.abs((repeat.particleDiagnostics?.corrected_directional_pb??0)-(r.particleDiagnostics?.corrected_directional_pb??0))>1e-12)throw new Error("Particle500 leaked mutable state across predictions");

  // EMA now smooths only the Physics-primary signal. XGB base margin may change,
  // but it cannot become the EMA source or flip the candidate direction.
  finalBundle.smoothing={method:"dynamic_post_clip_ema",profile:"balanced",enabled:true,early_alpha:.40,middle_alpha:.55,late_alpha:.70,noise_gain:0};
  finalBundle.base_margin=Math.log(.05/.95);
  const smoothHistory=[...history,"B"],smoothCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(smoothHistory),smoothOut=api.applyFinalPrediction(smoothHistory,smoothCore);
  const sr=smoothOut.finalProbability;
  if(sr.mode!=="physics_primary"||!(sr.smoothingAlpha>=.35&&sr.smoothingAlpha<=.75)||sr.smoothingProfile==="off")throw new Error("Physics-primary EMA fallback did not run");
  if(sr.primary_candidate==="B"&&!["B","Skip"].includes(smoothOut.direction))throw new Error("XGB flipped smoothed Banker candidate");
  if(sr.primary_candidate==="P"&&!["P","Skip"].includes(smoothOut.direction))throw new Error("XGB flipped smoothed Player candidate");
  delete finalBundle.smoothing;finalBundle.base_margin=Math.log(.90/.10);

  const tieHistory=[...history,"T"],tieCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(tieHistory),tiePrediction=api.applyFinalPrediction(tieHistory,tieCore);
  api.registerPrediction(tieHistory,tiePrediction);
  api.settlePending("T");
  rows=api.getTrainingRows();
  if(rows.length!==2||rows[1].actual_outcome!=="T"||rows[1].actual_b!==null||rows[1].is_directional_label!==false)throw new Error("tie context snapshot was not retained");

  const exported=JSON.parse(api.exportTrainingData());
  if(exported.schema_version!==7||exported.feature_names?.length!==57||exported.rows?.length!==2)throw new Error("snapshot export contract incomplete");

  console.log(JSON.stringify({ok:true,mode:r.mode,candidate,direction:out.direction,xgbFilter:r.aux_filter?.decision,snapshots:rows.length}));
})().catch(error=>{console.error(error);process.exit(1);});
