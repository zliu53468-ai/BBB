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
  training:{particle_physics_version:1},
  trees:[],
};
const useGeneratedBundle=process.env.BGS_USE_GENERATED_MODEL==="1";
global.fetch=async function(url){
  const name=String(url).replace(/^\.\//,"");
  try{
    const payload=JSON.parse(fs.readFileSync(name,"utf8"));
    return {ok:true,status:200,json:async()=>name==="final_probability_model.json"&& !useGeneratedBundle?finalBundle:payload};
  }catch(_){return {ok:false,status:404,json:async()=>({})};}
};

require("./app256forward.js");
require("./app256continuation.js");
require("./final_probability_runtime.js");

(async()=>{
  const api=global.__BGS_FINAL56__;
  if(!api)throw new Error("final probability runtime not installed");
  await api.loadModels();
  if(api.getModelStatus().mode!=="final56")throw new Error("direct model was not selected");
  if(api.early35Enabled())throw new Error("legacy model unexpectedly enabled Early-35 behavior");

  const history="BPPBTBBPPTBBPPBTBP".split("");
  const core=global.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
  const particleA=api.estimateParticlePhysics(history),particleB=api.estimateParticlePhysics(history);
  if(JSON.stringify(particleA.physics)!==JSON.stringify(particleB.physics))throw new Error("particle physics is not deterministic");
  if(particleA.physics?.length!==48)throw new Error("particle physics dimension mismatch");
  const particleCards=4*particleA.physics[0]+5*particleA.physics[1]+6*particleA.physics[2];
  const particleRanks=particleA.physics.slice(26,39).reduce((a,b)=>a+b,0);
  if(Math.abs(particleCards-particleRanks)>1e-6)throw new Error("particle card-count/rank-consumption mismatch");
  if(!(particleA.diagnostics.expected_consumed_cards>4*history.length&&particleA.diagnostics.expected_consumed_cards<6*history.length))throw new Error("particle consumed-card posterior invalid");
  const physicalEvB=particleA.physics[23]*.95-particleA.physics[24],physicalEvP=particleA.physics[24]-particleA.physics[23];
  if(Math.abs(particleA.physics[39]-physicalEvB)>1e-9||Math.abs(particleA.physics[40]-physicalEvP)>1e-9||Math.abs(particleA.physics[41]-(physicalEvB-physicalEvP))>1e-9)throw new Error("particle Physical EV mismatch");
  if(!(particleA.physics[42]>=0&&particleA.physics[42]<=1))throw new Error("particle uncertainty feature invalid");

  // No Core is supplied here: production must compute Physics/Physical EV first.
  const out=api.applyFinalPrediction(history);
  const r=out.finalProbability;
  if(r?.mode!=="final56")throw new Error("final56 inference path was not used");
  if(r.physics?.length!==48||r.extended?.length!==57)throw new Error("incorrect direct-model feature shape");
  const forecast=r.physicsForecast;
  if(!forecast||Object.keys(forecast.nextCardCountProbabilities||{}).length!==3)throw new Error("missing next-hand card-count forecast");
  if(Object.keys(forecast.rankExpectedConsumption||{}).length!==13)throw new Error("missing A-K consumption forecast");
  if(useGeneratedBundle&&(!forecast.physicalEv||!Number.isFinite(forecast.physicalEv.banker)||!Number.isFinite(forecast.physicalEv.player)))throw new Error("missing pre-Core Physical EV forecast");
  const cardExpectation=4*forecast.nextCardCountProbabilities["4_cards"]+5*forecast.nextCardCountProbabilities["5_cards"]+6*forecast.nextCardCountProbabilities["6_cards"];
  if(Math.abs(cardExpectation-forecast.expectedNextCardCount)>1e-9)throw new Error("incorrect next-hand card expectation");
  if(!r.physicsIntegrity||typeof r.physicsIntegrity.valid!=="boolean")throw new Error("missing physics integrity report");
  const firstPhysicsStep=useGeneratedBundle?"particle_physics":"legacy_physics";
  if(!Array.isArray(r.execution_order)||r.execution_order[0]!==firstPhysicsStep||r.execution_order.indexOf("frozen_core")<1)throw new Error("Physics was not executed before Frozen Core");
  if(useGeneratedBundle){
    const expectedOrder=["particle_physics","physical_ev","frozen_core","final_xgboost","final_ev","volume_guard"];
    if(JSON.stringify(r.execution_order)!==JSON.stringify(expectedOrder))throw new Error("Physical EV first execution order mismatch");
    if(!r.particleDiagnostics||!(r.particleDiagnostics.fusion_weight>=.1&&r.particleDiagnostics.fusion_weight<=.58))throw new Error("particle fusion diagnostics missing");
    if(!(r.particleDiagnostics.physical_ev_reliability>=.2&&r.particleDiagnostics.physical_ev_reliability<=.9)||!Number.isFinite(r.particleDiagnostics.effective_progress_round))throw new Error("adaptive stage diagnostics missing");
    if(!Number.isFinite(r.physical_ev_banker)||!Number.isFinite(r.physical_ev_player)||!Number.isFinite(r.physical_ev_gap)||!Number.isFinite(r.particle_uncertainty))throw new Error("Physical EV audit fields missing");
  }else if(r.particleDiagnostics?.enabled!==false)throw new Error("legacy model should not consume remapped Physical EV features");
  if(r.dataQuality?.stage!=="warm"||r.dataQuality?.directionalRounds!==15)throw new Error("incorrect prediction data stage");
  if(useGeneratedBundle){
    if(!(r.rawPB>=0&&r.rawPB<=1))throw new Error("generated model did not return a probability");
    if(!(r.finalPB>=.45&&r.finalPB<=.55))throw new Error("generated model ignored early dynamic bounds");
  }else{
    if(Math.abs(r.rawPB-.90)>1e-9)throw new Error("sigmoid probability was not used");
    if(Math.abs(r.finalPB-.55)>1e-9)throw new Error("early dynamic bounds were not applied");
  }
  if(!(Number.isFinite(out.ev_banker)&&Number.isFinite(out.ev_player)))throw new Error("EV fields missing");
  const activationThreshold=Number.isFinite(out.effective_activation_ev)?out.effective_activation_ev:(Number.isFinite(out.activation_ev)?out.activation_ev:out.min_ev);
  const confidenceBand=Number.isFinite(out.effective_confidence_band)?out.effective_confidence_band:0;
  const expectedDirection=Math.abs(r.finalPB-.5)>=confidenceBand&&out.ev_banker>activationThreshold&&out.ev_banker>out.ev_player?"B":Math.abs(r.finalPB-.5)>=confidenceBand&&out.ev_player>activationThreshold&&out.ev_player>out.ev_banker?"P":"Skip";
  if(out.direction!==expectedDirection)throw new Error("dynamic EV decision mismatch");
  if(useGeneratedBundle){
    if(!Number.isFinite(out.min_ev)||!Number.isFinite(activationThreshold)||out.min_ev<0||activationThreshold<0)throw new Error("generated EV thresholds invalid");
    const generated=JSON.parse(fs.readFileSync("final_probability_model.json","utf8")),training=generated.training||{},metrics=training.metrics||{};
    if((+training.stage_progress_version||0)<4)throw new Error("generated effective-progress v4 model metadata missing");
    if(metrics.retraining_success===true){
      if(out.decision_policy_profile!=="strict_selective_entry_v1"||out.soft_band!==0)throw new Error("promotable strict policy was not auditable in runtime");
      if(!Number.isFinite(out.confidence_band)||!Number.isFinite(out.effective_confidence_band)||!["strong","weak","skip"].includes(out.entry_tier))throw new Error("confidence-band metadata missing");
      if(out.direction==="Skip"&&Math.abs(out.confidence)>1e-12)throw new Error("generated Skip confidence mismatch");
      if(out.direction!=="Skip"&&!(out.confidence>0))throw new Error("generated action confidence missing");
    }else if(metrics.deployment_blocked!==true){
      throw new Error("non-promotable generated model was not marked deployment_blocked");
    }
  }else{
    const expectedConfidence=expectedDirection==="B"?out.ev_banker-out.min_ev:expectedDirection==="P"?out.ev_player-out.min_ev:0;
    if(Math.abs(out.confidence-expectedConfidence)>1e-9||Math.abs(out.min_ev-.020)>1e-9)throw new Error("default dynamic EV decision mismatch");
  }
  if(!useGeneratedBundle){
    finalBundle.calibration={method:"isotonic",x_thresholds:[0,1],y_thresholds:[.2,.6]};
    finalBundle.base_margin=Math.log(.25/.75);
    const calibratedOut=api.applyFinalPrediction(history,core);
    if(Math.abs(calibratedOut.finalProbability.rawPB-.30)>1e-9)throw new Error("isotonic calibration failed");
    delete finalBundle.calibration;
    finalBundle.base_margin=Math.log(.48/.52);
    const playerOut=api.applyFinalPrediction(history,core),playerResult=playerOut.finalProbability;
    if(playerOut.direction!=="P"||playerOut.final_direction!=="閒 P")throw new Error("binary Player EV normalization failed");
    if(Math.abs(playerResult.p_player-.52)>1e-9||Math.abs(playerOut.ev_player-.04)>1e-9||Math.abs(playerOut.confidence-.02)>1e-9)throw new Error("Player EV normalization or premium failed");
    finalBundle.base_margin=Math.log(.494/.506);
    const earlySkip=api.applyFinalPrediction(history,core);
    if(earlySkip.direction!=="Skip"||Math.abs(earlySkip.min_ev-.020)>1e-9)throw new Error("early EV threshold failed");
    const midHistory=Array.from({length:44},(_,i)=>i%2?"P":"B"),midCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(midHistory),midOut=api.applyFinalPrediction(midHistory,midCore);
    if(midOut.direction!=="P"||Math.abs(midOut.min_ev-.010)>1e-9)throw new Error("middle EV threshold failed");
    finalBundle.base_margin=Math.log(.497/.503);
    const lateHistory=Array.from({length:54},(_,i)=>i%2?"P":"B"),lateCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(lateHistory),lateOut=api.applyFinalPrediction(lateHistory,lateCore);
    if(lateOut.direction!=="P"||Math.abs(lateOut.min_ev-.005)>1e-9)throw new Error("late EV threshold failed");
    finalBundle.decision_policy={enabled:true,noise_threshold:1,max_noise_ev_penalty:.001,middle_relief:.001,late_relief:.001,middle_soft_band:.001,late_soft_band:.0015,min_confidence:.0005};
    finalBundle.base_margin=Math.log(.4955/.5045);
    const softOut=api.applyFinalPrediction(midHistory,midCore);
    if(softOut.direction!=="P"||softOut.decision_policy_enabled!==true)throw new Error("soft middle EV transition failed");
    if(Math.abs(softOut.min_ev-.009)>1e-9||Math.abs(softOut.activation_ev-.008)>1e-9||Math.abs(softOut.soft_band-.001)>1e-9)throw new Error("soft EV policy values mismatch");
    if(Math.abs(softOut.confidence-.0005)>1e-9)throw new Error("soft confidence floor failed");
    delete finalBundle.decision_policy;
    finalBundle.base_margin=Math.log(.90/.10);
  }

  api.registerPrediction(history,out);
  api.settlePending("B");
  let rows=api.getTrainingRows();
  if(rows.length!==1)throw new Error("directional prediction snapshot was not retained");
  const snapshot=rows[0];
  if(snapshot.schema_version!==7||snapshot.actual_b!==1||snapshot.is_directional_label!==true)throw new Error("invalid directional snapshot metadata");
  if(snapshot.physics_48d?.length!==48||snapshot.features_57d?.length!==57)throw new Error("prediction snapshot is missing exact model features");
  if(!Number.isFinite(snapshot.clipped_p_b)||!Number.isFinite(snapshot.smoothed_p_b)||snapshot.smoothing_alpha!==1||snapshot.smoothing_strength!==0)throw new Error("smoothing snapshot metadata is missing");
  if(!Number.isFinite(snapshot.confidence_band)||!Number.isFinite(snapshot.effective_confidence_band)||!["strong","weak","skip","core"].includes(snapshot.entry_tier))throw new Error("entry policy snapshot metadata is missing");
  if(!snapshot.particle_physics)throw new Error("particle physics snapshot metadata is missing");
  if(useGeneratedBundle){
    if(!Number.isFinite(snapshot.particle_physics.expected_consumed_cards)||!Number.isFinite(snapshot.physical_ev_banker)||!Number.isFinite(snapshot.physical_ev_player))throw new Error("Physical EV snapshot metadata is missing");
    if(JSON.stringify(snapshot.execution_order)!==JSON.stringify(["particle_physics","physical_ev","frozen_core","final_xgboost","final_ev","volume_guard"]))throw new Error("snapshot execution order mismatch");
  }
  const expectedFinalSchema=useGeneratedBundle?JSON.parse(fs.readFileSync("final_probability_model.json","utf8")).schema_version:1;
  if(snapshot.data_stage!=="warm"||snapshot.model_versions?.final_probability!==expectedFinalSchema)throw new Error("snapshot model metadata is missing");

  if(!useGeneratedBundle){
    finalBundle.smoothing={method:"dynamic_post_clip_ema",profile:"balanced",enabled:true,early_alpha:.40,middle_alpha:.55,late_alpha:.70,noise_gain:0};finalBundle.base_margin=Math.log(.40/.60);
    const smoothHistory=[...history,"B"],smoothCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(smoothHistory),smoothOut=api.applyFinalPrediction(smoothHistory,smoothCore);
    if(Math.abs(smoothOut.finalProbability.clippedPB-.45)>1e-9||Math.abs(smoothOut.finalProbability.smoothedPB-.51)>1e-9)throw new Error("post-Clip dynamic EMA failed");
    if(Math.abs(smoothOut.finalProbability.smoothingAlpha-.40)>1e-9||smoothOut.finalProbability.smoothingProfile!=="balanced")throw new Error("dynamic EMA alpha failed");
    delete finalBundle.smoothing;finalBundle.base_margin=Math.log(.90/.10);

    // V4 uses relative hand progress plus actual Particle card consumption;
    // these assertions are intentionally independent from Final EV/#43 policy.
    finalBundle.training={particle_physics_version:1,stage_progress_version:4,early35_version:1,shoe_error_correction_version:1};
    const lowCards={expected_consumed_cards:120,posterior_uncertainty:.30,recent_ess_ratio:1};
    const highCards={...lowCards,expected_consumed_cards:260};
    const p50=api.effectiveParticleProgress(40,highCards,50),p70=api.effectiveParticleProgress(40,highCards,70);
    if(!(p50>=.70&&p70<.70))throw new Error("relative estimated-total-hands progress mismatch");
    if(!(api.effectiveParticleProgress(50,highCards,60)>api.effectiveParticleProgress(50,lowCards,60)))throw new Error("card-consumption progress mismatch");
    const uncertain={...lowCards,posterior_uncertainty:.95},certain={...lowCards,posterior_uncertainty:.05};
    const round=50/60,card=120/(416-60);
    const highU=api.effectiveParticleProgress(50,uncertain,60),lowU=api.effectiveParticleProgress(50,certain,60);
    if(!(Math.abs(highU-round)<Math.abs(highU-card)&&Math.abs(lowU-card)<Math.abs(lowU-round)))throw new Error("uncertainty progress weighting mismatch");
    const early35Diag={expected_consumed_cards:180,posterior_uncertainty:.20,recent_ess_ratio:.90};
    const e5=api.early35EvidenceWeight(5,early35Diag,60),e15=api.early35EvidenceWeight(15,early35Diag,60),e25=api.early35EvidenceWeight(25,early35Diag,60),e35=api.early35EvidenceWeight(35,early35Diag,60);
    if(!(e5<e15&&e15<e25&&e25<e35))throw new Error("Early-35 evidence anchors are not ordered");
    if(!(api.early35EvidenceWeight(10,{...early35Diag,posterior_uncertainty:.80},60)<api.early35EvidenceWeight(10,early35Diag,60)))throw new Error("Early-35 uncertainty evidence shrinkage failed");
    if(!(api.early35EvidenceWeight(10,early35Diag,60)>api.early35EvidenceWeight(10,{...early35Diag,recent_ess_ratio:.20},60)))throw new Error("Early-35 ESS evidence scaling failed");
    if(Math.abs(api.early35EvidenceWeight(21,early35Diag,60)-api.early35EvidenceWeight(20,early35Diag,60))>=.10)throw new Error("Early-35 20/21 evidence transition jumped");
    if(Math.abs(api.early35EvidenceWeight(36,early35Diag,60)-e35)>=.10)throw new Error("Early-35 35/36 evidence transition jumped");
    if(Math.abs(api.early35PhysicalEvReliability(36,early35Diag,60)-api.early35PhysicalEvReliability(35,early35Diag,60))>=.10)throw new Error("Early-35 35/36 EV reliability transition jumped");
    if(!api.early35Enabled())throw new Error("Early-35 model version gate failed");
    const alpha49=api.dynamicEmaAlpha(50,.5,{},.49**3),alpha51=api.dynamicEmaAlpha(50,.5,{},.51**3);
    if(Math.abs(alpha49-alpha51)>=.03)throw new Error("V4 EMA discontinuity");
    const clip69=api.applyProbabilityBounds(.9,50,.1,.69**3),clip71=api.applyProbabilityBounds(.9,50,.1,.71**3);
    if(Math.abs(clip69.high-clip71.high)>=.02)throw new Error("V4 clip discontinuity");

    if(!api.shoeErrorCorrectionEnabled())throw new Error("shoe error-correction version gate failed");
    const makeSnapshot=(winner,gap=.08)=>({available:true,winner_distribution:winner,cards_distribution:[.25,.50,.25],player_final_point_distribution:Array(10).fill(.1),banker_final_point_distribution:Array(10).fill(.1),expected_consumed_cards:5,physical_ev_gap:gap,particle_uncertainty:.10,recent_ess_ratio:.90});
    const posterior={available:true,cards_distribution:[.20,.50,.30],player_final_point_distribution:[.2,...Array(9).fill(.8/9)],banker_final_point_distribution:[.2,...Array(9).fill(.8/9)],winner_distribution:[.2,.7,.1],expected_cards_consumed:5.2};
    const ordinary=api.updateShoeErrorCorrection(api.emptyShoeErrorCorrection(),[],makeSnapshot([.45,.45,.10]),posterior,"P",{recentEssRatio:.90,particleUncertainty:.10});
    const severe=api.updateShoeErrorCorrection(api.emptyShoeErrorCorrection(),[],makeSnapshot([.72,.20,.08]),posterior,"P",{recentEssRatio:.20,particleUncertainty:.80});
    if(!(ordinary.state.prediction_surprise<severe.state.prediction_surprise&&ordinary.state.shoe_posterior_health>severe.state.shoe_posterior_health&&ordinary.state.shoe_posterior_health>.90))throw new Error("surprise-sensitive health update mismatch");
    let repeated=severe;for(let i=0;i<5;i++)repeated=api.updateShoeErrorCorrection(repeated.state,repeated.memory,makeSnapshot([.72,.20,.08]),posterior,"P",{recentEssRatio:.20,particleUncertainty:.80});
    if(!(repeated.state.shoe_posterior_health<severe.state.shoe_posterior_health&&repeated.state.shoe_posterior_health>.30))throw new Error("asymmetric repeated-error health mismatch");
    let recovered=repeated;for(let i=0;i<8;i++)recovered=api.updateShoeErrorCorrection(recovered.state,recovered.memory,makeSnapshot([.50,.45,.05]),makeSnapshot([.50,.45,.05]),"B",{recentEssRatio:1,particleUncertainty:0});
    if(!(recovered.state.shoe_posterior_health>repeated.state.shoe_posterior_health))throw new Error("low-surprise recovery mismatch");
    if(!(severe.state.particle_observation_reused===true&&severe.state.particle_posterior_reweighted===false&&severe.state.physical_ev_reliability_multiplier>=.80&&severe.state.physical_ev_reliability_multiplier<=1))throw new Error("double-count or physical-EV correction mismatch");
    if(Math.sign(.12)===-Math.sign(.12*severe.state.physical_ev_reliability_multiplier))throw new Error("physical EV direction flipped");
    const reset=api.emptyShoeErrorCorrection();if(reset.shoe_posterior_health!==1||reset.draw_state_health!==1||reset.physical_ev_health!==1)throw new Error("new-shoe health reset mismatch");
    const correctionEstimate=api.estimateParticlePhysics(history);if(!correctionEstimate.diagnostics.pre_hand_snapshot?.available||correctionEstimate.diagnostics.error_memory_size>10)throw new Error("sequential correction replay audit missing");
    const correctedOut=api.applyFinalPrediction(history,core);if(correctedOut.finalProbability.physics?.length!==48||correctedOut.finalProbability.extended?.length!==57)throw new Error("error-correction changed dimensions");
  }

  const tieHistory=[...history,"T"],tieCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(tieHistory),tiePrediction=api.applyFinalPrediction(tieHistory,tieCore);
  api.registerPrediction(tieHistory,tiePrediction);
  api.settlePending("T");
  rows=api.getTrainingRows();
  if(rows.length!==2||rows[1].actual_outcome!=="T"||rows[1].actual_b!==null||rows[1].is_directional_label!==false)throw new Error("tie context snapshot was not retained");
  const exported=JSON.parse(api.exportTrainingData());
  if(exported.schema_version!==7||exported.feature_names?.length!==57||exported.rows?.length!==2)throw new Error("snapshot export contract is incomplete");

  console.log(JSON.stringify({ok:true,rawPB:r.rawPB,finalPB:r.finalPB,direction:out.direction,snapshots:rows.length}));
})().catch(error=>{console.error(error);process.exit(1);});
