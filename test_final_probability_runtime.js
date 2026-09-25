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

  const history="BPPBTBBPPTBBPPBTBP".split("");
  const core=global.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
  const out=api.applyFinalPrediction(history,core);
  const r=out.finalProbability;
  if(r?.mode!=="final56")throw new Error("final56 inference path was not used");
  if(r.physics?.length!==48||r.extended?.length!==57)throw new Error("incorrect direct-model feature shape");
  const forecast=r.physicsForecast;
  if(!forecast||Object.keys(forecast.nextCardCountProbabilities||{}).length!==3)throw new Error("missing next-hand card-count forecast");
  if(Object.keys(forecast.rankExpectedConsumption||{}).length!==13)throw new Error("missing A-K consumption forecast");
  if(Object.keys(forecast.suitExpectedConsumption||{}).length!==4)throw new Error("missing suit consumption forecast");
  const cardExpectation=4*forecast.nextCardCountProbabilities["4_cards"]+5*forecast.nextCardCountProbabilities["5_cards"]+6*forecast.nextCardCountProbabilities["6_cards"];
  if(Math.abs(cardExpectation-forecast.expectedNextCardCount)>1e-9)throw new Error("incorrect next-hand card expectation");
  if(!r.physicsIntegrity||typeof r.physicsIntegrity.valid!=="boolean")throw new Error("missing physics integrity report");
  if(r.dataQuality?.stage!=="warm"||r.dataQuality?.directionalRounds!==15)throw new Error("incorrect prediction data stage");
  if(useGeneratedBundle){
    if(!(r.rawPB>=0&&r.rawPB<=1))throw new Error("generated model did not return a probability");
    if(!(r.finalPB>=.45&&r.finalPB<=.55))throw new Error("generated model ignored early dynamic bounds");
  }else{
    if(Math.abs(r.rawPB-.90)>1e-9)throw new Error("sigmoid probability was not used");
    if(Math.abs(r.finalPB-.55)>1e-9)throw new Error("early dynamic bounds were not applied");
  }
  if(out.direction!=="B"||out.final_direction!=="莊 B")throw new Error("EV direction rule changed");
  if(!(Number.isFinite(out.ev_banker)&&Number.isFinite(out.ev_player)&&out.ev_banker>out.ev_player))throw new Error("EV fields missing or inconsistent");
  if(!useGeneratedBundle){
    finalBundle.base_margin=Math.log(.48/.52);
    const playerOut=api.applyFinalPrediction(history,core),playerResult=playerOut.finalProbability;
    if(playerOut.direction!=="P"||playerOut.final_direction!=="閒 P")throw new Error("binary Player EV normalization failed");
    if(Math.abs(playerResult.p_player-.52)>1e-9||Math.abs(playerOut.ev_player-.04)>1e-9)throw new Error("Player EV still subtracts tie probability");
    finalBundle.base_margin=Math.log(.90/.10);
  }

  api.registerPrediction(history,out);
  api.settlePending("B");
  let rows=api.getTrainingRows();
  if(rows.length!==1)throw new Error("directional prediction snapshot was not retained");
  const snapshot=rows[0];
  if(snapshot.schema_version!==6||snapshot.actual_b!==1||snapshot.is_directional_label!==true)throw new Error("invalid directional snapshot metadata");
  if(snapshot.physics_48d?.length!==48||snapshot.features_57d?.length!==57)throw new Error("prediction snapshot is missing exact model features");
  if(snapshot.data_stage!=="warm"||snapshot.model_versions?.final_probability!==1)throw new Error("snapshot model metadata is missing");

  const tieHistory=[...history,"T"],tieCore=global.__BGS256_CONTINUATION_TEST__.hazardChoose(tieHistory),tiePrediction=api.applyFinalPrediction(tieHistory,tieCore);
  api.registerPrediction(tieHistory,tiePrediction);
  api.settlePending("T");
  rows=api.getTrainingRows();
  if(rows.length!==2||rows[1].actual_outcome!=="T"||rows[1].actual_b!==null||rows[1].is_directional_label!==false)throw new Error("tie context snapshot was not retained");
  const exported=JSON.parse(api.exportTrainingData());
  if(exported.schema_version!==6||exported.feature_names?.length!==57||exported.rows?.length!==2)throw new Error("snapshot export contract is incomplete");

  console.log(JSON.stringify({ok:true,rawPB:r.rawPB,finalPB:r.finalPB,direction:out.direction,snapshots:rows.length}));
})().catch(error=>{console.error(error);process.exit(1);});
