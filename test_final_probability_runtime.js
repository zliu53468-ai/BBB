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
  if(r.physics?.length!==48||r.extended?.length!==56)throw new Error("incorrect direct-model feature shape");
  if(useGeneratedBundle){
    if(!(r.rawPB>=0&&r.rawPB<=1))throw new Error("generated model did not return a probability");
    if(!(r.finalPB>=.40&&r.finalPB<=.60))throw new Error("generated model ignored probability bounds");
  }else{
    if(Math.abs(r.rawPB-.90)>1e-9)throw new Error("sigmoid probability was not used");
    if(Math.abs(r.finalPB-.60)>1e-9)throw new Error("probability bounds were not applied");
  }
  if(out.direction!=="B")throw new Error("B/P decision rule changed");
  console.log(JSON.stringify({ok:true,rawPB:r.rawPB,finalPB:r.finalPB,direction:out.direction}));
})().catch(error=>{console.error(error);process.exit(1);});
