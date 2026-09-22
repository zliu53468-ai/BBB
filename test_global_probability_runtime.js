#!/usr/bin/env node
"use strict";
const fs=require("fs");
global.window=global;
const store=new Map();
global.localStorage={getItem:k=>store.has(k)?store.get(k):null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)};
global.document={getElementById:()=>null,createElement:()=>({click(){},remove(){}}),body:{appendChild(){}}};
global.URL={createObjectURL:()=>"",revokeObjectURL(){}};
global.Blob=function(){};
global.fetch=async function(url){
  const name=String(url).replace(/^\.\//,"");
  try{
    const payload=JSON.parse(fs.readFileSync(name,"utf8"));
    return {ok:true,status:200,json:async()=>payload};
  }catch(e){return {ok:false,status:404,json:async()=>({})};}
};
require("./app256forward.js");
require("./app256continuation.js");
require("./global_probability_runtime.js");

(async()=>{
  const api=global.__BGS_GLOBAL56__;
  if(!api)throw new Error("global runtime not installed");
  await api.loadModels();
  const status=api.getModelStatus();
  if(status.mode!=="global56")throw new Error("expected global56 mode: "+JSON.stringify(status));

  const history="BPPBTBBPPTBBPPBTBP".split("");
  const core=global.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
  const out=api.applyGlobalPrediction(history,core);

  if(out.globalProbability?.mode!=="global56")throw new Error("did not use global56");
  if(out.globalProbability.physics?.length!==48)throw new Error("physics dim mismatch");
  if(out.globalProbability.extended?.length!==56)throw new Error("extended dim mismatch");
  if(!(out.globalProbability.rawPB>=0&&out.globalProbability.rawPB<=1))throw new Error("raw probability out of range");
  if(!(out.globalProbability.finalPB>=0&&out.globalProbability.finalPB<=1))throw new Error("final probability out of range");
  if(!["B","P"].includes(out.direction))throw new Error("invalid direction");

  console.log(JSON.stringify({
    ok:true,status,
    corePB:out.globalProbability.corePB,
    rawPB:out.globalProbability.rawPB,
    finalPB:out.globalProbability.finalPB,
    direction:out.direction
  }));
})().catch(e=>{console.error(e);process.exit(1);});
