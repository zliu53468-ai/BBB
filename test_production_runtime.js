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
require("./physics_residual_runtime.js");

(async()=>{
  const api=global.__BGS_PHYSICS56__;
  if(!api)throw new Error("physics runtime not installed");
  await api.loadModels();
  const status=api.getModelStatus();
  if(status.mode!=="physics56")throw new Error("expected physics56 mode: "+JSON.stringify(status));
  const history="BPPBTBBPPTBBPPBTBP".split("");
  const core=global.__BGS256_CONTINUATION_TEST__.hazardChoose(history);
  const out=api.applyCorrection(history,core);
  if(out.residualBias?.mode!=="physics56")throw new Error("did not use physics56");
  if(out.residualBias.physics?.length!==48)throw new Error("physics dim mismatch");
  if(out.residualBias.extended?.length!==56)throw new Error("extended dim mismatch");
  if(Math.abs(out.residualBias.delta)>.1000001)throw new Error("delta clip violated");
  if(!["B","P"].includes(out.direction))throw new Error("invalid direction");
  console.log(JSON.stringify({ok:true,status,corePB:out.residualBias.corePB,delta:out.residualBias.delta,finalPB:out.residualBias.finalPB,direction:out.direction}));
})().catch(e=>{console.error(e);process.exit(1);});
