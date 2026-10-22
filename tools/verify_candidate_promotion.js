"use strict";
const fs=require("node:fs"),crypto=require("node:crypto");
const {assess}=require("./no_pf_acceptance.js");
const sha=file=>crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
function verify(report,candidateFinal,candidatePhysics,productionFinal,productionPhysics){
  const result=assess(report);
  const fail=reason=>{result.passed=false;result.failures.push(reason);};
  if(report.current?.final_model_sha256!==sha(candidateFinal))fail("candidate final model differs from tested artifact");
  if(report.current?.physics_model_sha256!==sha(candidatePhysics))fail("candidate Physics differs from tested artifact");
  if(report.baseline?.final_model_sha256!==sha(productionFinal))fail("baseline is not the current production final model");
  if(report.baseline?.physics_model_sha256!==sha(productionPhysics))fail("baseline is not the current production Physics");
  for(const file of ["final_probability_runtime.js","macro_ema.js","cardcount_profiles.json"]){
    if(report.current?.runtime_fingerprints?.[file]!==sha(file))fail("runtime/config differs from tested artifact: "+file);
  }
  const config=JSON.parse(fs.readFileSync("cardcount_profiles.json","utf8"));
  if(report.current?.active_profiles?.entry!==config.default_entry_profile)fail("tested entry profile is not the deployment default");
  if(report.current?.active_profiles?.macro!==config.default_macro_profile)fail("tested Macro profile is not the deployment default");
  if(report.chronology!=="chronological_whole_shoe")fail("chronological whole-shoe audit missing");
  if(JSON.parse(fs.readFileSync(candidateFinal,"utf8")).training?.metrics?.retraining_success!==true)fail("internal training gates not passed");
  return result;
}
if(require.main===module){
  const [report,...files]=process.argv.slice(2);
  if(!report||files.length!==4)throw Error("Usage: node tools/verify_candidate_promotion.js report.json candidate-final.json candidate-physics.json production-final.json production-physics.json");
  const result=verify(JSON.parse(fs.readFileSync(report,"utf8")),...files);
  console.log(JSON.stringify(result,null,2));process.exitCode=result.passed?0:1;
}
module.exports={verify};
