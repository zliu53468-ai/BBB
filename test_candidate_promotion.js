"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),os=require("node:os"),path=require("node:path"),crypto=require("node:crypto");
const {verify}=require("./tools/verify_candidate_promotion.js");
const folder=fs.mkdtempSync(path.join(os.tmpdir(),"bbb-promotion-"));
const files=["candidate-final.json","candidate-physics.json","production-final.json","production-physics.json"].map(f=>path.join(folder,f));
const sha=f=>crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex");
try{
  fs.writeFileSync(files[0],JSON.stringify({training:{metrics:{retraining_success:true}}}));
  for(let i=1;i<files.length;i++)fs.writeFileSync(files[i],JSON.stringify({fixture:i}));
  const metric={bets:1000,brier:.24,hit_rate_on_bets:.56,realized_ev_per_bet:.04,skip_rate:.5};
  const metrics=Object.fromEntries(["all","early","middle","late"].map(stage=>[stage,{...metric}]));
  const config=require("./cardcount_profiles.json"),runtime_fingerprints={};
  for(const f of ["final_probability_runtime.js","macro_ema.js","cardcount_profiles.json"])runtime_fingerprints[f]=sha(f);
  const report={data_kind:"real_holdout_observations",chronology:"chronological_whole_shoe",shoes:100,
    current:{no_flip_violations:0,metrics,final_model_sha256:sha(files[0]),physics_model_sha256:sha(files[1]),runtime_fingerprints,active_profiles:{entry:config.default_entry_profile,macro:config.default_macro_profile}},
    baseline:{metrics:structuredClone(metrics),final_model_sha256:sha(files[2]),physics_model_sha256:sha(files[3])}};
  assert(verify(report,...files).passed);
  for(const mutate of [r=>r.data_kind="synthetic",r=>r.current.runtime_fingerprints["macro_ema.js"]="wrong",r=>r.current.final_model_sha256="wrong",r=>r.baseline.final_model_sha256="wrong",r=>r.current.active_profiles.entry="wrong",r=>r.current.metrics.all.realized_ev_per_bet=-1]){
    const bad=structuredClone(report);mutate(bad);assert(!verify(bad,...files).passed);
  }
  console.log("candidate promotion: real holdout, quality, model hashes and exact runtime/profile matching passed");
}finally{
  for(const file of files)if(fs.existsSync(file))fs.unlinkSync(file);
  fs.rmdirSync(folder);
}
