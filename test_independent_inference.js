"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const crypto=require("node:crypto"),locked=require("./tools/top_parameters/production_manifest.json").sha256;
for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","physics_multitask_model.json","final_probability_model.json"])assert.equal(crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex"),locked[f],"locked file changed: "+f);
const profile=JSON.parse(fs.readFileSync("cardcount_profile.json","utf8"));
async function make(independent=true){
 const reads=[],store=new Map([["bgs_xgb_final_shoe_id_v5","independent-test"]]);
 const config={...profile,inference:{mode:independent?"independent_v1":"legacy_stateful"}};
 const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>(reads.push(k),store.get(k)??null),setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async file=>({ok:true,json:async()=>file==="cardcount_profile.json"?config:JSON.parse(fs.readFileSync(file,"utf8"))})};
 ctx.window=ctx;vm.createContext(ctx);
 for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(f,"utf8"),ctx,{filename:f});
 const api=ctx.__BGS_FINAL56__;await api.loadModels();reads.length=0;return {api,store,reads};
}
const signature=p=>JSON.stringify({direction:p.direction,forecast:p.next_hand_forecast,raw:p.finalProbability.rawPB,clip:p.finalProbability.clippedPB,final:p.finalProbability.finalPB,physics:p.finalProbability.physics,features:p.finalProbability.extended,evB:p.ev_banker,evP:p.ev_player,band:p.effective_confidence_band});
(async()=>{
 for(const h of ["","B","BPPBTBBPPTBBPPBTBP","BP".repeat(30)]){
  const a=await make(),b=await make(),history=h.split("");
  const rows=[.01,.99,.5].map((v,i)=>({shoe_id:"independent-test",round_index:i+1,physics_smoothed_p_b:v,final_p_b:v,predicted_direction:"B",activation_ev:0}));
  b.store.set("bgs_xgb_final_training_v5",JSON.stringify(rows));
  b.store.set("bgs_macro_observations_v1",JSON.stringify({shoe_id:"independent-test",events:[{outcome:"B",card_count:6,player_score:0,banker_score:9}]}));
  b.store.set("bgs_browser_calibration_v1",JSON.stringify({poisoned:true}));
  a.reads.length=0;b.reads.length=0;
  const expected=a.api.applyFinalPrediction(history),actual=b.api.applyFinalPrediction(history);
  assert.equal(signature(actual),signature(expected),"old prediction state affected inference");
  assert.equal(actual.finalProbability.smoothingAlpha,1);assert.equal(actual.finalProbability.smoothingStrength,0);
  assert.equal(actual.finalProbability.inference_mode,"independent_v1");
  assert.equal(actual.finalProbability.macro_adjustment.delta,0);
  assert.equal(actual.finalProbability.finalPB,actual.finalProbability.clippedPB);
  assert.equal(actual.browser_calibration.applied,false);assert.equal(actual.volume_guard_active,false);
  for(const k of ["bgs_xgb_final_training_v5","bgs_macro_observations_v1","bgs_browser_calibration_v1"])assert(!b.reads.includes(k),"inference read previous state: "+k);
  assert.equal(actual.finalProbability.physics.length,48);assert.equal(actual.finalProbability.extended.length,57);
  const legacy=await make(false),previous=legacy.api.applyFinalPrediction(history);
  assert.equal(JSON.stringify(previous.finalProbability.extended),JSON.stringify(actual.finalProbability.extended),"57D definitions changed");
  assert.equal(signature(b.api.applyFinalPrediction(history)),signature(actual),"retry changed result");
 }
 const c=await make(),prefix="BPPBTBBPPT".split(""),p=c.api.applyFinalPrediction(prefix);
 c.api.registerPrediction(prefix,p);c.api.settlePending("B",{outcome:"B"},prefix);
 const history=[...prefix,"B"],fresh=await make();
 assert.equal(signature(c.api.applyFinalPrediction(history)),signature(fresh.api.applyFinalPrediction(history)),"settlement affected next prediction");
 const policy=c.api.decisionPolicy(1,.4);
 assert.equal(c.api.volumeGuardState(1,.6,policy).observations,0,"unexpected wager warmup");
 assert.equal(c.api.displayDirection({available:true,direction:"B",directional:{B:.51,P:.49}}),"B");
 assert.equal(c.api.displayDirection({available:true,direction:"B",directional:{B:.5099,P:.4901}}),"Skip");
 assert.equal(c.api.noSkipTestDirection({available:true,directional:{B:.5,P:.5}}),"B");
 console.log("independent inference: prior-state invariance, no wager warmup, unchanged 48D/57D, retry/settlement isolation and display guards passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
