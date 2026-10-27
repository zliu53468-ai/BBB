"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),crypto=require("node:crypto");
const profile=JSON.parse(fs.readFileSync("cardcount_profile.json","utf8"));
async function make(legacy=false,commission=.05,gap=.02){
 const store=new Map([["bgs_xgb_final_shoe_id_v5","ev-direction-test"]]),nodes=new Map();
 const config={...profile,selective_entry_v2:{...profile.selective_entry_v2,enabled:false},display_gap:gap,settlement:{...profile.settlement,banker_commission:commission},direction_selection:{schema_version:1,mode:legacy?"physics_candidate_v1":"post_draw_ev_argmax_v1",ev_tie_tolerance:1e-12}};
 const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async f=>({ok:true,json:async()=>f==="cardcount_profile.json"?config:JSON.parse(fs.readFileSync(f,"utf8"))})};ctx.window=ctx;vm.createContext(ctx);
 for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(f,"utf8"),ctx,{filename:f});
 const api=ctx.__BGS_FINAL56__;await api.loadModels();return {api,ctx,store,nodes};
}
const close=(a,b)=>assert(Math.abs(a-b)<1e-11,a+" != "+b);
(async()=>{
 const a=await make(),b=await make(true),noFee=await make(false,0);
 const settlement=a.api.conditionalSettlement(.504,.1),forecast=a.api.evDirectionForecast(settlement,.2);
 assert.equal(forecast.direction,"P","EV, not win-probability argmax, must select the side");
 close(forecast.directional.B,.504);close(forecast.directional.P,.496);
 close(forecast.banker_ev,.95*.4536-.4464);close(forecast.player_ev,.4464-.4536);
 assert.equal(a.api.noSkipTestDirection(forecast),"P");
 assert.equal(a.api.displayDirection(forecast),"Skip","2pp display threshold retained");
 const feeFree=noFee.api.evDirectionForecast(noFee.api.conditionalSettlement(.504,.1),.2);assert.equal(feeFree.direction,"B");
 assert.equal(a.api.selectEvDirection(0,0).direction,"Balanced");
 assert.equal(a.api.selectEvDirection(-.1,-.2).direction,"B","negative EV can have a directional preference, never a forced wager");
 assert.equal(a.api.selectEvDirection(-.2,-.1).direction,"P");
 const equal=a.api.evDirectionForecast(a.api.conditionalSettlement(2/3.95,.1),.2);assert.equal(equal.direction,"Balanced");
 assert.equal(a.api.noSkipTestDirection(equal),"B","exact EV ties keep a deterministic test-only side");
 assert.equal(a.api.evDirectionForecast(a.api.settledOutcomeEv({B:0,P:0,T:1}),.2).available,false);
 const ui=await make(false,.05,.005);
 for(const id of ["directionOrb","directionText","directionCode","confidence","regime","strength","modePill","roundCount","message"])ui.nodes.set(id,{textContent:"",className:""});
 ui.ctx.document.getElementById=id=>ui.nodes.get(id)||null;
 ui.api.renderPrediction({next_hand_forecast:forecast,direction:"Skip"},1);
 assert.equal(ui.nodes.get("directionCode").textContent,"PLAYER");
 assert.match(ui.nodes.get("regime").textContent,/EV選向/);
 assert.match(ui.nodes.get("confidence").textContent,/50\.40/);
 let n=0,overrides=0;
 for(const h of ["B","T","BPPBTBBPPTBBPPBTBP","BP".repeat(25)]){
  for(const prior of [null,.01,.99]){
   for(const c of [a,b]){
    c.store.delete("bgs_xgb_final_training_v5");
    if(prior!==null)c.store.set("bgs_xgb_final_training_v5",JSON.stringify([{shoe_id:"ev-direction-test",round_index:Math.max(1,h.length-1),physics_smoothed_p_b:prior,final_p_b:prior,p_tie:.1,predicted_direction:"Skip"}]));
   }
   const p=a.api.applyFinalPrediction(h.split("")),legacy=b.api.applyFinalPrediction(h.split("")),r=p.finalProbability,f=p.next_hand_forecast;
   assert.equal(r.direction_selection,"post_draw_ev_argmax_v1");
   const best=a.api.selectEvDirection(p.ev_banker,p.ev_player);
   assert.equal(p.ev_direction.direction,best.direction);assert.equal(f.direction,best.direction);
   close(f.banker_ev,p.ev_banker);close(f.player_ev,p.ev_player);
   close(f.directional.B,r.finalPB);
   close(f.probabilities.B+f.probabilities.P+f.probabilities.T,1);
   assert.equal(JSON.stringify(r.extended),JSON.stringify(legacy.finalProbability.extended),"57D inputs changed");
   assert.equal(JSON.stringify(r.physics),JSON.stringify(legacy.finalProbability.physics),"draw inference changed");
   assert.equal(r.finalPB,legacy.finalProbability.finalPB,"model probability was rewritten to imitate EV");
   assert.equal(p.browser_calibration.used_for_direction,false);
   assert.equal(p.browser_calibration.applied,false);
   if(p.direction!=="Skip"){
    assert.equal(p.direction,best.direction);assert(best.best_ev>0);assert(r.skip_diagnostics.failed_conditions.length===0);
    if(p.direction!==r.primary_candidate)overrides++;
   }
   n++;
  }
 }
 assert(overrides>0,"old raw candidate still vetoes the EV-selected final direction");
 const original=Array(48).fill(0);original[23]=.4536;original[24]=.4464;original[25]=.1;original[42]=.2;
 assert.equal(a.api.nextHandForecast(original).direction,"B","raw winner-head diagnostic must remain unmodified");
 const locked=require("./tools/top_parameters/production_manifest.json").sha256;
 for(const f of ["app256forward.js","app256continuation.js","physics_multitask_model.json","final_probability_model.json"])assert.equal(crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex"),locked[f]);
 console.log(JSON.stringify({passed:true,ev_vs_probability_choice:true,runtime_cases:n,final_ev_overrode_raw_candidate:overrides,probability_and_57d_unchanged:true,negative_ev_wagers_forced:false}));
})().catch(e=>{console.error(e);process.exitCode=1;});
