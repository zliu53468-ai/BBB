"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const M=require("./browser_calibration.js");
function storage(){const map=new Map();return {map,getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k)};}
const forecast=(p=.55,uncertainty=.2)=>({available:true,source:"direct_physics_particle500_winner_head",direction:p===.5?"Balanced":p>.5?"B":"P",directional:{B:p,P:1-p},probabilities:{B:.9*p,P:.9*(1-p),T:.1},uncertainty});
const a=storage(),b=storage(),client=M.createClient(a),other=M.createClient(b),history=[];
const shoe="s",model="m",f=forecast(),raw=JSON.stringify(f);
assert.equal(client.apply(f,history,shoe,model).status.applied,false);
assert.equal(a.map.size,0,"inference must not learn or persist outcomes");
for(let i=0;i<50;i++){
  const ticket=client.prepare(history,f,shoe,model);
  assert.equal(client.commit(ticket,"B",history),false,"cannot train before outcome is durably in history");
  history.push("B");assert(client.commit(ticket,"B",history));
  assert(!client.commit(ticket,"B",history),"duplicate click must not train twice");
  if(i<39)assert.equal(client.apply(f,history,shoe,model).status.applied,false,"not enough forward-scored validation");
}
const learned=client.apply(f,history,shoe,model);
assert(learned.status.ready&&learned.status.applied);assert(learned.status.delta>0&&learned.status.delta<=.005+1e-12);
assert.equal(JSON.stringify(f),raw,"never mutate raw Physics/PF prediction");
assert.equal(learned.forecast.probabilities.T,f.probabilities.T);
assert(Math.abs(Object.values(learned.forecast.probabilities).reduce((x,y)=>x+y,0)-1)<1e-12);
assert.equal(client.apply(forecast(.499),history,shoe,model).forecast.direction,"P");
assert(client.apply(forecast(.499),history,shoe,model).forecast.directional.B<.5);
assert.equal(client.apply(forecast(.5),history,shoe,model).status.delta,0);
assert.equal(client.apply(forecast(.55,.95),history,shoe,model).status.delta,0);
assert.equal(client.apply(forecast(1,.95),history,shoe,model).status.delta,0);
assert.equal(client.apply(forecast(0,.95),history,shoe,model).status.delta,0);
assert.equal(client.apply(f,history,shoe,model,{enabled:false}).status.delta,0);
assert.equal(other.inspect(history,shoe,model).directional_samples,0,"other browser has no local learning");
const reload=M.createClient(a);assert.equal(reload.inspect(history,shoe,model).directional_samples,50);
assert.equal(reload.inspect(history,shoe,"new-model").directional_samples,0,"changed weights invalidate local calibration");
assert.equal(reload.inspect(history,"new-shoe",model).directional_samples,0);
const beforeTie=reload.inspect(history,shoe,model).directional_samples,ticket=reload.prepare(history,f,shoe,model);
history.push("T");assert(reload.commit(ticket,"T",history));assert.equal(reload.inspect(history,shoe,model).directional_samples,beforeTie);
assert.equal(reload.inspect(history,shoe,model).ties,1);
history.pop();history.pop();reload.reconcile(history,shoe,model);assert.equal(reload.inspect(history,shoe,model).directional_samples,49);
const corrected=history.slice(0,10);corrected[9]="P";reload.reconcile(corrected,shoe,model);
assert.equal(reload.inspect(corrected,shoe,model).directional_samples,9,"history replacement discards incompatible suffix");
assert.equal(reload.apply({available:false},corrected,shoe,model).status.reason,"model_unavailable");
reload.clear();assert.equal(a.getItem(M.KEY),null);
const quotaStore=storage(),quotaClient=M.createClient(quotaStore),quotaHistory=[];
for(let i=0;i<50;i++){const ticket=quotaClient.prepare(quotaHistory,f,shoe,model);quotaHistory.push("B");quotaClient.commit(ticket,"B",quotaHistory);}
assert(quotaClient.apply(f,quotaHistory,shoe,model).status.applied);
quotaStore.setItem=()=>{throw Error("quota");};
const quotaTicket=quotaClient.prepare(quotaHistory,f,shoe,model);quotaHistory.push("B");
assert.equal(quotaClient.commit(quotaTicket,"B",quotaHistory),false);
assert.equal(quotaClient.apply(f,quotaHistory,shoe,model).status.delta,0,"failed persistence must fall back to raw probability");
// The online comparison can reject an unhelpful candidate; repeated inference
// must neither change the comparison nor use the current outcome for itself.
const poorStore=storage(),poor=M.createClient(poorStore),bad=[];
for(let i=0;i<40;i++){
  const prior=poor.prepare(bad,forecast(.55),shoe,model);bad.push(i<24?"B":"P");poor.commit(prior,bad.at(-1),bad);
}
const status=poor.inspect(bad,shoe,model);assert(status.validation_count>=16);assert(!status.ready,"worse forward Brier must retain raw probability");
const before=poorStore.getItem(M.KEY);for(let i=0;i<5;i++)poor.apply(f,bad,shoe,model);assert.equal(poorStore.getItem(M.KEY),before);

(async()=>{
  const backing=storage(),ctx={console,localStorage:backing,document:{getElementById:()=>null},fetch:async file=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(file,"utf8"))})};ctx.window=ctx;vm.createContext(ctx);
  for(const file of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(file,"utf8"),ctx,{filename:file});
  const api=ctx.__BGS_FINAL56__;await api.loadModels();const h=Array(50).fill("B"),baseline=api.applyFinalPrediction(h);
  const local=ctx.__BBB_BROWSER_CALIBRATION__.createClient(backing),shoeId=backing.getItem("bgs_xgb_final_shoe_id_v5");
  const tag=api.prepareCalibrationTicket(h,baseline).model;
  const prefix=[];for(const result of h){const ticket=local.prepare(prefix,f,shoeId,tag);prefix.push(result);local.commit(ticket,result,prefix);}
  const next=api.applyFinalPrediction(h);
  assert.equal(next.browser_calibration.directional_samples,50);
  assert.equal(next.direction,baseline.direction,"unvalidated calibration must not change formal bet decision");
  assert.equal(next.finalProbability.finalPB,baseline.finalProbability.finalPB,"formal EV probability stays unchanged");
  assert.equal(JSON.stringify(next.finalProbability.extended),JSON.stringify(baseline.finalProbability.extended),"57D features unchanged");
  assert.equal(JSON.stringify(next.raw_next_hand_forecast),JSON.stringify(baseline.raw_next_hand_forecast));
  assert.equal(api.getTrainingCount(),0,"local calibration creates no formal wager training rows");
  console.log("browser calibration: prequential gate, bounds/no flip, ties, reload/undo, isolation, bad-candidate rejection and formal pipeline preservation passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
