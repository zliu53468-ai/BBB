"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm");
(async()=>{
  const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:()=>null,setItem:()=>{},removeItem:()=>{}},
    fetch:async path=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path,"utf8"))})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const path of ["app256forward.js","app256continuation.js","particle_filter_runtime.js",
                     "macro_ema.js","browser_calibration.js","final_probability_runtime.js"])
    vm.runInContext(fs.readFileSync(path,"utf8"),ctx,{filename:path});
  const api=ctx.__BGS_FINAL56__;await api.loadModels();
  const p=api.applyFinalPrediction("BPBPPTBBBBPPPTBPBP".split(""));
  const d=p.direction_bias_audit;
  assert(d&&d.mode==="read_only");
  assert.equal(d.observation_basis,"BPT_only");
  assert.equal(p.finalProbability.direction_bias_audit,d);
  assert.equal(d.bet_direction,p.direction);
  assert.equal(d.direct_physics_side,p.finalProbability.physicsDirect[23]>p.finalProbability.physicsDirect[24]?"B":"P");
  assert.equal(d.pf_flip_blocked,p.finalProbability.particleDiagnostics?.direction_flip_blocked===true);
  assert(d.pf_weight===null||d.pf_weight>=0&&d.pf_weight<=1);
  // 不能偷將偏好當成已授權下注：negative EV 仍須受原 guard。
  const data=api.selectEvDirection(-.03,-.06);
  assert.equal(data.direction,"B");
  assert(data.best_ev<0);
  const nodes=new Map();
  for(const id of ["directionOrb","directionText","directionCode","confidence","regime",
                   "strength","modePill","roundCount","message"])nodes.set(id,{textContent:"",className:""});
  ctx.document.getElementById=id=>nodes.get(id)||null;
  const forecast=api.evDirectionForecast(api.settledOutcomeEv({B:.4586,P:.4462,T:.0952}),.3);
  api.renderPrediction({next_hand_forecast:forecast,direction:"Skip",direction_bias_audit:{
     ev_comparison:{both_negative:true},pf_flip_blocked:true}},8);
  assert.equal(nodes.get("directionCode").textContent,"SKIP");
  assert.match(nodes.get("regime").textContent,/Direct Physics觀望/);
  // B/P/T mode keeps the current legacy label; diagnostics stay read-only for offline audit.
  console.log(JSON.stringify({passed:true,side_audit:true,bet_unchanged:true,protected_ui_unchanged:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
