"use strict";
// 還原 V3 原預測完全一致；本金操作不得回寫模型或方向。
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const {PhysicsForecastFirstEngine,VERSION}=require("./physics_forecast_first_runtime");
const {calculateStakeExposure}=require("./quant256_bankroll_ui");
assert.equal(VERSION,"BBB_PHYSICS_FORECAST_FIRST_V3_MICRO_FUSION");
const seq="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const baseline=new PhysicsForecastFirstEngine({threshold:.010,lossGate:.035}).syncHistory(seq);
const plain=JSON.stringify({direction:baseline.research_signal.direction,
  action:baseline.action,score:baseline.research_signal.score,
  gate:baseline.dynamic_gate,ev:baseline.ev_gap_deviation,
  winStreak:baseline.win_streak});
assert.equal(JSON.stringify({
  direction:new PhysicsForecastFirstEngine({threshold:.010,lossGate:.035}).syncHistory(seq).research_signal.direction,
  action:baseline.action,score:baseline.research_signal.score,
  gate:baseline.dynamic_gate,ev:baseline.ev_gap_deviation,
  winStreak:baseline.win_streak
}),plain);
// 本金與滑桿為純函數；只通過時顯示非零參考金額。
assert.equal(calculateStakeExposure({bankroll:5000,percent:20,confidenceWeight:.5,signal:false}).amount,0);
assert.equal(calculateStakeExposure({bankroll:5000,percent:20,confidenceWeight:.5,signal:true}).amount,500);
assert.equal(calculateStakeExposure({bankroll:50000,percent:50,confidenceWeight:1,signal:true}).amount,25000);
assert.throws(()=>calculateStakeExposure({bankroll:15000}),/invalid_bankroll_preset/);
assert.throws(()=>calculateStakeExposure({percent:51}),/invalid_stake_percent/);
const ids=["bankrollSelect","stakePercent","bankrollCurrent","stakePercentValue",
  "stakeConfidenceWeight","stakeRecommendation","stakeLocalStatus","stakeStatus"];
function mock(){
  const listeners={};
  const dom=Object.fromEntries(ids.map(id=>[id,{value:"",textContent:"",
    addEventListener(type,fn){(listeners[id]??={})[type]=fn;}}]));
  const stored=new Map();
  const window={document:{getElementById:id=>dom[id]||null},localStorage:{
    getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)}};
  vm.runInNewContext(fs.readFileSync("./quant256_bankroll_ui.js","utf8"),{window,console});
  return{window,dom,listeners,stored};
}
const {window,dom,listeners,stored}=mock();
const ui=window.__BBB_V3_CAPITAL_UI__;
assert.equal(ui.VERSION,"BBB_V3_CAPITAL_UI_V1");
assert.equal(ui.capital,5000);
assert.equal(ui.stake_percent,20);
assert.equal(dom.stakeRecommendation.textContent,"NT$0");
const result=ui.renderStake(baseline);
assert.equal(result.signal,baseline.research_signal.available);
assert.equal(dom.stakeLocalStatus.textContent,"V3 已分析 "+seq.length+" 局");
assert(["NT$0","NT$500","NT$501"].includes(dom.stakeRecommendation.textContent)||
  /NT\$[0-9,]+/.test(dom.stakeRecommendation.textContent));
const priorAmount=result.amount;
dom.bankrollSelect.value="30000";listeners.bankrollSelect.change();
assert.equal(ui.capital,30000);
assert.equal(JSON.parse(stored.get("bbb_stake_exposure_panel_v1")).capital,30000);
dom.stakePercent.value="35";listeners.stakePercent.input();
assert.equal(ui.stake_percent,35);
assert.equal(dom.bankrollCurrent.textContent,"NT$30,000");
assert.equal(dom.stakePercentValue.textContent,"35%");
assert(ui.renderStake(baseline).amount>=priorAmount);
ui.renderStake({...baseline,research_signal:{...baseline.research_signal,available:false}});
assert.equal(dom.stakeRecommendation.textContent,"NT$0");
assert(dom.stakeStatus.textContent.includes("觀望"));
const after=new PhysicsForecastFirstEngine({threshold:.010,lossGate:.035}).syncHistory(seq);
assert.equal(after.research_signal.direction,baseline.research_signal.direction);
assert.equal(after.dynamic_gate,baseline.dynamic_gate);
assert.equal(after.ev_gap_deviation,baseline.ev_gap_deviation);
assert(!fs.readFileSync("./auth-client.js","utf8").includes("quant256_local_calibration.js"),
  "actual production must not load V4 local 12 model");
assert(!fs.readFileSync("./physics_forecast_first_runtime.js","utf8").includes("this.local"),
  "V3 must not instantiate local 12 model");
console.log("PASS: V3 engine only, 0.010/0/0.035, UI bankroll controls isolated, skipped exposure $0, no local A/b upgrade");
