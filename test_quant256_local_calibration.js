"use strict";
const assert=require("node:assert/strict");
const {WindowLocalLinUCB,project,calculateStakeExposure,ACTIVE_INDICES,VERSION}=
  require("./quant256_local_calibration");
assert.equal(VERSION,"BBB_LOCAL_256D_WINDOW12_V1");
assert.equal(ACTIVE_INDICES.length,42);
const zeros=()=>Array(256).fill(0);
const staticScores={ucb_banker:.12,ucb_player:-.09,
  arms:{B:{uncertainty:.3},P:{uncertainty:.4}}};
const model=new WindowLocalLinUCB({windowSize:12,maxBlend:.35});
let initial=model.predict(zeros(),staticScores);
assert.equal(initial.sample_count,0);
assert.equal(initial.blend_weight,0);
assert.equal(initial.recorded_rounds,0);
assert.equal(initial.blended_ucb.B,.12);
assert.equal(initial.blended_ucb.P,-.09);
assert.equal(initial.calibrated_probability,false);
const x=zeros();x[128]=.4;x[152]=.9;x[153]=-.6;
let projected=project(x);
assert(projected.some(v=>v!==0));
assert.equal(projected.length,42);
for(let i=0;i<12;i++){
  const y=x.slice();y[128]=(i%2===0?.4:-.4);y[152]=(i%3)/3;
  model.observe(y,i%5===0?"T":i%2===0?"B":"P");
}
const data=model.predict(x,staticScores);
assert.equal(data.recorded_rounds,12);
assert.equal(data.sample_count,9);
assert.equal(data.tied_rounds,3);
assert.equal(data.blend_weight,.35);
assert(data.A.length===42&&data.A.every(row=>row.length===42));
for(let i=0;i<42;i++){
  assert(Math.abs(data.b.B[i]+data.b.P[i])<1e-12);
  assert(Number.isFinite(data.A[i][i]));
}
assert(data.A[12][36]!==0,"full A must include road × interaction offdiagonal");
assert(Number.isFinite(data.gap));
assert(data.uncertainty_ratio>=0&&data.uncertainty_ratio<=1);
// 第 13 局加入後，第一筆自動從局部窗口消失；不保留全部歷史。
const baseline=JSON.stringify(data.b);
model.observe(x,"P");
const shifted=model.predict(x,staticScores);
assert.equal(shifted.recorded_rounds,12);
assert.equal(model.totalResults,13);
assert.notEqual(JSON.stringify(shifted.b),baseline);
// 回放等價：可藉 reset 從歷史重新建構一模一樣的局部 A/b。
const replay=new WindowLocalLinUCB();
for(const e of model.observations)replay.observe(Array.from({length:256},(_,i)=>{
  const pos=ACTIVE_INDICES.indexOf(i);
  return pos<0?0:e.x[pos]*Math.sqrt(42);
}),e.outcome);
const replayed=replay.predict(x,staticScores);
for(let i=0;i<42;i++){
  assert(Math.abs(shifted.b.B[i]-replayed.b.B[i])<1e-10);
  for(let j=0;j<42;j++)
    assert(Math.abs(shifted.A[i][j]-replayed.A[i][j])<1e-10);
}
assert.throws(()=>model.observe(x,"X"),/invalid_outcome/);
assert.throws(()=>project(Array(42).fill(0)),/invalid_256D_feature/);
assert.throws(()=>new WindowLocalLinUCB({windowSize:2}),/invalid_window_size/);
// 滑桿：本金 5,000～50,000，10～50% 比例；不通過訊號時金額必為零。
const none=calculateStakeExposure({bankroll:5000,percent:20,confidenceWeight:.5,signal:false});
assert.equal(none.amount,0);
const normal=calculateStakeExposure({bankroll:5000,percent:20,confidenceWeight:.5,signal:true});
assert.equal(normal.amount,500);
assert.equal(normal.exposureRatio,.1);
const high=calculateStakeExposure({bankroll:50000,percent:50,confidenceWeight:1,signal:true});
assert.equal(high.amount,25000);
assert.equal(high.exposureRatio,.5);
const floor=calculateStakeExposure({bankroll:10000,percent:10,confidenceWeight:.5,signal:true});
assert.equal(floor.amount,1000);
assert.equal(floor.is_real_wager_advice,false);
assert.throws(()=>calculateStakeExposure({bankroll:12000}),/invalid_bankroll_preset/);
assert.throws(()=>calculateStakeExposure({percent:51}),/invalid_stake_percent/);
assert.throws(()=>calculateStakeExposure({confidenceWeight:1.2}),/invalid_confidence_weight/);
console.log("PASS local12: 42D full SPD A/b, prequential 12-round truncation, static/local blend, exact replay, bankroll and stake formula");
