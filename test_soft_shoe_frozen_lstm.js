"use strict";
const assert=require("node:assert/strict");
const seq=require("./soft_shoe_frozen_lstm_runtime.js");
const core=require("./soft_shoe_decision_runtime.js");

assert.equal(seq.SCHEMA,"frozen_lstm_bpt_8x8_v1");
assert.equal(seq.INPUT,8);
assert.equal(seq.HIDDEN,8);
assert.equal(seq.WINDOW,24);
const inputs=seq.features(list("BBPTBB"));
assert.equal(inputs.length,6);
assert.deepEqual(inputs[0].slice(0,3),[1,0,0]);
assert.deepEqual(inputs[1].slice(0,5),[1,0,0,2/6,1]);
assert.deepEqual(inputs[2].slice(0,3),[0,1,0]);
assert.deepEqual(inputs[3].slice(0,3),[0,0,1]);
assert.equal(inputs[5][5],4/6);
assert.equal(inputs[5][6],1/6);
assert.equal(inputs[5][7],1/6);
assert.throws(()=>seq.features(["B","?" ]),/invalid_lstm_bpt_history/);
assert.equal(seq.features(Array(30).fill("B")).length,24);
function list(s){return s.split("");}
function zeroArray(n){return Array(n).fill(0);}
const weights={};
for(const g of ["i","f","g","o"]){
 weights["W"+g]=zeroArray(64);
 weights["U"+g]=zeroArray(64);
 weights["b"+g]=zeroArray(8);
}
weights.Wy=zeroArray(24);
weights.by=[0.9,0.1,-1.2];
weights.bg=Array(8).fill(.5);
const mock=Object.freeze({
 schema:seq.SCHEMA,input_dim:8,hidden_dim:8,window:24,
 trained:true,frozen:true,calibrated:true,
 validation:{real_shoe_data:true,out_of_shoe_holdout:true},
 weights
});
assert(seq.validWeights(mock));
let p=seq.predict(list("BBPTB"),mock);
assert.equal(p.available,true);
assert.equal(p.architecture,"frozen_lstm_bpt_8x8_v1");
assert.equal(p.trained,true);
assert(p.probabilities.B>p.probabilities.P);
assert(Math.abs(Object.values(p.probabilities).reduce((a,b)=>a+b,0)-1)<1e-12);
assert.deepEqual(seq.predict(list("BBPTB"),mock),p);
assert.equal(seq.predict(["B"],{...mock,trained:false}).available,false);
assert.equal(seq.predict(["B"],{...mock,validation:{...mock.validation,real_shoe_data:false}}).available,false);
assert.equal(seq.predict(["B"],{...mock,weights:{...weights,Ui:[1]}}).reason,"invalid_lstm_weights");
assert.equal(seq.predict(["B"],{...mock,temperature:0}).reason,"invalid_lstm_temperature");
assert.throws(()=>seq.predict(["X"],mock),/invalid_lstm_bpt_history/);
assert.equal(seq.predict(["B"],null).reason,"no_validated_frozen_lstm");

const fallback=core.frozenCore(list("BBPTB"),seq,null);
assert.equal(fallback.trained,false);
assert.equal(fallback.validated,false);
assert.equal(fallback.reason,"validated_frozen_lstm_unavailable");
const trained=core.frozenCore(list("BBPTB"),seq,mock);
assert.equal(trained.trained,true);
assert.deepEqual(trained.probabilities,p.probabilities);
const history=list("BBPTB");
const onlyPhysical=core.run(history,{SEQUENCE_WEIGHT:.05});
const untrained=core.run(history,{SEQUENCE_WEIGHT:.05},null,seq,null);
assert.equal(untrained.sequence_weight_applied,0);
assert.deepEqual(onlyPhysical.probabilities,untrained.probabilities,
 "missing LSTM weights must not change physics or EV");
assert.deepEqual(onlyPhysical.research_view,untrained.research_view,
 "untrained sequence must not modify live direction gate");
assert.deepEqual(onlyPhysical.features48,untrained.features48);
const withLstm=core.run(history,{SEQUENCE_WEIGHT:.05},null,seq,mock);
assert.equal(withLstm.sequence_weight_applied,.05);
assert.equal(withLstm.frozen_core.architecture,seq.SCHEMA);
assert.deepEqual(withLstm.features48.slice(-3),[
 withLstm.frozen_core.probabilities.B,
 withLstm.frozen_core.probabilities.P,
 withLstm.frozen_core.probabilities.T,
]);
assert.equal(withLstm.feature_schema,"soft_shoe_bpt_48d_v2");
assert.equal(withLstm.feature_dimension,48);
assert.equal(withLstm.action,"Skip","trained mock LSTM alone must NOT authorize bets");
assert.equal(withLstm.bankroll_fraction,0);
assert.equal(withLstm.validated_edge,false);
assert.equal(withLstm.model,"physics_only_untrained_decision_fallback");
assert.equal(core.modelDecision({B:.4,P:.5,T:.1},Array(48).fill(0),trained,
 {schema:"soft_shoe_bpt_48d_v1",feature_dimension:48,trained:true,calibrated:true,
  validation:{real_shoe_data:true,out_of_shoe_holdout:true},predict:()=>({B:.7,P:.2,T:.1})}).model,
 "physics_only_untrained_decision_fallback",
 "old XGBoost 48D feature schema must fail closed");

console.log("PASS Frozen 8x8 LSTM inference, BPT+road features, v2 48D schema and safe untrained fallback");
