"use strict";
const assert=require("node:assert/strict");
const Cal=require("./calibrate_macro_direction.js");
const Macro=require("./macro_ema.js");
const hands=Cal.synthetic(90,20261009);
const result=Cal.report(hands,"synthetic_mechanics_test");
assert.equal(result.shoes,90);
assert(hands.length>4500);
assert.equal(result.timing.startsWith("snapshot prior"),true);
assert.deepEqual(Object.keys(result.analysis),["six_card","low_score","point_diff"]);
for(const k of ["six_card","low_score","point_diff"]){
  const a=result.analysis[k];
  for(const split of ["train","calibration","holdout"]){
    assert(a[split].observations>100);
    assert(a[split].shoes>0);
    assert(Number.isFinite(a[split].pearson_r));
  }
  assert([0,.01,-.01,.02,-.02].includes(result.candidate_coefficients[k]));
}
assert.deepEqual(result.candidate_coefficients,{six_card:0,low_score:0,point_diff:0});
assert.equal(result.production_activation_recommended,false);
const first=hands[0];
assert.equal(first.round_index,1);
const pre=Macro.createState("causal");
assert.equal(Macro.snapshot(pre).Six_Card_Spread,null);
Macro.update(pre,first);
assert.equal(Macro.snapshot(pre).rounds,1);
assert.throws(()=>Cal.report([{...first,round_index:2}],"invalid"));
const modified=hands.map(x=>({...x}));
modified[2]={...modified[2],round_index:9};
assert.throws(()=>Cal.report(modified,"invalid"),/consecutive/);
console.log(JSON.stringify({ok:true,shoes:result.shoes,hands:hands.length,coefficient_units:"probability_delta",
 checks:["deterministic_8_deck","causal_prehand","shoe_disjoint","bounded_candidates","no_auto_activation","strict_sequence"]}));
