"use strict";
const assert=require("node:assert/strict");
const {assess}=require("./tools/no_pf_acceptance.js");
const m={bets:1000,brier:.24,hit_rate_on_bets:.56,realized_ev_per_bet:.04,skip_rate:.5};
function fixture(delta){
  const metrics=Object.fromEntries(["all","early","middle","late"].map(stage=>[stage,{...m}]));
  const r={data_kind:"real_holdout_observations",shoes:100,current:{no_flip_violations:0,metrics:structuredClone(metrics)},baseline:{metrics}};
  for(const stage of Object.keys(metrics))r.current.metrics[stage].skip_rate+=delta;
  return r;
}
assert(assess(fixture(-.1)).passed,"Skip decreases must be allowed");
assert(assess(fixture(.05)).passed,"up to +5pp is allowed");
assert(!assess(fixture(.051)).passed,"more than +5pp must fail");
const poorPattern=fixture(0);poorPattern.current.pattern_segments_report_only={long_run:{hit_rate_on_bets:0}};
assert(assess(poorPattern).passed,"pattern reports must not gate promotion");
for(const metric of ["hit_rate_on_bets","realized_ev_per_bet"]){
  const r=fixture(-.1);r.current.metrics.all[metric]-=.01;
  assert(!assess(r).passed,"more bets cannot compensate for quality regression");
}
const badBrier=fixture(0);badBrier.current.metrics.all.brier+=.002;
assert(!assess(badBrier).passed);
const synthetic=fixture(0);synthetic.data_kind="synthetic";
assert(!assess(synthetic).passed);
console.log("counting acceptance: Skip ceiling, quality guards, report-only patterns passed");
