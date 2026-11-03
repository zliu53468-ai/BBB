"use strict";
/**
 * V2.1 two-sided B/P baseline-offset research guard.
 * Mathematically coupled offset diagnostics, no independent signal claim.
 */
const assert=require("node:assert/strict");
const {
  PhysicsForecastFirstEngine,bidirectionalOffsetGuard,
  BIDIRECTIONAL_OFFSET,BASE,BASE_EV
}=require("./physics_forecast_first_runtime.js");
const near=(x,y,eps=1e-10)=>assert(Math.abs(x-y)<=eps, String(x)+" != "+String(y));
assert.deepEqual({...BIDIRECTIONAL_OFFSET},{
  minConditionalLift:.002,maxSupplementalGate:.0025
});
const baseMass=BASE.B+BASE.P,baseCond=BASE.B/baseMass;
function physicalFromShift(shift,source="hypothetical_particle_posterior"){
  const B=(baseCond+shift)*baseMass,P=(1-baseCond-shift)*baseMass,T=BASE.T;
  return{source,simulated_probability:{B,P,T},
    ev_banker:.95*B-P,ev_player:P-B,simulation_exhausted:false};
}
const baseline=physicalFromShift(0);
const b0=bidirectionalOffsetGuard(baseline);
near(b0.offset_banker,0);near(b0.offset_player,0);
near(b0.supplemental_gate,.0025);
assert(b0.active);
assert(b0.correlated_offsets&&b0.changes_physical_ev===false);
assert(b0.changes_direction===false&&b0.validated_edge===false);
const positive=bidirectionalOffsetGuard(physicalFromShift(.007));
const negative=bidirectionalOffsetGuard(physicalFromShift(-.007));
assert.equal(positive.physical_direction,"B");
assert.equal(negative.physical_direction,"P");
near(positive.offset_banker,.007);near(positive.offset_player,-.007);
near(negative.offset_banker,-.007);near(negative.offset_player,.007);
near(positive.supplemental_gate,0);near(negative.supplemental_gate,0);
for(const shift of [-.007,-.003,-.0005,0,.0005,.003,.007]){
  const x=bidirectionalOffsetGuard(physicalFromShift(shift));
  near(x.offset_banker+x.offset_player,0);
  assert(x.supplemental_gate>=0&&x.supplemental_gate<=.0025);
  near(x.conditional_probability.B+x.conditional_probability.P,1);
}
const smallB=bidirectionalOffsetGuard(physicalFromShift(.0005));
const smallP=bidirectionalOffsetGuard(physicalFromShift(-.0005));
assert.equal(smallB.physical_direction,"B");
assert.equal(smallP.physical_direction,"P");
near(smallB.supplemental_gate,smallP.supplemental_gate,1e-9);
assert(smallB.supplemental_gate>0&&smallB.supplemental_gate<.0025);
// With a 5% banker commission, the physically preferred side can still be B
// even if the outcome-relative-to-baseline offset drifts slightly toward P.
// This is exactly the kind of noisy research signal the guard may penalize.
const conflicting=bidirectionalOffsetGuard(physicalFromShift(-.0001));
assert.equal(conflicting.physical_direction,"B");
near(conflicting.supplemental_gate,.0025);
const observed=bidirectionalOffsetGuard(physicalFromShift(.0005,"observed_card_count"),{observed:true});
assert(!observed.active);
near(observed.supplemental_gate,0);
assert.equal(observed.source,"observed_cards");
assert(!bidirectionalOffsetGuard({...baseline,source:"unobserved_baseline"}).active);
near(bidirectionalOffsetGuard({...baseline,source:"unobserved_baseline"}).supplemental_gate,0);
const unavailable=bidirectionalOffsetGuard({...baseline,simulation_exhausted:true});
assert(!unavailable.active);near(unavailable.supplemental_gate,0);
const impossible=bidirectionalOffsetGuard({
  simulated_probability:{B:0,P:0,T:1},ev_banker:-1,ev_player:-1
});
assert(!impossible.active);near(impossible.supplemental_gate,0);
const e=new PhysicsForecastFirstEngine();
const first=e.predict();
assert.equal(first.bidirectional_offset.active,false);
near(first.effective_research_gate,.005);
assert.deepEqual(first,e.predict(),"cached empty prediction must remain deterministic");
const seq="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const report=e.syncHistory(seq);
assert.equal(report.physical_ev.samples,256*160);
assert.equal(report.action,"SKIP","B/P/T-only is not real-card BET");
assert.equal(report.bet_allowed,false);
assert.deepEqual(report.bidirectional_offset,report.research_signal.bidirectional_offset);
near(report.bidirectional_offset.offset_banker+
  report.bidirectional_offset.offset_player,0);
near(report.ev_banker,report.physical_ev.ev_banker);
near(report.ev_player,report.physical_ev.ev_player);
assert.equal(report.research_signal.direction,
  report.ev_banker>=report.ev_player?"B":"P",
  "no reversal of original physical direction");
near(report.effective_research_gate,
  report.dynamic_gate+
  (report.early_shoe_guard?.supplemental_gate||0)+
  report.shoe_distribution_guard.additional_gate+
  report.bidirectional_offset.supplemental_gate);
assert.equal(report.feature_weights.physics,5);
assert.equal(report.feature_weights.road,.1);
assert.deepEqual(report,e.predict(),"read-only prediction is stable");
assert.deepEqual(report,new PhysicsForecastFirstEngine().syncHistory(seq),
  "full model replay must stay deterministic");
e.syncHistory(seq.slice(0,-1));assert.deepEqual(report,e.syncHistory(seq),
  "undo/replay must retain exact output");
const real=new PhysicsForecastFirstEngine();
assert.equal(real.observedSamples,12000);
assert.equal(real.simulationsPerParticle,160);
near(BASE_EV.B,.95*BASE.B-BASE.P);
console.log("PASS two-sided B/P offset: symmetric baseline lifts, capped research-only gate, observed bypass, physical authority, replay and original N/L/M");
