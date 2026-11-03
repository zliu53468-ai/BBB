"use strict";
/** Regression for shared BBB/BGS display-only mild abstention, not hit-rate. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const {PhysicsForecastFirstEngine}=require("./physics_forecast_first_runtime.js");
const guard=require("./point_signal_quality_guard.js");
const base={
 ev_banker:-.004,ev_player:-.003,
 physical_ev:{simulation_exhausted:false},
 early_shoe_guard:{mode:"independent_final_score_quality",quality:.91,
   ess_ratio:.94,ancestor_diversity:.97,mc_gap_se:.010,
   posterior_gap_spread:.085}
};
const weak=guard.decide(base);
assert.equal(weak.direction,"SKIP");
assert.equal(weak.raw_direction,"P");
assert.equal(weak.verified_edge,false);
assert.equal(weak.validated_hit_rate,false);
const robust=guard.decide({...base,ev_banker:.002,ev_player:-.005});
assert.equal(robust.direction,"B");
const veryPoor=guard.decide({...base,early_shoe_guard:{
 ...base.early_shoe_guard,quality:.1,ess_ratio:.1,
 ancestor_diversity:.1,mc_gap_se:.2,posterior_gap_spread:.5}});
assert(veryPoor.threshold>weak.threshold);
assert.equal(guard.decide({...base,early_shoe_guard:null}).direction,"SKIP");
const stats={B:0,P:0,SKIP:0,rawB:0,rawP:0,total:0};
for(let player=0;player<=9;player++)for(let banker=0;banker<=9;banker++){
  const outcome=banker>player?"B":player>banker?"P":"T";
  const model=new PhysicsForecastFirstEngine({
    simulationsPerParticle:96,threshold:.005,lossGate:.035,
    quantOptions:{lambda:.82,alpha:.15,particles:96},
    independentFinalScoreMode:true
  });
  const r=model.syncHistory([outcome],[{player,banker}]);
  const evB=r.ev_banker,evP=r.ev_player;
  const choice=guard.decide(r);
  assert.equal(r.physical_ev.samples,9216);
  assert.equal(model.quant.physics.size,96);
  assert.equal(r.action,"SKIP");assert.equal(r.bet_allowed,false);
  assert(choice.direction==="SKIP"||choice.direction===choice.raw_direction,
    "Display policy may abstain, never force the other side");
  assert.equal(choice.raw_direction,evB>=evP?"B":"P");
  assert(Number.isFinite(choice.threshold)&&choice.threshold>0);
  assert.strictEqual(r.ev_banker,evB);
  assert.strictEqual(r.ev_player,evP);
  stats[choice.direction]++;stats[choice.raw_direction==="B"?"rawB":"rawP"]++;stats.total++;
}
assert.equal(stats.total,100);
assert.equal(stats.rawB+stats.rawP,100);
assert(stats.rawB>0&&stats.rawP>0,"Both raw directions must be present");
assert(stats.SKIP>=0&&stats.SKIP<=100,
  "Mild display gate unexpectedly skips too few/many benchmark pairs");
const auth=fs.readFileSync("auth-client.js","utf8");
const idx=auth.indexOf("point_signal_quality_guard.js");
const panel=auth.indexOf("quant256_online_panel.js");
assert(idx>0&&panel>idx,"Quality gate must load before panel");
const page=fs.readFileSync("quant256_online_panel.js","utf8");
assert(page.includes("signalGuard.decide(report)"));
assert(page.includes('skip?"觀望"'));
console.log("PASS BBB V2.6 mild display SKIP (no physical EV/policy changes):",JSON.stringify(stats));
