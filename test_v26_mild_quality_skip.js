"use strict";
/** Regression for V4 selective SKIP, V3/V2/V1 rollback and immutable physical EV. */
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
const legacyWeak=guard.decide(base,{enabled:false});
const priorWeak=guard.decide(base,{profile:"moderate"});
const v3Weak=guard.decide(base,{profile:"moderate_high"});
assert(weak.threshold>v3Weak.threshold);
assert(v3Weak.threshold>priorWeak.threshold);
assert(priorWeak.threshold>legacyWeak.threshold);
assert(weak.threshold>legacyWeak.threshold);
assert.equal(weak.moderate_entry_enabled,true);
assert.equal(legacyWeak.moderate_entry_enabled,false);
assert.equal(legacyWeak.threshold_addition,0);
assert.equal(guard.MODERATE_ENTRY.pf_quality_threshold,.70);
assert.equal(guard.ACTIVE_PROFILE,"selective_v4");
assert.equal(guard.ENTRY_PROFILES.selective_v4.additional_ev_gap,.00300);
assert.equal(guard.ENTRY_PROFILES.selective_v4.high_uncertainty_extra,.00040);
assert.equal(guard.ENTRY_PROFILES.moderate.additional_ev_gap,.00105);
assert.equal(guard.ENTRY_PROFILES.moderate_high.additional_ev_gap,.00200);
assert.equal(guard.ENTRY_PROFILES.moderate_high.low_quality_extra,.00055);
assert.equal(guard.ENTRY_PROFILES.moderate_high.uncertainty_proxy_threshold,.48);
assert.equal(guard.ENTRY_PROFILES.moderate_high.high_uncertainty_extra,.00035);
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
const rollbackStats={B:0,P:0,SKIP:0,rawB:0,rawP:0,total:0};
const moderateStats={B:0,P:0,SKIP:0,rawB:0,rawP:0,total:0};
const v3Stats={B:0,P:0,SKIP:0,rawB:0,rawP:0,total:0};
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
  const rollback=guard.decide(r,{enabled:false});
  const moderate=guard.decide(r,{profile:"moderate"});
  const v3=guard.decide(r,{profile:"moderate_high"});
  assert(choice.threshold>=v3.threshold);
  assert(v3.direction===choice.direction||choice.direction==="SKIP",
    "V4 must only abstain, not flip or enable new directions");
  assert(choice.threshold>=moderate.threshold);
  assert(moderate.threshold>=rollback.threshold);
  assert(choice.raw_direction===moderate.raw_direction);
  assert(choice.direction===moderate.direction||choice.direction==="SKIP",
    "V3 can only abstain on extra weak signals, never flip or force new bets");
  assert(choice.threshold>=rollback.threshold);
  assert.equal(rollback.threshold_addition,0);
  assert.equal(choice.raw_direction,rollback.raw_direction);
  assert(choice.direction===rollback.direction||choice.direction==="SKIP",
    "Moderate guard may only newly abstain, never enable or flip a direction");
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
  rollbackStats[rollback.direction]++;rollbackStats[rollback.raw_direction==="B"?"rawB":"rawP"]++;rollbackStats.total++;
  moderateStats[moderate.direction]++;moderateStats[moderate.raw_direction==="B"?"rawB":"rawP"]++;moderateStats.total++;
  v3Stats[v3.direction]++;v3Stats[v3.raw_direction==="B"?"rawB":"rawP"]++;v3Stats.total++;
}
assert.equal(stats.total,100);
assert.equal(stats.rawB+stats.rawP,100);
assert(stats.rawB>0&&stats.rawP>0,"Both raw directions must be present");
assert.deepEqual(rollbackStats,{B:48,P:40,SKIP:12,rawB:56,rawP:44,total:100},
  "Legacy V2.6 display gate must remain exactly reversible");
assert.deepEqual(moderateStats,{B:44,P:37,SKIP:19,rawB:56,rawP:44,total:100},
  "Previous moderate V2 profile must remain exactly reversible");
assert.equal(stats.rawB,56);assert.equal(stats.rawP,44);
assert.deepEqual(v3Stats,{B:36,P:34,SKIP:30,rawB:56,rawP:44,total:100},
  "V3 rollback must stay exact");
assert(stats.SKIP>=35&&stats.SKIP<=45,
  "V4 should be visibly more selective without an extreme 50%+ fixed-score SKIP");
const auth=fs.readFileSync("auth-client.js","utf8");
const idx=auth.indexOf("point_signal_quality_guard.js");
const panel=auth.indexOf("quant256_online_panel.js");
assert(idx>0&&panel>idx,"Quality gate must load before panel");
const page=fs.readFileSync("quant256_online_panel.js","utf8");
assert(page.includes("signalGuard.decide(report)"));
assert(page.includes('skip?"觀望"'));
console.log("PASS BBB V2.6 V4 selective SKIP (physical EV intact):",JSON.stringify({v4:stats,v3:v3Stats,v2:moderateStats,legacy:rollbackStats}));
