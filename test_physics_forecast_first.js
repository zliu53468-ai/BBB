"use strict";
/** 真實物理證據與盲估資料源必須嚴格分流。 */
const assert=require("node:assert/strict");
const {
  PhysicsForecastFirstEngine,observedCounts,observedHand,
  deal,simulate,decisionFromEV,weightedContext,BASE,BASE_EV,INITIAL,VERSION
}=require("./physics_forecast_first_runtime.js");
const near=(x,y,eps=1e-8)=>assert(Math.abs(x-y)<=eps,String(x)+" != "+String(y));
assert.equal(VERSION,"BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE");
assert.equal(INITIAL.length,10);
assert.equal(INITIAL[0],128);
assert(INITIAL.slice(1).every(x=>x===32));
assert.equal(INITIAL.reduce((a,b)=>a+b),416);
near(BASE.B,.458597);near(BASE.P,.446247);near(BASE.T,.095156);
assert(BASE_EV.B<0&&BASE_EV.P<0,"standard 8 deck bets have negative base EV");
const engine=new PhysicsForecastFirstEngine();
const initial=engine.predict();
assert.equal(initial.action,"SKIP");
assert.equal(initial.direction,"Skip");
assert.equal(initial.observed_card_evidence,false);
assert.equal(initial.reason,"dynamic_signal_filter");
assert.equal(initial.dynamic_gate,.005);
assert.equal(initial.research_signal.direction,"B");
assert.equal(initial.research_signal.action,"SKIP");
assert.equal(initial.research_signal.source,"simulated_bpt_only");
assert.equal(initial.physical_ev.ev_gap_deviation,0);
assert.equal(initial.context.length,256);
assert.equal(initial.shoe_context.length,128);
assert.equal(initial.road_context.length,128);
assert.equal(initial.feature_weights.physics,5);
assert.equal(initial.feature_weights.road,.1);
assert(initial.road_context.every(x=>Math.abs(x)<=.1));
assert.equal(initial.road_has_direction_authority,false);
assert.equal(initial.proxy_only,true);
assert.equal(initial.validated_edge,false);
const seq="BPPBTBBPPTBBPPBTBPBPPBP".split("");
const bpt=engine.syncHistory(seq);
assert.equal(bpt.action,"SKIP","unobserved ranks must never be labelled BET");
assert(["unobserved_card_composition","dynamic_signal_filter"].includes(bpt.reason));
assert(["B","P"].includes(bpt.research_signal.direction));
assert(["SIGNAL","SKIP"].includes(bpt.research_signal.action));
assert([0,.005,.035].includes(bpt.dynamic_gate));
assert.equal(bpt.research_signal.evidence_qualified,false);
assert.equal(bpt.card_count_source,"hypothetical_particle_posterior");
assert(bpt.physical_ev.samples>=1000);
assert(bpt.physical_ev.particle_support>=8);
assert(bpt.physical_ev.remaining<=416);
assert(Number.isFinite(bpt.ev_banker));
assert(Number.isFinite(bpt.ev_player));
assert.equal(bpt.context.length,256);
assert.equal(bpt.shoe_context.length,128);
assert.equal(bpt.road_context.length,128);
assert(bpt.road_context.every(x=>Math.abs(x)<=.1));
assert(bpt.model_road_debug.direction==="B"||
  bpt.model_road_debug.direction==="P"||
  bpt.model_road_debug.direction==="Skip");
assert.equal(bpt.bet_allowed,false);
const saved=e=>JSON.stringify({
  action:e.action,ev_banker:e.ev_banker,ev_player:e.ev_player,
  ev_gap_deviation:e.ev_gap_deviation,context:e.context
});
assert.equal(saved(bpt),saved(engine.predict()),"read-only forecast must be deterministic");
assert.equal(saved(bpt),saved(new PhysicsForecastFirstEngine().syncHistory(seq)),
  "refresh replay must be deterministic");
engine.syncHistory(seq.slice(0,-1));
assert.equal(engine.history.length,seq.length-1);
engine.syncHistory(seq);
assert.equal(saved(engine.predict()),saved(bpt),"undo/replay must recreate prior output");
assert.throws(()=>engine.syncHistory(["X"]),/invalid_history/);
const drawResult=simulate(INITIAL,{samples:8192,seed:0xF27});
assert(Math.abs(drawResult.win_probabilities.B-BASE.B)<.025,"MC banker vs exact standard odds");
assert(Math.abs(drawResult.win_probabilities.P-BASE.P)<.025,"MC player vs exact standard odds");
assert(Math.abs(drawResult.win_probabilities.T-BASE.T)<.015,"MC tie vs exact standard odds");
assert.equal(drawResult.win_probabilities.B+
  drawResult.win_probabilities.P+drawResult.win_probabilities.T,1);
near(drawResult.ev_banker,.95*drawResult.win_probabilities.B-drawResult.win_probabilities.P);
near(drawResult.ev_player,drawResult.win_probabilities.P-drawResult.win_probabilities.B);
const theoreticalDeviation=(drawResult.ev_banker-drawResult.ev_player)-
  (BASE_EV.B-BASE_EV.P);
near(drawResult.ev_gap_deviation,theoreticalDeviation);
assert.throws(()=>simulate(Array(10).fill(0)),/remaining_shoe_not_playable/);
assert.throws(()=>simulate(INITIAL,{samples:10}),/invalid_mc_sample_count/);
assert.throws(()=>simulate([128,...Array(9).fill(31.3)]),/invalid_point_counts/);
// 4,5,6 張牌的合法出牌必須可驗證。順序是 P1/B1/P2/B2/P3?/B3?。
assert.equal(observedHand([9,8,9,8]),"P");
assert.equal(observedHand([7,9,7,9]),"B");
assert.equal(observedHand([9,9,9,9]),"T");
assert.throws(()=>observedHand([0,0,0,0]),/observed_hand_draw_rule_mismatch/);
assert.throws(()=>observedHand([9,8,9,8,0]),/observed_hand_draw_rule_mismatch/);
const nextPseudo=(()=>{let n=0x13879;return()=>{n^=n<<13;n^=n>>>17;n^=n<<5;return(n>>>0)/4294967296;};})();
const records=[];
const counts=INITIAL.slice();
for(let i=0;i<8;i++){
  const h=deal(counts,nextPseudo);
  assert.equal(observedHand(h.cards),h.outcome);
  records.push({outcome:h.outcome,observation:{cards:h.cards}});
  for(const card of h.cards)counts[card]--;
}
const recorded=observedCounts(records,records.map(r=>r.outcome));
assert.deepEqual(recorded,counts,"observed ranks must be removed exactly");
assert.throws(()=>observedCounts(records,["B"]),/cover_entire_shoe/);
assert.throws(()=>observedCounts(
  [{outcome:"P",observation:{cards:[7,9,7,9]}}],["P"]),/observed_result_mismatch/);
const real=new PhysicsForecastFirstEngine({observedSamples:2400});
real.syncHistory(records.map(r=>r.outcome));
real.setObservedRounds(records);
const actual=real.predict();
assert.equal(actual.observed_card_evidence,true);
assert.equal(actual.card_count_source,"observed_card_count");
assert.equal(actual.proxy_only,false);
assert.equal(actual.physical_ev.remaining,416-records.reduce((s,r)=>s+r.observation.cards.length,0));
assert.equal(actual.physical_ev.samples,2400);
assert.equal(actual.card_count_gate.evidence_source,"supplied_observed_cards");
assert(actual.action==="SKIP"||actual.action==="BET");
assert.equal(actual.action==="BET",actual.card_count_gate.bet_allowed);
real.setObservedRounds(null);
assert.equal(real.predict().action,"SKIP","remove observed cards immediately disables BET");
assert(["unobserved_card_composition","dynamic_signal_filter"].includes(real.predict().reason));
assert.throws(()=>real.setObservedRounds([{outcome:"B"}]),/invalid_observed_rows/);
const strong={
  ev_banker:.13,ev_player:-.13,ev_gap_deviation:.18,se_gap:.005,
  se_banker:.004,se_player:.004
};
assert.equal(decisionFromEV(strong,{observed:false}).action,"SKIP",
  "even a huge virtual MC EV cannot claim real edge");
assert.equal(decisionFromEV(strong,{observed:true}).action,"BET",
  "observed composition + meaningful positive lower CI can pass");
assert.equal(decisionFromEV({...strong,ev_gap_deviation:.01},{observed:true}).action,"SKIP");
assert.equal(decisionFromEV({...strong,ev_banker:-.02,ev_player:-.09},
  {observed:true}).action,"SKIP","no positive EV cannot become BET");
assert.equal(decisionFromEV({...strong,se_gap:.15},{observed:true}).action,"SKIP",
  "unreliable MC cannot create a physical advantage");
assert.throws(()=>new PhysicsForecastFirstEngine({threshold:-1}),/invalid_ev_gate/);
real.reset();
assert.equal(real.history.length,0);
assert.equal(real.predict().action,"SKIP");
const data=weightedContext(bpt.model_road_debug,bpt.physical_ev);
assert.equal(data.vector.length,256);
assert.equal(data.physics_weight,5);
assert.equal(data.road_weight,.1);
assert.equal(data.road_has_direction_authority,false);
// 動態順風車：僅在上一局研究訊號有方向、且實際結算 B/P 時更新。
// 重複 predict() 讀取不得增加連勝或重新消耗緩衝。
const dynamic=new PhysicsForecastFirstEngine({threshold:.005,lossGate:.035});
const success={research_signal:{available:true,direction:"B"}};
const filtered={research_signal:{available:false,direction:"P"}};
assert.equal(dynamic.predict().dynamic_gate,.005);
dynamic._settle(success,"B");
assert.equal(dynamic.phase,"overdrive");
assert.equal(dynamic.winStreak,1);
assert.equal(dynamic.predict().dynamic_gate,0);
dynamic._settle(success,"B");
assert.equal(dynamic.winStreak,2);
assert.equal(dynamic.predict().dynamic_gate,0);
dynamic._settle(success,"P");
assert.equal(dynamic.phase,"cushion");
assert.equal(dynamic.winStreak,0);
assert.equal(dynamic.predict().dynamic_gate,.035);
dynamic._settle(filtered,"T");
assert.equal(dynamic.predict().dynamic_gate,.035,"Tie must not consume one-hand cushion");
dynamic._settle(filtered,"P");
assert.equal(dynamic.phase,"base");
assert.equal(dynamic.predict().dynamic_gate,.005,"one missed opportunity reverts to base");
dynamic._settle(success,"B");
assert.equal(dynamic.predict().dynamic_gate,0,"a correct research signal enables overdrive");
dynamic._settle(success,"P");
assert.equal(dynamic.predict().dynamic_gate,.035,"another wrong research signal reactivates cushion");
const before=dynamic.winStreak;
dynamic.predict();dynamic.predict();
assert.equal(dynamic.winStreak,before,"read-only predictions must not settle outcomes");
const snapshot=new PhysicsForecastFirstEngine();
const output=snapshot.syncHistory(seq);
const beforeSettlements={...snapshot.settlements};
snapshot.syncHistory(seq);
assert.deepEqual(snapshot.settlements,beforeSettlements,
  "repeated run must not settle the same historical rounds twice");
const snapshotReset=new PhysicsForecastFirstEngine();
snapshotReset.syncHistory(seq.slice(0,-1));
snapshotReset.syncHistory(seq);
assert.equal(snapshotReset.gatePhase,undefined);
assert.deepEqual(snapshotReset.settlements,beforeSettlements,"incremental replay must be identical");
assert.equal(snapshotReset.predict().dynamic_gate,output.dynamic_gate);
assert.equal(snapshotReset.predict().research_signal.direction,output.research_signal.direction);
console.log("PASS physics-first V2 dynamic 0.005/0/0.035: rules, shadow/real EV separation, research direction, prequential replay and one-round recovery");