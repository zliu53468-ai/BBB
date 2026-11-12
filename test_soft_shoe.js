"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const s=require("./soft_shoe_decision_runtime.js");

const roadAux=require("./derived_road_aux.js");
const roadDefaults={
 ROAD_AUX_ENABLED:true,ROAD_AUX_MIN_HISTORY:12,
 ROAD_AUX_AGREEMENT_MIN:.67,ROAD_AUX_CONFIDENCE_MIN:.45,
 ROAD_AUX_MAX_BOOST:.08,ROAD_AUX_CONFLICT_PENALTY:.06,
 ROAD_AUX_REQUIRE_SOFT_ELIGIBLE:true
};
for(const [key,value] of Object.entries(roadDefaults))
 assert.equal(s.DEFAULT[key],value,"road auxiliary default "+key);
assert.throws(()=>s.config({ROAD_AUX_MAX_BOOST:.11}),/invalid_ROAD_AUX_MAX_BOOST/);
assert.throws(()=>s.config({ROAD_AUX_MIN_HISTORY:0}),/invalid_ROAD_AUX_MIN_HISTORY/);
assert.throws(()=>s.config({ROAD_AUX_AGREEMENT_MIN:1.01}),/invalid_ROAD_AUX_AGREEMENT_MIN/);
assert.throws(()=>s.config({ROAD_AUX_CONFIDENCE_MIN:-.01}),/invalid_ROAD_AUX_CONFIDENCE_MIN/);
assert.throws(()=>s.config({ROAD_AUX_CONFLICT_PENALTY:1.01}),/invalid_ROAD_AUX_CONFLICT_PENALTY/);
assert.throws(()=>s.config({ROAD_AUX_ENABLED:"false"}),/invalid_ROAD_AUX_ENABLED/);
assert.throws(()=>s.config({ROAD_AUX_REQUIRE_SOFT_ELIGIBLE:0}),/invalid_ROAD_AUX_REQUIRE_SOFT_ELIGIBLE/);
// Derived roads are deterministic pattern bookkeeping. T never creates a bead.
const roadShort=roadAux.analyze(Array(11).fill("B"));
assert.equal(roadShort.available,false);
assert.equal(roadShort.reason,"insufficient_non_tie_history");
assert.equal(roadShort.big_road_depth,11);
const roadTies=roadAux.analyze(Array(12).fill("T"));
assert.equal(roadTies.big_road_depth,0);
assert.deepEqual(roadTies.probes,{big_eye:"N",small:"N",cockroach:"N"});
const roadSingleColumn=roadAux.analyze(Array(12).fill("B"));
assert.equal(roadSingleColumn.available,false);
assert.equal(roadSingleColumn.reason,"insufficient_derived_road_columns");
const roadAlternating=roadAux.analyze(Array.from({length:12},(_,i)=>i%2?"P":"B"));
assert.equal(roadAlternating.big_road_columns,12);
assert.equal(roadAlternating.road_agreement,1);
assert.deepEqual(roadAlternating.probes,{big_eye:"R",small:"R",cockroach:"R"});
assert.equal(roadAlternating.road_side,"P","majority R maps to last side (research heuristic)");
assert.equal(roadAlternating.estimated_only,true);
assert.equal(roadAlternating.validated_betting_edge,false);
const withTies=["B","T","P","T","B","T","P","T","B","T","P","T",
 "B","T","P","T","B","T","P","T","B","T","P","T"];
assert.deepEqual(roadAux.analyze(withTies),roadAlternating,
 "ties are ignored in the derived-road structure");
const cols=[{depth:2},{depth:2},{depth:1},{depth:3},{depth:1}];
assert.equal(roadAux.deriveColor(cols,2,0,1),"R");
assert.equal(roadAux.deriveColor([{depth:2},{depth:1},{depth:1}],2,0,1),"B");
assert.equal(roadAux.deriveColor(cols,1,2,1),"B",
 "one missing left cell but occupied above produces blue");
assert.equal(roadAux.deriveColor(cols,1,3,1),"R",
 "two missing cells to the left produce red");
assert.equal(roadAux.mapRoadSide("B",0,3),"P");
assert.equal(roadAux.mapRoadSide("P",2,2),"NEUTRAL");
// Isolated display-score fusion: never grants betting or flips Soft candidate.
const syntheticResearch={eligible:true,direction:"B",candidate:"B",ev_gap_snr:1.4,
 estimated_only:true,validated_betting_edge:false};
const syntheticPath={available:true,future_edge_score:.50,
 estimated_only:true,validated_betting_edge:false};
const makeRoad=side=>({...roadAlternating,available:true,road_side:side,
 road_agreement:1,road_confidence:.8,estimated_only:true,
 validated_betting_edge:false});
const confirmed=s.fuseRoadResearch(syntheticResearch,syntheticPath,
 makeRoad("B"),s.config(),"B");
assert.equal(confirmed.research_view.road_confirmed,true);
assert.equal(confirmed.research_view.direction,"B");
assert(Math.abs(confirmed.future_path.future_edge_score-.58)<1e-12);
assert.equal(confirmed.future_path.unadjusted_future_edge_score,.5);
const conflict=s.fuseRoadResearch(syntheticResearch,syntheticPath,
 makeRoad("P"),s.config(),"B");
assert.equal(conflict.research_view.road_conflict,true);
assert.equal(conflict.research_view.direction,"B");
assert(Math.abs(conflict.future_path.future_edge_score-.44)<1e-12);
const capped=s.fuseRoadResearch(syntheticResearch,
 {...syntheticPath,future_edge_score:.98},makeRoad("B"),s.config(),"B");
assert.equal(capped.future_path.future_edge_score,1);
const notEligible=s.fuseRoadResearch({...syntheticResearch,eligible:false,
 direction:"SKIP"},syntheticPath,makeRoad("B"),s.config(),"B");
assert.equal(notEligible.research_view.eligible,false);
assert.equal(notEligible.research_view.road_confirmed,false);
assert.equal(notEligible.research_view.direction,"SKIP");
const roadOff=s.fuseRoadResearch(syntheticResearch,syntheticPath,
 makeRoad("P"),s.config({ROAD_AUX_ENABLED:false}),"B");
assert.strictEqual(roadOff.research_view,syntheticResearch);
assert.strictEqual(roadOff.future_path,syntheticPath);
// Full runtime: toggling road affects only display fields and config.
const sameHistory=Array.from({length:21},(_,i)=>"BPPBTBBPPT"[i%10]);
const liveOn=s.run(sameHistory);
const liveOff=s.run(sameHistory,{ROAD_AUX_ENABLED:false});
for(const k of ["version","candidate","action","direction","probabilities",
 "physical_probabilities","ev_banker","ev_player","ev_tie","particle_ess",
 "ancestor_diversity","adaptive_sampling","wager_gate","bankroll_fraction",
 "simulation_samples","features48","seed"]){
 assert.deepEqual(liveOn[k],liveOff[k],"road must not change physical/formal "+k);
}
assert.equal(liveOn.research_view.direction,liveOff.research_view.direction);
assert.equal(liveOn.research_view.eligible,liveOff.research_view.eligible);
assert.equal(liveOn.action,"Skip");
assert.equal(liveOff.action,"Skip");
assert.equal(liveOn.validated_edge,false);
assert.equal(liveOff.validated_edge,false);
assert.equal(liveOn.road_aux.estimated_only,true);
assert.equal(liveOff.road_aux.reason,"disabled");
const postOn=s.posterior(sameHistory,s.config());
const postOff=s.posterior(sameHistory,s.config({ROAD_AUX_ENABLED:false}));
assert.deepEqual(postOn.particles,postOff.particles,
 "road switch cannot change physical posterior particles");
assert.deepEqual(postOn.weights,postOff.weights,
 "road switch cannot change particle weight fingerprint");
assert.equal(s.run(Array(10).fill("B")).road_aux.available,false);

assert.equal(s.DEFAULT.PARTICLE_COUNT,448);
assert.equal(s.DEFAULT.LIKELIHOOD_DRAWS,80);
assert.equal(s.DEFAULT.NEXT_HAND_DRAWS,128);
assert.equal(s.DEFAULT.ADAPTIVE_EXTRA_DRAWS,32);
assert.equal(s.DEFAULT.ADAPTIVE_EXTRA_DRAWS_MAX,48);
assert.equal(s.DEFAULT.ADAPTIVE_MULTI_HORIZON_ENABLED,true);
assert.equal(s.DEFAULT.ADAPTIVE_HORIZON_HANDS,2);
assert.equal(s.DEFAULT.ADAPTIVE_HORIZON_WEIGHT,.15);
assert.equal(s.DEFAULT.ADAPTIVE_GAP_MARGIN,.008);
assert.throws(()=>s.config({ADAPTIVE_EXTRA_DRAWS:49}),/adaptive_sample_budget_exceeded/);
assert.throws(()=>s.config({ADAPTIVE_EXTRA_DRAWS_MAX:49}),/invalid_ADAPTIVE_EXTRA_DRAWS_MAX/);
assert.throws(()=>s.config({ADAPTIVE_MULTI_HORIZON_ENABLED:"true"}),
 /invalid_ADAPTIVE_MULTI_HORIZON_ENABLED/);
assert.equal(s.DEFAULT.MIN_EV_THRESHOLD,.0030);
assert.equal(s.DEFAULT.MID_SHOE_MIN_EV_THRESHOLD,.00235);
assert.equal(s.DEFAULT.LATE_SHOE_MIN_EV_THRESHOLD,.0019);
assert.equal(s.DEFAULT.MIN_CONFIDENCE,.75);
assert.equal(s.DEFAULT.LATE_MIN_CONFIDENCE,.80);
assert.equal(s.DEFAULT.MIN_BET_EV_GAP,.0015);
assert.equal(s.DEFAULT.MIN_BET_ESS_RATIO,.45);
assert.equal(s.DEFAULT.MIN_BET_QUALITY,.65);
assert.equal(s.DEFAULT.MAX_BET_EV_SE,.060);
assert.equal(s.DEFAULT.FRACTIONAL_KELLY,.225);
assert.equal(s.DEFAULT.MAX_BET_FRACTION,.0175);
assert.equal(s.DEFAULT.MIN_REMAINING_CARDS_FOR_BET,25);
assert.equal(s.DEFAULT.END_SHOE_OVERRIDE_MIN_EV,.025);
assert.equal(s.DEFAULT.END_SHOE_OVERRIDE_CONFIDENCE,.90);
assert.equal(s.DEFAULT.END_SHOE_OVERRIDE_MAX_EV_SE,.030);
assert.equal(s.DEFAULT.AVERAGE_CARDS_PER_HAND,4.95);
assert.equal(s.DEFAULT.DEPTH_RELIABILITY,.5);
assert.equal(s.DEFAULT.MIN_DIRECTION_ROUNDS,3);
assert.equal(s.DEFAULT.MIN_DIRECTION_EV_GAP,.010);
assert.equal(s.DEFAULT.MIN_DIRECTION_GAP_SNR,1.05);
assert.equal(s.DEFAULT.MIN_DIRECTION_ESS_RATIO,.40);
assert.equal(s.DEFAULT.MIN_DIRECTION_DIVERSITY,.25);
assert.equal(s.DEFAULT.MAX_DIRECTION_GAP_SE,.070);
assert.equal(s.DEFAULT.RESEARCH_DIRECTION_RELAXED,true);
assert.throws(()=>s.config({RESEARCH_DIRECTION_RELAXED:"false"}),
 /invalid_RESEARCH_DIRECTION_RELAXED/);

const gateArgs={
 candidate:"B",ev:{B:.12,P:-.15},
 physical:{samples:128*14,p:{B:.53,P:.38,T:.09},
  per:Array.from({length:128},()=>({B:7,P:5,T:2}))},
 observations:8,essRatio:.85,diversity:.70,config:s.DEFAULT
};
const eligible=s.moderateDirectionGate(gateArgs);
assert.equal(eligible.direction,"B","clear modeled EV separation should display candidate");
assert.equal(eligible.eligible,true);
assert.equal(eligible.validated_betting_edge,false);
assert.equal(eligible.estimated_only,true);
// Synthetic boundary fixture: it passes the relaxed display gate but fails the
// V5 strict gate. Values are model diagnostics, not actual observed cards.
const criticalPhysical={samples:128*250,p:{B:.4,P:.384,T:.216},
 per:Array.from({length:128},(_,i)=>i<64?{B:120,P:100,T:30}:{B:80,P:92,T:78}),
 weights:Array(128).fill(1/128)};
const criticalArgs={candidate:"B",ev:s.evs(criticalPhysical.p),physical:criticalPhysical,
 observations:3,essRatio:.43,diversity:.27,config:s.config()};
const relaxedCritical=s.moderateDirectionGate(criticalArgs);
const strictCritical=s.moderateDirectionGate({...criticalArgs,
 config:s.config({RESEARCH_DIRECTION_RELAXED:false})});
assert.equal(relaxedCritical.eligible,true,"near-threshold research case passes relaxed gate");
assert.equal(relaxedCritical.direction,"B");
assert(relaxedCritical.ev_gap_snr>=1.05&&relaxedCritical.ev_gap_snr<1.15);
assert.equal(strictCritical.eligible,false,"rollback restores V5 strict threshold behavior");
assert.equal(strictCritical.direction,"SKIP");
assert(strictCritical.reasons.includes("weak_ev_separation"));
assert.equal(s.moderateDirectionGate({...gateArgs,observations:1}).reason,"insufficient_rounds");
assert.equal(s.moderateDirectionGate({...gateArgs,ev:{B:.003,P:0}}).reason,"weak_ev_separation");
assert.equal(s.moderateDirectionGate({...gateArgs,essRatio:.1}).reason,"low_particle_ess");
assert.equal(s.moderateDirectionGate({...gateArgs,diversity:.05}).reason,"low_particle_diversity");
assert.equal(s.moderateDirectionGate({...gateArgs,physical:{...gateArgs.physical,samples:100}})
 .reason,"high_estimator_uncertainty");
assert.equal(s.moderateDirectionGate({...gateArgs,ev:{B:.040,P:0}})
 .reason,"low_gap_to_noise");

assert.throws(()=>s.run(["X"]),/invalid_bpt_history/);
assert.throws(()=>s.config({PARTICLE_COUNT:1024}),/invalid_PARTICLE_COUNT/);
assert.throws(()=>s.config({MAX_BET_EV_SE:-1}),/invalid_MAX_BET_EV_SE/);
assert.throws(()=>s.config({MID_SHOE_MIN_EV_THRESHOLD:.005}),/invalid_wager_gate_tier_order/);
assert.throws(()=>s.config({LATE_MIN_CONFIDENCE:.6}),/invalid_wager_gate_tier_order/);
assert.throws(()=>s.config({END_SHOE_OVERRIDE_MAX_EV_SE:.09}),/invalid_wager_gate_tier_order/);

// Formal wager-layer contract only. These deterministic synthetic boundary
// fixtures are not retrospective real-shoe results and do not predict a 53%-54% hit rate.
const cfg=s.config();
const policy=remainingRatio=>s.wagerPolicy(416*remainingRatio,cfg);
assert.equal(policy(.61).min_ev_threshold,.0030,"early shoe >60% threshold");
assert.equal(policy(.60).min_ev_threshold,.00235,"60% inclusive in middle tier");
assert.equal(policy(.30).min_ev_threshold,.00235,"30% inclusive in middle tier");
assert.equal(policy(.2999).min_ev_threshold,.0019,"below 30% late tier");
assert.equal(policy(.35).min_confidence,.75,"at 35% remain uses standard");
assert.equal(policy(.3499).min_confidence,.80,"below 35% remain confidence 80%");
const strong={
 cfg,remainingCards:416*.70,ess:230,quality:.82,
 ev:{B:.05,P:.012,T:0},candidate:"B",
 confidence:.88,se:.020,lower:.012,enoughDepth:true,observations:12,
 model:"validated_external_xgb"
};
const gate=(patch={})=>s.wagerGate({...strong,...patch});
assert.equal(gate().eligible,true,"all formal hypothetical thresholds pass");
assert.equal(gate().min_ev_threshold,.0030);
assert.equal(gate().min_confidence,.75);
assert.equal(gate({model:"physics_only_untrained_decision_fallback"}).eligible,false,
 "no verified XGB cannot authorize a stake");
assert.equal(gate({observations:0}).eligible,false);
assert.equal(gate({enoughDepth:false}).eligible,false);
assert.equal(gate({ess:.45*448}).eligible,true,"ESS 45% exact threshold passes");
assert(gate({ess:.45*448-.001}).reasons.includes("low_particle_ess"));
assert(gate({quality:.649}).reasons.includes("low_particle_quality"));
assert(gate({ev:{B:.05,P:.048501,T:0}}).reasons.includes("ev_gap_too_small"));
assert.equal(gate({ev:{B:.05,P:.0485,T:0}}).eligible,true,
 "exact gap 0.0015 passes");
assert(gate({confidence:.749}).reasons.includes("confidence_below_dynamic_gate"));
assert(gate({remainingCards:416*.34,confidence:.799}).reasons.includes("confidence_below_dynamic_gate"));
assert.equal(gate({remainingCards:416*.34,confidence:.80}).eligible,true);
assert(gate({remainingCards:416*.7,ev:{B:.0029,P:-.015}}).reasons.includes("ev_below_dynamic_threshold"));
assert.equal(gate({remainingCards:416*.7,ev:{B:.003,P:-.015}}).eligible,true);
assert(gate({se:.06001}).reasons.includes("ev_uncertainty_too_high"));
assert(gate({lower:.003}).reasons.includes("ev_lower_bound_too_low"));
assert.equal(gate({lower:.0030001}).eligible,true);
assert(gate({remainingCards:24.999}).reasons.includes("shoe_end_no_bet"));
assert.equal(gate({remainingCards:25}).eligible,true,
 "exact 25 cards does not require special override");
assert.equal(gate({
 remainingCards:20,ev:{B:.025,P:0,T:0},
 confidence:.90,se:.030,lower:.010
}).eligible,true,"exception demands high EV, high confidence and low SE");
assert(gate({
 remainingCards:20,ev:{B:.0249,P:0,T:0},
 confidence:.90,se:.030
}).reasons.includes("shoe_end_no_bet"));
assert(gate({
 remainingCards:20,ev:{B:.025,P:0,T:0},
 confidence:.899,se:.030
}).reasons.includes("shoe_end_no_bet"));
assert(gate({
 remainingCards:20,ev:{B:.025,P:0,T:0},
 confidence:.90,se:.03001
}).reasons.includes("shoe_end_no_bet"));
const noModel=s.run(["B","P","T","B","B"]);
const noModelStrict=s.run(["B","P","T","B","B"],{RESEARCH_DIRECTION_RELAXED:false});
assert.equal(noModel.wager_gate.model_verified,false);
assert.equal(noModel.wager_gate.min_ev_threshold,.003);
assert.equal(noModel.action,"Skip");
assert.equal(noModelStrict.action,"Skip","research rollback never opens formal betting");
assert.equal(noModel.bankroll_fraction,0);
assert.equal(noModelStrict.bankroll_fraction,0);
assert.deepEqual(noModel.physical_probabilities,noModelStrict.physical_probabilities,
 "research threshold setting does not change forecast probabilities");
assert.deepEqual(noModel.adaptive_sampling,noModelStrict.adaptive_sampling,
 "research threshold setting does not change adaptive next-hand sampling");
assert.equal(noModel.research_view.estimated_only,true);
assert.equal(noModel.research_view.validated_betting_edge,false);
assert.deepEqual(noModel.wager_gate.reasons[0],"trained_decision_model_unavailable");

const counts=[128,32,32,32,32,32,32,32,32,32];
for(let seed=1;seed<=400;seed++){
 let r=s.deal(counts,(()=>{let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};})());
 assert(["B","P","T"].includes(r.result));
 assert(r.draws>=4&&r.draws<=6);
 assert.equal(r.counts.reduce((a,b)=>a+b,0),416-r.draws);
 assert.equal(r.result,r.banker>r.player?"B":r.player>r.banker?"P":"T");
}
const empty=s.run([]);
assert.equal(empty.direction,"SKIP");
assert.equal(empty.feature_dimension,48);
assert.equal(empty.feature_schema,"soft_shoe_bpt_48d_v2");
assert.equal(empty.frozen_core.validated,false);
assert.equal(empty.features48.length,48);
assert.equal(empty.observed_actual_cards,false);
assert.equal(empty.bankroll_fraction,0);
for(const h of [["B"],["P"],["T"],["B","P","T","B"],Array(18).fill("T")]){
 const first=s.run(h),second=s.run(h);
 assert.deepEqual(first,second,"same observed sequence must be deterministic");
 assert.equal(first.observed_actual_cards,false);
 assert.equal(first.feature_dimension,48);
 assert.equal(first.feature_schema,"soft_shoe_bpt_48d_v2");
 assert.equal(first.sequence_weight_applied,0);
 assert.deepEqual(first.features48.slice(-3),
  [first.frozen_core.probabilities.B,first.frozen_core.probabilities.P,
   first.frozen_core.probabilities.T]);
 assert(first.features48.every(Number.isFinite));
 assert([448*128,448*(128+19)].includes(first.simulation_samples));
 assert.equal(first.adaptive_sampling.baseline_draws,128);
 assert([0,32].includes(first.adaptive_sampling.extra_draws));
 assert.equal(first.adaptive_sampling.accepted,
  first.adaptive_sampling.extra_draws===32);
 assert(first.future_path);
 assert.equal(first.future_path.estimated_only,true);
 assert.equal(first.future_path.validated_betting_edge,false);
 if(first.future_path.available){
  assert(first.future_path.future_edge_score>=0&&first.future_path.future_edge_score<=1);
  assert(first.future_path.path_consistency>=0&&first.future_path.path_consistency<=1);
 }
 const p=first.probabilities;
 assert(Math.abs(p.B+p.P+p.T-1)<1e-10);
 assert(Math.abs(first.ev_banker-(.95*p.B-p.P))<1e-12);
 assert(Math.abs(first.ev_player-(p.P-p.B))<1e-12);
 assert(Math.abs(first.ev_tie-(9*p.T-1))<1e-12);
 assert(["B","P"].includes(first.candidate),"candidate available for every valid forecast");
 assert(first.research_view&&["B","P","SKIP"].includes(first.research_view.direction));
 assert.equal(first.research_view.candidate,first.candidate);
 assert(first.research_view.ev_gap_se>0&&Number.isFinite(first.research_view.ev_gap_se));
 assert.equal(first.research_view.validated_betting_edge,false);
 assert.equal(first.direction,"SKIP","keep independent wager-eligibility gate");
 assert.equal(first.action,"Skip","no unvalidated betting recommendation");
 assert.equal(first.bankroll_fraction,0,"no staking advice without validation");
 assert.equal(first.frozen_core.trained,false);
 assert.equal(first.model,"physics_only_untrained_decision_fallback");
 assert(first.estimated_penetration>0);
 assert(first.estimated_penetration<1);
 assert(first.particle_ess>=1&&first.particle_ess<=448+1e-8);
}
// Sequential importance sampling: only ESS-triggered iterations resample.
const oneHand=s.posterior(["B"]);
assert.equal(oneHand.available,true);
assert.equal(oneHand.weights.length,s.DEFAULT.PARTICLE_COUNT);
assert(Math.abs(oneHand.weights.reduce((a,b)=>a+b,0)-1)<1e-9);
assert.equal(oneHand.resampleCount,0,"high first-hand ESS does not force resampling");
const forced=s.posterior(["B","P","T","B","P"],{RESAMPLE_ESS_RATIO:1});
assert(forced.resampleCount>0,"high ESS threshold activates systematic resampling");
assert(Math.abs(forced.weights.reduce((a,b)=>a+b,0)-1)<1e-9);
const pWeighted=s.handProbabilities(oneHand.particles,s.DEFAULT,oneHand.seed,oneHand.weights);
assert.equal(pWeighted.samples,448*128);
assert(Math.abs(Object.values(pWeighted.p).reduce((a,b)=>a+b)-1)<1e-9);
assert(pWeighted.sd.B>=0&&Number.isFinite(pWeighted.sd.B));
// CPU diagnostic only (synthetic history), NOT evidence of wagering accuracy.
// Keep sample timings visible in CI before promoting to LINE's 8-second worker.
for(const length of [10,30,50,70]){
 const history=Array.from({length},(_,i)=>"BPPBTBBPPT"[i%10]);
 const start=performance.now();
 const result=s.run(history);
 const elapsed=performance.now()-start;
 console.log("[SOFT_SHOE_PERF]",JSON.stringify({length,ms:+elapsed.toFixed(2),
  available:!!result.probabilities,reason:result.reason||"",resamples:
  s.posterior(history).resampleCount}));
 assert(result.probabilities&&Number.isFinite(result.ev_banker));
}
assert.equal(s.run(Array(73).fill("B")).reason,"shoe_depth_limit");
const adaptiveCfg=s.config();
const nearThreshold={abs_ev_gap:.013,ev_gap_snr:1.18};
const farThreshold={abs_ev_gap:.040,ev_gap_snr:2.00};
assert.equal(s.shouldRefine(nearThreshold,3,.60,.40,adaptiveCfg),true,
 "near EV-gap/SNR threshold with adequate posterior quality refines");
assert.equal(s.shouldRefine(farThreshold,3,.60,.40,adaptiveCfg),false,
 "non-critical estimate does not refine");
assert.equal(s.shouldRefine(nearThreshold,3,.41,.40,adaptiveCfg),false,
 "ESS below 0.42 does not refine");
assert.equal(s.shouldRefine(nearThreshold,3,.60,.27,adaptiveCfg),false,
 "diversity below 0.28 does not refine");
assert.equal(s.shouldRefine(nearThreshold,3,.75,.40,adaptiveCfg),false,
 "highly concentrated ESS does not refine");
assert.equal(s.shouldRollbackAdaptive(
 {ess_ratio:.60,diversity:.50},{ess_ratio:.55,diversity:.50},adaptiveCfg),true,
 "more than 8% ESS decline rolls back");
assert.equal(s.shouldRollbackAdaptive(
 {ess_ratio:.60,diversity:.50},{ess_ratio:.60,diversity:.47},adaptiveCfg),false,
 "diversity decline under 8% does not roll back");
assert.equal(s.directionConsistency(["B","T","P"],"B"),.5,
 "ties are excluded from direction consistency");
const futureScore=s.futureEdgeScore(1.20,.80,.25);
assert(futureScore>=0&&futureScore<=1);
assert.equal(futureScore,1,"future score is clamped to 0..1");
assert.equal(s.futureEdgeScore(1,.5,null),null,
 "incomplete path metrics do not create a score");
const disabled=s.run(["B","P","B"],{ADAPTIVE_MULTI_HORIZON_ENABLED:false});
assert.equal(disabled.adaptive_sampling.accepted,false);
assert.equal(disabled.simulation_samples,448*128);
assert.equal(disabled.future_path.reason,"feature_disabled",
 "switch restores V5 sampling and disables new diagnostics");
const physical={B:.458597,P:.446247,T:.095156};
assert.deepEqual(s.modelDecision(physical,[],s.frozenCore([]),null),
 {p:physical,model:"physics_only_untrained_decision_fallback"});
const f=Array(48).fill(0);
const mock={trained:true,calibrated:true,schema:"soft_shoe_bpt_48d_v2",feature_dimension:48,
 validation:{out_of_shoe_holdout:true,real_shoe_data:true},
 predict:()=>({B:.6,P:.3,T:.1})};
assert.equal(s.modelDecision(physical,f,s.frozenCore(["B"]),mock).model,"validated_external_xgb");
assert.equal(s.modelDecision(physical,f,s.frozenCore(["B"]),{...mock,feature_dimension:57}).model,
 "physics_only_untrained_decision_fallback");
const goodTree={nodeid:0,split:"f0",split_condition:0.5,yes:1,no:2,
 children:[{nodeid:1,leaf:0.8},{nodeid:2,leaf:-0.4}]};
assert.equal(s.evaluateXgbTree(goodTree,Array(48).fill(0)),0.8);
const bundle={trees:[goodTree],base_logits:[0,0,0],tree_info:[0],learning_rate:1,temperature:1};
const evaluated=s.predictXgbJson(bundle,Array(48).fill(0));
assert(evaluated.B>evaluated.P);
assert(Math.abs(evaluated.B+evaluated.P+evaluated.T-1)<1e-10);
assert.equal(s.modelDecision(physical,Array(48).fill(0),s.frozenCore(["B"]),
 {...mock,predict:undefined,...bundle}).model,"validated_external_xgb");
assert.equal(s.modelDecision(physical,f,s.frozenCore(["B"]),
 {...mock,validation:{out_of_shoe_holdout:true,real_shoe_data:false}}).model,
 "physics_only_untrained_decision_fallback");
const css=fs.readFileSync("styles.css","utf8");
assert(!/#predictionApp\s+\.bbb-results-block\s*,/.test(css),
 "legacy point-entry CSS must not hide the B/P/T panel");
assert(!/#predictionApp\.bbb-point-mode\s+\.bbb-results-block\s*\{\s*display:\s*none/.test(css),
 "stale point-mode class must not hide B/P/T buttons");
assert(css.includes(".result-buttons{display:grid;grid-template-columns:repeat(3,1fr)"));
const html=fs.readFileSync("index.html","utf8"),auth=fs.readFileSync("auth-client.js","utf8"),
 panel=fs.readFileSync("soft_shoe_panel.js","utf8"),worker=fs.readFileSync("soft_shoe_worker.js","utf8");
for(const id of ["btnB","btnP","btnT","btnStart","btnBack","btnEnd"])
 assert(html.includes('id="'+id+'"'));
assert(html.includes("styles.css?v=20261106-bigroad-v1"));
assert(html.includes("auth-client.js?v=20261112-v5-1-road-aux-1"));
assert(auth.includes('"derived_road_aux.js?v=20261112-v5-1-road-aux-1"'));
assert(auth.includes('"soft_shoe_decision_runtime.js?v=20261112-v5-1-road-aux-1"'));
assert(auth.includes('"bpt_big_road.js?v=20261106-bigroad-v1"'));
assert(auth.includes('"soft_shoe_panel.js?v=20261112-v5-1-road-aux-1"'));
assert(!/const scripts=\[[\s\S]*?pf_eor_v4_panel\.js/.test(auth));
assert(panel.includes('for(const [id,side] of [["btnB","B"],["btnP","P"],["btnT","T"]]'));
assert(worker.includes("derived_road_aux.js?v=20261112-v5-1-road-aux-1"));
assert(worker.includes("soft_shoe_decision_runtime.js?v=20261112-v5-1-road-aux-1"));
assert(worker.includes('soft_shoe_frozen_lstm_runtime.js?v=20261106-layered-v4'));
assert(worker.includes('soft_shoe_frozen_model_bundle.js?v=20261106-layered-v4'));
assert(panel.includes("soft_shoe_worker.js?v=20261112-v5-1-road-aux-1"));
assert(panel.includes('d=qualified&&["B","P"].includes(view.direction)?view.direction:"SKIP"'));
assert(panel.includes('!["B","P","SKIP"].includes(report.research_view.direction)'));
assert(panel.includes('set("directionText",d==="B"?"莊":d==="P"?"閒":"觀望")'));
assert(panel.includes("觀望規則"));
assert(html.includes('id="roadAuxStatus"'));
assert(panel.includes("路同向"));
assert(panel.includes("路分歧"));
assert(panel.includes("牌路為輔助圖案訊號，非保證勝率"));
assert(panel.includes("syncBigRoad();"));
assert(panel.includes("Future Edge"));
assert(panel.includes("路徑 EV₂"));
assert(auth.indexOf('"bpt_big_road.js')<auth.indexOf('"soft_shoe_panel.js'));
assert(panel.includes("觀望：EV 差異估計誤差過大"));
assert(panel.includes("觀望：有效粒子不足"));


const diagnosticCounts={B:0,P:0,SKIP:0};
const diagnosticViews=[];
let diagnosticState=0x516ec24d;
for(let i=0;i<32;i++){
 const history=[];
 for(let j=0;j<3+(i%18);j++){
  diagnosticState=(Math.imul(diagnosticState,1664525)+1013904223)>>>0;
  const u=diagnosticState/4294967296;
  history.push(u<.459?"B":u<.905?"P":"T");
 }
 const report=s.run(history);
 const d=report.research_view?.direction||"SKIP";
 assert(["B","P","SKIP"].includes(d));
 diagnosticCounts[d]++;
 diagnosticViews.push({observations:history.length,view:report.research_view||{}});
 assert.equal(report.bankroll_fraction,0);
 assert.equal(report.validated_edge,false);
}
assert(diagnosticCounts.B+diagnosticCounts.P>=1,"do not ship all-SKIP synthetic regression");
console.log("Moderate-gate synthetic BPT sequences (not holdout/hitrate)",diagnosticCounts);
const conditionBreakdown={gap:0,snr:0,ess:0,div:0,se:0,all:0};
const profileCurrent={gap:.025,snr:.90,ess:.40,div:.25,maxSe:.070};
for(const {view} of diagnosticViews){
 if(view.abs_ev_gap>=profileCurrent.gap)conditionBreakdown.gap++;
 if(view.ev_gap_snr>=profileCurrent.snr)conditionBreakdown.snr++;
 if(view.ess_ratio>=profileCurrent.ess)conditionBreakdown.ess++;
 if(view.ancestor_diversity>=profileCurrent.div)conditionBreakdown.div++;
 if(view.ev_gap_se>0&&view.ev_gap_se<=profileCurrent.maxSe)conditionBreakdown.se++;
}
console.log("[SOFT_SHOE_CONDITION_BREAKDOWN]",JSON.stringify(conditionBreakdown));
console.log("[SOFT_SHOE_DIRECTION_SAMPLES]",JSON.stringify(diagnosticViews.map(({observations,view})=>({
 r:observations,gap:+(view.abs_ev_gap||0).toFixed(4),snr:+(view.ev_gap_snr||0).toFixed(3),
 se:+(view.ev_gap_se||0).toFixed(4),ess:+(view.ess_ratio||0).toFixed(3),
 div:+(view.ancestor_diversity||0).toFixed(3),
 candidate:view.candidate
})).slice(0,16)));
const researchProfiles={
 previous:{gap:.025,snr:.90,ess:.40,div:.25,maxSe:.070},
 middle:{gap:.030,snr:1.00,ess:.40,div:.25,maxSe:.070},
 conservative:{gap:.035,snr:1.15,ess:.45,div:.30,maxSe:.060},
 soft:{gap:.025,snr:.85,ess:.35,div:.20,maxSe:.080},
 signalA:{gap:.012,snr:1.15,ess:.45,div:.30,maxSe:.060},
 signalB:{gap:.015,snr:1.15,ess:.45,div:.30,maxSe:.060},
 signalC:{gap:.018,snr:1.15,ess:.45,div:.30,maxSe:.060},
 signalD:{gap:.020,snr:1.00,ess:.40,div:.25,maxSe:.070},
};
for(const [label,p] of Object.entries(researchProfiles)){
 const counts={B:0,P:0,SKIP:0};
 for(const {observations,view} of diagnosticViews){
  const pass=observations>=3&&view.abs_ev_gap>=p.gap&&view.ev_gap_se>0&&
    Number.isFinite(view.ev_gap_se)&&view.ev_gap_se<=p.maxSe&&
    view.ev_gap_snr>=p.snr&&view.ess_ratio>=p.ess&&
    view.ancestor_diversity>=p.div;
  const dir=pass?view.candidate:"SKIP";counts[dir||"SKIP"]++;
 }
 console.log("[SOFT_SHOE_GATE_SWEEP]",JSON.stringify({label,...counts}));
}

console.log("PASS BPT latent PF, legal baccarat deals, 48D features, model guard and interface");
