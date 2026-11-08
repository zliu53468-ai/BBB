"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const s=require("./soft_shoe_decision_runtime.js");
assert.equal(s.DEFAULT.PARTICLE_COUNT,448);
assert.equal(s.DEFAULT.LIKELIHOOD_DRAWS,80);
assert.equal(s.DEFAULT.NEXT_HAND_DRAWS,128);
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
assert.equal(s.DEFAULT.MIN_DIRECTION_EV_GAP,.035);
assert.equal(s.DEFAULT.MIN_DIRECTION_GAP_SNR,1.15);
assert.equal(s.DEFAULT.MIN_DIRECTION_ESS_RATIO,.45);
assert.equal(s.DEFAULT.MIN_DIRECTION_DIVERSITY,.30);
assert.equal(s.DEFAULT.MAX_DIRECTION_GAP_SE,.060);

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
assert.equal(noModel.wager_gate.model_verified,false);
assert.equal(noModel.wager_gate.min_ev_threshold,.003);
assert.equal(noModel.action,"Skip");
assert.equal(noModel.bankroll_fraction,0);
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
 assert.equal(first.simulation_samples,448*128);
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
assert(html.includes("auth-client.js?v=20261108-pf448-l80-m128-v8"));
assert(auth.includes('"soft_shoe_decision_runtime.js?v=20261108-pf448-l80-m128-v8"'));
assert(auth.includes('"bpt_big_road.js?v=20261106-bigroad-v1"'));
assert(auth.includes('"soft_shoe_panel.js?v=20261106-bigroad-v1"'));
assert(!/const scripts=\[[\s\S]*?pf_eor_v4_panel\.js/.test(auth));
assert(panel.includes('for(const [id,side] of [["btnB","B"],["btnP","P"],["btnT","T"]]'));
assert(worker.includes("soft_shoe_decision_runtime.js?v=20261108-pf448-l80-m128-v8"));
assert(worker.includes('soft_shoe_frozen_lstm_runtime.js?v=20261106-layered-v4'));
assert(worker.includes('soft_shoe_frozen_model_bundle.js?v=20261106-layered-v4'));
assert(panel.includes("soft_shoe_worker.js?v=20261108-pf448-l80-m128-v8"));
assert(panel.includes('d=view.direction||"SKIP"'));
assert(panel.includes('if(!report.research_view||!["B","P","SKIP"].includes(report.research_view.direction))'));
assert(panel.includes('set("directionText",d==="B"?"莊":d==="P"?"閒":"觀望")'));
assert(panel.includes("適度觀望"));
assert(panel.includes("syncBigRoad();"));
assert(auth.indexOf('"bpt_big_road.js')<auth.indexOf('"soft_shoe_panel.js'));
assert(panel.includes("觀望：EV 差異估計誤差過大"));
assert(panel.includes("觀望：有效粒子不足"));


const diagnosticCounts={B:0,P:0,SKIP:0};
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
 assert.equal(report.bankroll_fraction,0);
 assert.equal(report.validated_edge,false);
}
console.log("Moderate-gate synthetic BPT sequences (not holdout/hitrate)",diagnosticCounts);

console.log("PASS BPT latent PF, legal baccarat deals, 48D features, model guard and interface");
