"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const s=require("./soft_shoe_decision_runtime.js");
assert.equal(s.DEFAULT.PARTICLE_COUNT,128);
assert.equal(s.DEFAULT.LIKELIHOOD_DRAWS,14);
assert.equal(s.DEFAULT.NEXT_HAND_DRAWS,14);
assert.equal(s.DEFAULT.MIN_EV_THRESHOLD,.0015);
assert.equal(s.DEFAULT.MIN_CONFIDENCE,.65);
assert.equal(s.DEFAULT.FRACTIONAL_KELLY,.30);
assert.equal(s.DEFAULT.MAX_BET_FRACTION,.025);
assert.equal(s.DEFAULT.AVERAGE_CARDS_PER_HAND,4.95);
assert.equal(s.DEFAULT.DEPTH_RELIABILITY,.5);
assert.equal(s.DEFAULT.MIN_DIRECTION_ROUNDS,3);
assert.equal(s.DEFAULT.MIN_DIRECTION_EV_GAP,.020);
assert.equal(s.DEFAULT.MIN_DIRECTION_GAP_SNR,.70);
assert.equal(s.DEFAULT.MIN_DIRECTION_ESS_RATIO,.35);
assert.equal(s.DEFAULT.MIN_DIRECTION_DIVERSITY,.20);
assert.equal(s.DEFAULT.MAX_DIRECTION_GAP_SE,.080);

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
assert.equal(s.moderateDirectionGate({...gateArgs,ev:{B:.025,P:0}})
 .reason,"low_gap_to_noise");

assert.throws(()=>s.run(["X"]),/invalid_bpt_history/);
assert.throws(()=>s.config({PARTICLE_COUNT:1024}),/invalid_PARTICLE_COUNT/);
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
assert.equal(empty.features48.length,48);
assert.equal(empty.observed_actual_cards,false);
assert.equal(empty.bankroll_fraction,0);
for(const h of [["B"],["P"],["T"],["B","P","T","B"],Array(18).fill("T")]){
 const first=s.run(h),second=s.run(h);
 assert.deepEqual(first,second,"same observed sequence must be deterministic");
 assert.equal(first.observed_actual_cards,false);
 assert.equal(first.feature_dimension,48);
 assert(first.features48.every(Number.isFinite));
 assert.equal(first.simulation_samples,128*14);
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
 assert(first.particle_ess>=1&&first.particle_ess<=128+1e-8);
}
assert.equal(s.run(Array(73).fill("B")).reason,"shoe_depth_limit");
const physical={B:.458597,P:.446247,T:.095156};
assert.deepEqual(s.modelDecision(physical,[],s.frozenCore([]),null),
 {p:physical,model:"physics_only_untrained_decision_fallback"});
const f=Array(48).fill(0);
const mock={trained:true,calibrated:true,schema:"soft_shoe_bpt_48d_v1",feature_dimension:48,
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
assert(html.includes("styles.css?v=20261106-bpt-ui-fix-v2"));
assert(html.includes("auth-client.js?v=20261106-moderate-v3"));
assert(auth.includes('"soft_shoe_decision_runtime.js?v=20261106-moderate-v3"'));
assert(auth.includes('"soft_shoe_panel.js?v=20261106-moderate-v3"'));
assert(!/const scripts=\[[\s\S]*?pf_eor_v4_panel\.js/.test(auth));
assert(panel.includes('for(const [id,side] of [["btnB","B"],["btnP","P"],["btnT","T"]]'));
assert(worker.includes("soft_shoe_decision_runtime.js?v=20261106-moderate-v3"));
assert(panel.includes('d=view.direction||"SKIP"'));
assert(panel.includes('if(!report.research_view||!["B","P","SKIP"].includes(report.research_view.direction))'));
assert(panel.includes('set("directionText",d==="B"?"莊":d==="P"?"閒":"觀望")'));
assert(panel.includes("適度觀望"));
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
