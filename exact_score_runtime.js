/** Exact single-hand Bayesian marginalization under a fixed eight-deck prior.
 * No live cards, shoe depth or previous hands are inferred as observed facts.
 */
(function(root,factory){
 const table=(root&&root.BBBExactScoreTable)||
  (typeof module==='object'&&module.exports?require('./exact_score_table.js'):null);
 const api=factory(table);
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.BBBExactScore=api;
})(typeof window!=='undefined'?window:null,function(table){
 'use strict';
 const VERSION='BBB_EXACT_SINGLE_SCORE_V1',TOL=1e-10;
 if(!table||table.meta?.version!==VERSION||table.meta.decks!==8||table.rows?.length!==100)
  throw Error('exact_score_table_unavailable');
 const frozen=table.rows.map(row=>Object.freeze(row.slice()));
 let totalEvidence=0;
 for(let i=0;i<100;i++){
  const r=frozen[i];
  if(r.length!==19||r[0]!==Math.floor(i/10)||r[1]!==i%10||
    !r.every(Number.isFinite)||r.slice(2).some(v=>v<0)||
    Math.abs(r[3]+r[4]+r[5]-1)>TOL||r[2]<=0)
   throw Error('invalid_exact_score_table');
  if(Math.abs(r[7]+r[8]+r[9]-1)>TOL||
     Math.abs(r.slice(10).reduce((a,b)=>a+b,0)-1)>TOL)
   throw Error('invalid_exact_length_mass');
  for(let j=0;j<3;j++)if(Math.abs(r[7+j]-r.slice(10+3*j,13+3*j).reduce((a,b)=>a+b,0))>TOL)
   throw Error('invalid_exact_length_marginal');
  totalEvidence+=r[2];
 }
 if(Math.abs(totalEvidence-1)>TOL)throw Error('invalid_exact_evidence_mass');
 const valid=v=>Number.isInteger(v)&&v>=0&&v<=9;
 const baseline={B:0,P:0,T:0};
 for(const r of frozen)baseline[r[1]>r[0]?'B':r[0]>r[1]?'P':'T']+=r[2];
 const baseGap=1.95*baseline.B-2*baseline.P;
 function evaluate(player,banker){
  if(!valid(player)||!valid(banker))throw Error('invalid_point_observation');
  const r=frozen[player*10+banker],p={B:r[3],P:r[4],T:r[5]};
  const evB=.95*p.B-p.P,evP=p.P-p.B,gap=evB-evP;
  const direction=evB>=evP?'B':'P',lengths=[4,5,6];
  const current=Object.fromEntries(lengths.map((k,j)=>[k,r[7+j]]));
  const joint=Object.fromEntries(lengths.map((k,j)=>[k,Object.fromEntries(lengths.map((n,m)=>[n,r[10+3*j+m]]))]));
  const next=Object.fromEntries(lengths.map(k=>[k,lengths.reduce((a,n)=>a+joint[n][k],0)]));
  const exact={complete:true,version:VERSION,legal_value_sequences:339400,
   merged_depletion_states:6958,numeric_tolerance:TOL,monte_carlo_samples:0,
   evidence_probability:r[2],prior:'fresh_eight_deck_single_hand',
   observed_card_ranks:false,real_world_edge_verified:false};
  const physical={simulated_probability:p,model_probability:{...p},
   ev_banker:evB,ev_player:evP,ev_gap:gap,ev_gap_deviation:gap-baseGap,
   source:'exact_single_score_marginal',simulation_exhausted:false,
   samples:0,mc_se_gap:0,posterior_sd_gap:r[6],
   remaining:416-lengths.reduce((a,k)=>a+k*current[k],0),
   draw_length_diagnostics:{version:'BBB_EXACT_TWO_HAND_LENGTH_V1',
    current_length_probability:current,next_length_probability:next,
    joint_probability:joint,samples:0,enumerated_states:6958,
    actual_current_length_known:false,diagnostic_only:true,affects_ev_or_direction:false}};
  // A precise calculation under a hypothetical prior is not evidence of positive EV.
  return {version:VERSION,history_length:1,point_score_observation_count:1,
   exact_enumeration:exact,physical_ev:physical,
   action:'SKIP',direction:'Skip',bet_allowed:false,validated_edge:false,
   reason:'unobserved_card_composition',observed_card_evidence:false,
   ev_banker:evB,ev_player:evP,ev_gap_deviation:gap-baseGap,
   dynamic_gate:.005,early_shoe_guard:null,
   research_signal:{available:false,direction,score:Math.abs(gap-baseGap),
    dynamic_gate:.005,evidence_qualified:false,reason:'model_direction_not_validated'},
   display_direction:direction,baseline_probability:{...baseline},
   hit_rate:null,signal_rate:null,proxy_only:true,
   notice:'單局點數的精確模型推估；起始牌組是假設，並非已知現場殘牌。'};
 }
 class ExactScoreEngine {
  constructor(){this.reset();}
  reset(){this.history=[];this.scoreRows=[];this.lastReport=null;return this;}
  syncHistory(seq,scores){
   if(!Array.isArray(seq)||seq.length!==1||!Array.isArray(scores)||scores.length!==1)
    throw Error('exact_score_requires_one_resolved_hand');
   const pair=scores[0];
   if(!pair||!valid(pair.player)||!valid(pair.banker)||
      seq[0]!== (pair.banker>pair.player?'B':pair.player>pair.banker?'P':'T'))
    throw Error('invalid_point_observation');
   const report=evaluate(pair.player,pair.banker);
   this.history=seq.slice();this.scoreRows=[{player:pair.player,banker:pair.banker}];
   this.lastReport=report;return report;
  }
  setObservedRounds(rows){if(rows!==null)throw Error('web_point_only_mode');return this;}
 }
 return Object.freeze({VERSION,ExactScoreEngine,evaluate,baseline:Object.freeze(baseline)});
});
