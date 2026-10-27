#!/usr/bin/env node
"use strict";
// Candidate-only two-parameter calibration; never patches website/runtime models.
const fs=require('fs'),crypto=require('crypto'),S=require('./no_skip_parameter_search.js'),V=require('./no_skip_validation.js');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
function choose(cache,fitted,settings){
 const split=S.splitShoes(cache.rows),prior=S.fitPrior(split.train.rows);
 const selectionBaseline=V.metrics(split.selection.rows.map(r=>({...r,candidate:['B','P'].includes(r.current)?r.current:'B'})),'candidate');
 const scores=fitted.candidates.map((calibrator,i)=>{const config={id:'bounded_platt_'+i,source:'markov8',alpha:'off',prior_weight:0,markov_weight:.08,architecture_eligible:true,calibrator};return {config,metrics:V.metrics(S.runPolicy(split.selection.rows,config,prior,settings),'candidate')};}).sort(S.ranking);
 return {schema_version:1,candidate_only:true,production_promotion_allowed:false,training_data_kind:fitted.training_data_kind,training_seed:fitted.training_seed,training_input_sha256:fitted.training_input_sha256,training_cache_sha256:fitted.training_cache_sha256,runtime_sha256:fitted.runtime_sha256,prior,selected:scores[0].config,selection_scores:scores,selection_baseline:selectionBaseline,selection_safety_passed:scores[0].metrics.hit_rate_excluding_pushes>=selectionBaseline.hit_rate_excluding_pushes&&scores[0].metrics.realized_ev_per_bet>=selectionBaseline.realized_ev_per_bet,unused_previous_holdout_shoes:fitted.unused_previous_holdout_shoes,previous_dataset_shoe_ids:[...split.train.ids,...split.selection.ids,...split.holdout.ids],note:'Frozen before new holdout collection; no model weights or final57 features changed'};
}
function evaluate(cache,candidate,settings){
 if(typeof cache.data_kind!=="string"||typeof cache.input_sha256!=="string"||!Array.isArray(cache.rows))throw Error("Complete frozen cache metadata required");
 if(cache.input_sha256===candidate.training_input_sha256)throw Error('Training input reused as holdout');
 if(JSON.stringify(cache.hashes)!==JSON.stringify(candidate.runtime_sha256))throw Error('Runtime changed since calibration fit');
 if(cache.data_kind.includes('synthetic')&&candidate.training_data_kind.includes('synthetic')){
  if(cache.seed===null||candidate.training_seed===null||cache.seed===candidate.training_seed)throw Error('Fresh synthetic seed required');
 }else if(cache.data_kind==='real_observations'&&candidate.training_data_kind==='real_observations'&&cache.rows.some(r=>candidate.previous_dataset_shoe_ids.includes(r.shoe_id)))throw Error('Real training/holdout shoes overlap');
 const output=S.runPolicy(cache.rows,candidate.selected,candidate.prior,settings),baseline=cache.rows.map(r=>({...r,candidate:['B','P'].includes(r.current)?r.current:'B'}));
 const stats=V.metrics(output,'candidate'),base=V.metrics(baseline,'candidate'),ci=V.bootstrap(output,'candidate'),paired=V.pairedBootstrap(output,baseline),gate=V.assess(cache.data_kind,stats,ci);
 if(candidate.selection_safety_passed!==true)gate.failures.push('Calibration-split hit/EV safety check failed before new holdout');
 if(stats.hit_rate_excluding_pushes<base.hit_rate_excluding_pushes)gate.failures.push('Hit rate regressed against current zero-skip baseline');
 if(stats.realized_ev_per_bet<base.realized_ev_per_bet)gate.failures.push('EV regressed against current zero-skip baseline');
 if(stats.brier_bpt>base.brier_bpt+.0015)gate.failures.push('Brier regressed by >0.0015');if(!paired.hit_change95||paired.hit_change95[0]<0||!paired.ev_change95||paired.ev_change95[0]<0)gate.failures.push('Paired bootstrap does not establish non-regression');gate.statistical_checks_passed=!gate.failures.length;
 const stages={};for(const [name,f] of Object.entries({all:()=>true,'1-20':r=>r<=20,'21-45':r=>r>=21&&r<=45,'46-70':r=>r>=46&&r<=70,'<=40':r=>r<=40,'41-50':r=>r>=41&&r<=50,'>50':r=>r>50,'50-70':r=>r>=50&&r<=70}))stages[name]={candidate:V.metrics(output.filter(r=>f(r.round_index)),'candidate'),baseline:V.metrics(baseline.filter(r=>f(r.round_index)),'candidate')};
 return {schema_version:1,data_kind:cache.data_kind,seed:cache.seed,input_sha256:cache.input_sha256,runtime_sha256:cache.hashes,selected:candidate.selected,stages,whole_shoe_bootstrap:ci,paired_change:paired,gate,production_changed:false,physics_pf_retrained:false};
}
function main(args){
 const mode=args[0],option=k=>{const i=args.indexOf(k);return i<0?null:args[i+1]};const cache=JSON.parse(fs.readFileSync(option('--cache'),'utf8')),output=option('--output');if(!output||fs.existsSync(output))throw Error('Use a fresh --output');
 const settings=require('../cardcount_profile.json').local_calibration;let report;
 if(mode==='choose'){const fitted=JSON.parse(fs.readFileSync(option('--fitted'),'utf8'));report=choose(cache,fitted,settings);}
 else if(mode==='evaluate'){const candidate=JSON.parse(fs.readFileSync(option('--candidate'),'utf8'));report=evaluate(cache,candidate,settings);}
 else throw Error('choose or evaluate required');
 fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(mode==='choose'?{selected:report.selected,selection:report.selection_scores}:{holdout:report.stages.all,ci:report.whole_shoe_bootstrap,gate:report.gate},null,2));
 return mode==='choose'||report.gate.statistical_checks_passed?0:2;
}
module.exports={choose,evaluate};if(require.main===module){try{process.exitCode=main(process.argv.slice(2));}catch(e){console.error(e);process.exitCode=1;}}
