"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const HISTORY='bgs256d_frozen_6x15_forward_v18',SHOE='bgs_xgb_final_shoe_id_v5';
function make({enabled=true,xgb=0,failExperiment=false,storage=null,fixture=true}={}){
 const map=storage||new Map();if(!map.has(SHOE))map.set(SHOE,'test-shoe');
 const localStorage={getItem:k=>map.get(k)??null,setItem:(k,v)=>{if(failExperiment&&k.startsWith('bgs_no_skip_experiment_'))throw Error('quota');map.set(k,String(v));},removeItem:k=>map.delete(k)};
 const ctx={console,localStorage,document:{getElementById:()=>null},fetch:async name=>({ok:true,json:async()=>{const data=JSON.parse(fs.readFileSync(name,'utf8'));if(name==='cardcount_profile.json'&&fixture){
 // Synthetic display-model fixture tests mechanics only. Production's old 500P calibrator stays disabled.
 const fp=ctx.__BBB_BROWSER_CALIBRATION__.fingerprint,c=data.experimental_no_skip_v1;
 c.enabled=enabled;c.version='TEST_FIXTURE_ONLY_CURRENT_RUNTIME';c.model={type:'bounded_monotone_platt',slope:1.02,intercept:.001,max_delta:.01};
 c.expected_model_fingerprint=fp(c.model);c.expected_particle_version=ctx.__BGS_PARTICLE500__.version;
 c.expected_physics_fingerprint=fp(JSON.parse(fs.readFileSync('physics_multitask_model.json','utf8')));
 const final=JSON.parse(fs.readFileSync('final_probability_model.json','utf8'));
 c.expected_input_fingerprint=fp({markov:data.hitrate_max_v1.markov,pf:data.pf,pf_max:data.anchors.pf_max,pf_draws:data.anchors.pf_draws,noise_score_version:final.training?.noise_score_version||1});
 }if(name==='final_probability_model.json'&&xgb){data.trees=[];data.base_margin=xgb;data.calibration={method:'identity'};}return data;}})};ctx.window=ctx;vm.createContext(ctx);
 for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(f,'utf8'),ctx,{filename:f});return {api:ctx.__BGS_FINAL56__,map,ctx};
}
const plain=x=>JSON.parse(JSON.stringify(x));
(async()=>{
 const a=make(),off=make({enabled:false}),opposed=make({xgb:20});await Promise.all([a.api.loadModels(),off.api.loadModels(),opposed.api.loadModels()]);
 const production=make({fixture:false});await production.api.loadModels();assert(!production.api.getExperimentalStatus([]).enabled,"production incompatible 500P calibration must remain disabled");
 assert(a.api.getExperimentalStatus([]).enabled);assert(!off.api.getExperimentalStatus([]).enabled);
 for(const text of ['B','BPTBPBBPP','P'.repeat(50)])assert.deepEqual(plain(a.api.applyFinalPrediction(text.split(''))),plain(off.api.applyFinalPrediction(text.split(''))),'normal mode changed');
 a.api.setNoSkipTestMode(true);opposed.api.setNoSkipTestMode(true);
 for(const text of ['B','BPTBPBBPP','P'.repeat(50)]){const h=text.split(''),p=a.api.applyFinalPrediction(h),b=off.api.applyFinalPrediction(h),x=opposed.api.applyFinalPrediction(h);
  assert(p.experimental_no_skip.active);assert.equal(p.direction,b.direction);assert.equal(p.finalProbability.finalPB,b.finalProbability.finalPB);assert.equal(p.ev_banker,b.ev_banker);assert.equal(p.ev_player,b.ev_player);
  assert(Math.abs(p.experimental_no_skip.delta)<=.010000000001);assert.equal(p.next_hand_forecast.probabilities.T,p.base_raw_next_hand_forecast.probabilities.T);
  assert.equal(p.next_hand_forecast.directional.B,x.next_hand_forecast.directional.B,'XGB affected experimental output');
  a.api.registerPrediction(h,p);assert.equal(a.api.getTrainingRows().length,0,'experiment was registered as wager');
 }
 let history=[];for(const actual of 'BPTBBPPBBPPTBPBPPBTBBPPTBBPPTBBPPBTBP'){
  const p=a.api.applyFinalPrediction(history),q=off.api.applyFinalPrediction(history);
  const t=a.api.prepareDisplayCalibrationTickets(history,p.base_raw_next_hand_forecast||p.raw_next_hand_forecast),u=off.api.prepareDisplayCalibrationTickets(history,q.raw_next_hand_forecast);
  history.push(actual);a.map.set(HISTORY,JSON.stringify({history}));off.map.set(HISTORY,JSON.stringify({history}));a.api.commitDisplayCalibrationTickets(t,actual,history);off.api.commitDisplayCalibrationTickets(u,actual,history);a.api.recordCompletedHand(history,{outcome:actual});off.api.recordCompletedHand(history,{outcome:actual});
 }
 assert.deepEqual(plain(a.api.getLocalCalibrationStatus(history)),plain(off.api.getLocalCalibrationStatus(history)),'normal calibration polluted');
 const rows=a.api.getExperimentalRows(history);assert.equal(rows.length,history.length);assert.equal(rows[0].round_index,1);assert(rows.every(r=>r.display_only&&r.unvalidated));
 assert(a.map.has('bgs_browser_calibration_v1'));assert(a.map.has('bgs_no_skip_experiment_bgs_browser_calibration_v1'));
 const before=a.api.applyFinalPrediction(history).next_hand_forecast;
 assert.deepEqual(plain(before),plain(a.api.applyFinalPrediction(history).next_hand_forecast),'repeat advanced state');
 const reloaded=make({storage:a.map});await reloaded.api.loadModels();reloaded.api.setNoSkipTestMode(true);assert.deepEqual(plain(before),plain(reloaded.api.applyFinalPrediction(history).next_hand_forecast));
 const xgbReload=make({storage:new Map(a.map),xgb:-20});await xgbReload.api.loadModels();xgbReload.api.setNoSkipTestMode(true);assert.deepEqual(plain(before),plain(xgbReload.api.applyFinalPrediction(history).next_hand_forecast),'XGB weights reset experimental learning');
 const undone=history.slice(0,-1);assert.equal(a.api.getExperimentalRows(undone).length,undone.length);
 a.api.setNoSkipTestMode(false);assert(!a.api.applyFinalPrediction(history).experimental_no_skip);assert.equal(a.api.getTrainingRows().length,0);
 const exportData=JSON.parse(a.api.exportTrainingData());assert.equal(exportData.rows.length,0);assert.equal(exportData.experimental_no_skip_rows.length,history.length);
 a.map.set(SHOE,'new-shoe');a.map.set(HISTORY,JSON.stringify({history:[]}));assert.equal(a.api.getExperimentalStatus([]).calibration.directional_samples,0);
 const quota=make({failExperiment:true});await quota.api.loadModels();quota.api.setNoSkipTestMode(true);const p=quota.api.applyFinalPrediction([]),t=quota.api.prepareDisplayCalibrationTickets([],p.base_raw_next_hand_forecast);quota.map.set(HISTORY,JSON.stringify({history:['B']}));quota.api.commitDisplayCalibrationTickets(t,'B',['B']);assert.equal(quota.api.getLocalCalibrationStatus(['B']).directional_samples,1);assert.equal(quota.api.getTrainingRows().length,0);
 const pending=make();await pending.api.loadModels();const nodes=new Map();for(const id of ['directionOrb','directionText','directionCode','confidence','regime','strength','modePill','roundCount','message','localCalibrationStatus','experimentalNoSkipStatus','btnNoSkipTest'])nodes.set(id,{textContent:'',className:'',setAttribute(k,v){this[k]=v;}});pending.ctx.document.getElementById=id=>nodes.get(id)||null;
 pending.map.set(HISTORY,JSON.stringify({history:[]}));pending.api.setNoSkipTestMode(true);const shown=pending.api.applyFinalPrediction([]);pending.api.renderPrediction(shown,0);assert(pending.api.getExperimentalPending([]).was_shown);
 pending.api.setNoSkipTestMode(false);const next=pending.api.applyFinalPrediction([]),tickets=pending.api.prepareDisplayCalibrationTickets([],next.raw_next_hand_forecast),actual=tickets.paper.direction==='B'?'P':'B';pending.map.set(HISTORY,JSON.stringify({history:[actual]}));pending.api.commitDisplayCalibrationTickets(tickets,actual,[actual]);const saved=pending.api.getExperimentalRows([actual]);assert.equal(saved.length,1);assert(saved[0].was_shown);assert.equal(saved[0].paper_profit_units,-1,'stop mode dropped the last displayed loss');assert.equal(pending.api.getExperimentalPending([actual]),null);
 await assert.rejects(()=>require('./tools/no_skip_validation.js').audit({data_kind:'TEST_FIXTURE_ONLY',rows:[{shoe_id:'audit',round_index:1,outcome:'B'},{shoe_id:'audit',round_index:2,outcome:'T'},{shoe_id:'audit',round_index:3,outcome:'P'}]},{experimental:true}),/Approved experimental configuration unavailable/,'production incompatible candidate must be rejected by offline audit too');
 console.log('Experimental rollout: normal invariance, XGB isolation, bounded/tie-preserving calibration, separate journals, first hand, reload/undo/new shoe, export separation, quota safety and no wagers passed');
})().catch(e=>{console.error(e);process.exitCode=1});
