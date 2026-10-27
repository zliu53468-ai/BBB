"use strict";
const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
function make(options={}){
 const store=new Map(),ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},fetch:async file=>({ok:true,json:async()=>{const m=JSON.parse(fs.readFileSync(file,'utf8'));if(file==='cardcount_profile.json'){if(options.disabled)m.hitrate_max_v1.enabled=false;if(options.markovOff)m.hitrate_max_v1.markov.enabled=false;}if(file==='final_probability_model.json'&&options.xgbUnavailable)m.trained=false;if(file==='final_probability_model.json'&&options.xgb!==undefined){m.base_margin=options.xgb;m.trees=[];m.calibration={method:'identity'};}return m;}})};
 ctx.window=ctx;vm.createContext(ctx);for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(f,'utf8'),ctx);return ctx.__BGS_FINAL56__;
}
const plain=v=>JSON.parse(JSON.stringify(v));
(async()=>{
 const off=make({disabled:true}),a=make({xgb:-20}),b=make({xgb:20}),missing=make({xgbUnavailable:true}),pure=make({markovOff:true});await Promise.all([off.loadModels(),a.loadModels(),b.loadModels(),missing.loadModels(),pure.loadModels()]);a.hitrateConfig().xgb_influence=1;
 for(const row of require('./validation/hitrate_max_rollback.json').rows){
  const p=off.applyFinalPrediction(row.history.split('')),r=p.finalProbability,q=pure.applyFinalPrediction(row.history.split(''));assert.equal(q.finalProbability.finalPB,r.finalPB,'Markov disabled changed pure Physics/PF probability');if(q.direction!=='Skip')assert.equal(q.direction,p.direction,'strict profile relaxed old qualification');
  assert.deepEqual(plain({history:row.history,direction:p.direction,forecast:p.next_hand_forecast,finalPB:r.finalPB,evB:p.ev_banker,evP:p.ev_player,smoothing:r.smoothingAlpha,guard:p.volume_guard_active,candidate:r.primary_candidate}),row,'disabled rollback');
  const x=a.applyFinalPrediction(row.history.split('')),y=b.applyFinalPrediction(row.history.split(''));
  assert.deepEqual([x.direction,x.finalProbability.finalPB,x.ev_banker,x.ev_player],[y.direction,y.finalProbability.finalPB,y.ev_banker,y.ev_player],'XGB changed a decision');const z=missing.applyFinalPrediction(row.history.split(''));assert.deepEqual([x.direction,x.finalProbability.finalPB],[z.direction,z.finalProbability.finalPB]);assert.equal(x.finalProbability.xgb_influence,0);
 }
 const physics=Array(48).fill(0);[physics[23],physics[24],physics[25]]=[.60,.31,.09];
 const history='B'.repeat(60).split(''),config={markov:{enabled:true,mix_weight:1,max_weight:1,min_observations:12,prior_strength:24}};
 const mixed=a.markovAdjustment(physics,history,config);assert(mixed.weight>0&&mixed.weight<=.12);assert.equal(mixed.markov_direction,'B');
 assert.deepEqual(plain(a.markovAdjustment(physics,history,{markov:{enabled:false}}).physics),physics);
 const opposed=a.markovAdjustment(physics,('BBP'.repeat(30)+'BB').split(''),config);assert.equal(opposed.weight,0);assert.equal(opposed.reason,'opposite_or_balanced_ignored');
 const prefix=a.markovAdjustment(physics,history,config);a.markovAdjustment(physics,['P','T','P'],config);assert.deepEqual(plain(a.markovAdjustment(physics,history,config)),plain(prefix),'undo/replay must be stateless');
 assert.equal(a.markovAdjustment(physics,[],config).weight,0,'new shoe must not inherit counts');
 const c=a.hitrateConfig();assert(a.hitrateGate('B',.04,-.06,10,.2,.2,.8,c).passed);
 for(const [r,t] of [[20,.012],[21,.009],[45,.009],[46,.007]])assert.equal(a.hitrateGate('B',.04,-.06,r,.2,.2,.8,c).threshold,t);
 assert(a.hitrateGate('P',.04,-.06,30,.2,.2,.8,c).reasons.includes('direction_consistency'));
 assert(a.hitrateGate('B',.04,-.06,30,.9,.2,.8,c).reasons.includes('noise'));
 assert(a.hitrateGate('B',.04,-.06,30,.2,.9,.8,c).reasons.includes('uncertainty'));
 assert(a.hitrateGate('B',.04,-.06,30,.2,.2,undefined,c).reasons.includes('particle_quality'));
 assert(!a.hitrateGate('B',-.01,-.04,30,.2,.2,.8,c).passed);
 assert(a.hitrateGate('B',.004,.003,30,.2,.2,.8,c).reasons.includes('ev_gap'));
 assert.equal(a.noSkipTestDirection({available:true,directional:{B:.5,P:.5}}),'Balanced');
 assert.equal(a.noSkipTestDirection({available:true,directional:{B:.49,P:.51}}),'P');assert.equal(a.noSkipTestDirection({available:true,directional:{B:.51,P:.49}}),'B');
 for(let n=10;n<26;n++){const h='B'.repeat(n).split(''),p=a.applyFinalPrediction(h);p.direction='Skip';p.finalProbability.finalDirection='觀望 Skip';a.registerPrediction(h,p);a.settlePending('B',{outcome:'B'},h);}
 const guard=a.volumeGuardState(45,.6,a.decisionPolicy(45,.2));assert(guard.active,'fixture must exercise old Volume relief');
 const strict=a.applyFinalPrediction('B'.repeat(35).split(''));assert.equal(strict.volume_guard_active,false);assert.equal(strict.hitrate_max.volume_guard.relief_applied,false);
 assert(a.getTrainingRows().at(-1).hitrate_max,'exported settlement row lost audit');
 console.log('hitrate_max: rollback, XGB isolation, capped same-side causal Markov, strict entry and unbiased exact ties passed');
})().catch(e=>{console.error(e);process.exitCode=1});
