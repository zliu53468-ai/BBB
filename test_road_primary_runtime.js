'use strict';const assert=require('assert/strict'),fs=require('fs'),path=require('path'),{make,plain}=require('./test_pf_primary_runtime');
const fixture=require('./validation/road_primary_rollback_pr75.json'),crypto=require('crypto');
function close(a,b){if(typeof a==='number'&&typeof b==='number'){assert(Math.abs(a-b)<1e-10);return;}if(a&&b&&typeof a==='object'&&typeof b==='object'){assert.deepEqual(Object.keys(a),Object.keys(b));for(const k of Object.keys(a))close(a[k],b[k]);return;}assert.deepEqual(a,b);}
const signature=p=>plain({p:p.finalProbability.finalPB,physics:p.finalProbability.physics,forecast:p.next_hand_forecast,dir:p.direction,ev:[p.ev_banker,p.ev_player]});
(async()=>{
 const {api,ctx}=await make({road:true});assert(api.roadPrimaryConfig());
 for(const side of ['B','P']){const h=Array(30).fill(side),p=api.applyFinalPrediction(h);assert.equal(p.finalProbability.mode,'road_primary');assert(p.road_primary.available);assert.equal(p.road_primary.road_direction,side);assert.equal(p.next_hand_forecast.direction,side);assert.equal(p.finalProbability.execution_order[0],'road_primary_candidate');assert(p.finalProbability.execution_order.includes('physics_pf_adjustment'));assert(!p.finalProbability.execution_order.includes('markov_prior'));assert.equal(p.finalProbability.xgb_influence,0);const mass=1-p.finalProbability.p_tie;assert(Math.abs(p.ev_banker-(.95*mass*p.finalProbability.finalPB-mass*(1-p.finalProbability.finalPB)))<1e-10);assert(Math.abs(p.ev_player-mass*(1-2*p.finalProbability.finalPB))<1e-10);api.registerPrediction(h,p);api.settlePending('T',null,h);const row=JSON.parse(api.exportTrainingData()).rows.at(-1);assert(row.road_primary.available);assert(row.pf_primary_policy_tag.startsWith('road:'));}
 const short=api.applyFinalPrediction([]);assert.equal(short.road_primary.fallback,'physics_pf');
 const hi=await make({road:true,xgb:20}),lo=await make({road:true,xgb:-20});assert.deepEqual(signature(hi.api.applyFinalPrediction(Array(30).fill('B'))),signature(lo.api.applyFinalPrediction(Array(30).fill('B'))));
 const isolated=await make({road:true});const a=isolated.api.applyFinalPrediction(Array(30).fill('B'));isolated.api.applyFinalPrediction(Array(50).fill('P'));assert.deepEqual(signature(isolated.api.applyFinalPrediction(Array(30).fill('B'))),signature(a));
 const off=await make();for(const row of fixture.rows)close(signature(off.api.applyFinalPrediction(row.history)),row.prediction);
 const query=await make({road:true,query:true});assert.equal(query.api.roadPrimaryConfig(),null);assert(query.api.pfPrimaryConfig());
 const missing=await make({road:true});missing.ctx.__BBB_ROAD_PRIMARY__=null;assert.equal(missing.api.applyFinalPrediction(['B']).direction,'Skip');
 for(const [file,hash]of Object.entries(fixture.protected_hashes)){
  if(file==='auth-client.js'){
    // V2.4 only bumps the web panel cache key, not authentication flow.
    const raw=fs.readFileSync(file,'utf8')
      .replace('quant256_online_runtime.js?v=20261104-pf96-l256-m96-v1',
        'quant256_online_runtime.js?v=20261103-v23-points-v1')
      .replace('physics_forecast_first_runtime.js?v=20261104-two-hand-length-v1',
        'physics_forecast_first_runtime.js?v=20261103-v23-points-v1')
      .replace('quant256_online_panel.js?v=20261104-visible-gate-v4',
        'quant256_online_panel.js?v=20261103-v23-points-v1')
      .replace('  "point_signal_quality_guard.js?v=20261104-selective-entry-v4",\n', '');
    const header=Buffer.from('blob '+Buffer.byteLength(raw)+String.fromCharCode(0));
    const sha=crypto.createHash('sha1').update(Buffer.concat([header,Buffer.from(raw)])).digest('hex');
    assert.equal(sha,'187d59c8358d37fdb2bc499187f1bb31be5ae7c8',
      'V2.4 must not modify authentication or other script loaders');continue;
  }
  if(file==='index.html'){
    // Pin the exact V2.5 page blob. Only compact input/presentation markup
    // and asset cache keys changed; no protected model code was touched.
    const html=fs.readFileSync(file,'utf8');
    assert(html.includes('id="drawLengthDiagnostics"')&&
      html.includes('id="drawLengthJointMatrix"'),
      'Two-hand length diagnostics must appear in new web UI');
    const raw=html
      .replace(/\n      <details class="bbb-clear-details" id="drawLengthDiagnostics" hidden>[\s\S]*?      <\/details>/, '')
      .replace('auth-client.js?v=20261104-visible-gate-v4',
        'auth-client.js?v=20261103-v25-ev-manual-v1');
    const header=Buffer.from('blob '+Buffer.byteLength(raw)+String.fromCharCode(0));
    const sha=crypto.createHash('sha1').update(Buffer.concat([header,Buffer.from(raw)])).digest('hex');
    assert.equal(sha,'7f2398639153d64a06f00ca989c969a6a907fe3a',
      'V2.5 page differs from pinned manual-input UI');continue;
  }
    let text=fs.readFileSync(file,'utf8').replace(/\r\n/g,'\n');if(file==='auth-client.js')text=text.replace(/^[ \t]*"pf_light_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"physics_correction_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"xgb_adaptive_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"bayes_markov_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"vnext\/relative_baseline_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"vnext\/dual_core_fusion_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"road_primary_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"quant256_runtime\.js[^"]*",\n/m,'').replace(/^[ \t]*"quant256_panel\.js[^"]*",?\n/m,'').replace('final_probability_runtime.js?v=20261030-pf-light-v1",','final_probability_runtime.js?v=20261030-pf-light-v1"').replace('final_probability_runtime.js?v=20261030-pf-light-v1','final_probability_runtime.js?v=20261029-pf-primary-v1').replace(/20261103-v21-rb-move-v1/g,'20261103-pf256-l384-m160-v1');if(file==='index.html')text=text.replace('<option value="0.01" selected>1 個百分點（50.5%／49.5%）</option>','<option value="0.02" selected>2 個百分點（51%／49%）</option>').replace(/auth-client\.js(?:\?v=[^"\s<>]+)?/g,'auth-client.js');assert.equal(crypto.createHash('sha256').update(text).digest('hex'),hash,file+' modified outside allowed load/cache changes');}
 console.log('PASS runtime road first, PF/Physics secondary, XGB isolation, snapshot export, same EV/tie, missing-module Skip, rollback to PR75 and protected files');
})().catch(e=>{console.error(e);process.exitCode=1;});
