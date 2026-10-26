"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),crypto=require('node:crypto');
const src=fs.readFileSync('particle_filter_runtime.js','utf8'),fixture=require('./validation/pf_update_legacy_fixture.json'),ctx={};ctx.window=ctx;vm.createContext(ctx);vm.runInContext(src,ctx);const pf=ctx.__BGS_PARTICLE500__;
assert.equal(crypto.createHash('sha256').update(src.split('function estimate(')[0]).digest('hex'),fixture.rules_prefix_sha256,'draw/third-card/forecast rules changed');
const run=(h,c)=>pf.estimate(h,fixture.direct,'quality',{forecast_draws:3,...c});
const old=run(fixture.history,{likelihood_draws:2,fallback_draws:2});assert.equal(JSON.stringify(old.physics),JSON.stringify(fixture.physics));assert.equal(JSON.stringify(old.posterior),JSON.stringify(fixture.posterior));
const config=require('./cardcount_profile.json').pf;assert.equal(config.likelihood_draws,4);assert.equal(config.fallback_draws,4);
const a=run(fixture.history,config),b=run(fixture.history,config);assert.equal(JSON.stringify(a),JSON.stringify(b),'same history must be deterministic');
assert.equal(a.diagnostics.observations_processed,fixture.history.length);assert.equal(a.diagnostics.likelihood_draws,4);assert.equal(a.diagnostics.fallback_draws,4);assert.equal(a.diagnostics.persistent_state,false);assert.equal(a.diagnostics.particle_count,500);assert(a.diagnostics.blend_weight<=.16);
assert.notEqual(JSON.stringify(a.posterior),JSON.stringify(old.posterior),'new sampling budget not used');
assert.notEqual(JSON.stringify(run([...fixture.history,'B'],config).posterior),JSON.stringify(run([...fixture.history,'P'],config).posterior),'new observed outcome must update posterior');
assert.equal(JSON.stringify(a),JSON.stringify(run(fixture.history,config)),'undo must restore posterior');
for(const start of [0,3,13,23]){const size=start===0||start===23?3:10;assert(Math.abs(a.posterior.slice(start,start+size).reduce((x,y)=>x+y,0)-1)<1e-10);}
const html=fs.readFileSync('index.html','utf8'),auth=fs.readFileSync('auth-client.js','utf8');assert(!/bbbScan|bbbCapture|accept="image/.test(html));assert(!/bbb-screenshot-road|bbb-desktop-capture/.test(auth));for(const id of ['btnB','btnP','btnT','btnStart','btnBack','btnEnd','btnNoSkipTest'])assert(html.includes('id="'+id+'"'));
console.log(JSON.stringify({passed:true,legacy_2_draws_exact:true,rules_unchanged:true,posterior_updates_per_observation:true,particle_count:500,image_analysis_unloaded:true}));
