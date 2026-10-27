"use strict";
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const baseline=require('./validation/pr59_prediction_baseline.json'),current=require('./cardcount_profile.json');
const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.equal(sha('final_probability_runtime.js'),baseline.runtime_sha256,'PR59 runtime must be byte-identical');
const expected=structuredClone(baseline.profile);Object.assign(expected.pf,{update_profile:'per_hand_posterior_v2',likelihood_draws:4,fallback_draws:4});assert.deepEqual(current,expected,'only PF update settings may differ from PR59');
for(const [file,hash] of Object.entries(baseline.ui_sha256))assert.equal(sha(file),hash,'current UI must remain unchanged');
assert.equal(current.adaptive_smoothing,undefined);assert.equal(current.direction_selection.mode,'post_draw_ev_argmax_v1');
console.log(JSON.stringify({passed:true,pr59_runtime_exact:true,only_pf_parameters_differ:true,current_ui_unchanged:true}));
