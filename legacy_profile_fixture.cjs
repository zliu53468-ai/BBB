// Test-only: existing PR75 suites explicitly exercise rollback behavior.
// New production-default road suites run without this preload.
const fs=require('fs'),path=require('path'),read=fs.readFileSync;
fs.readFileSync=function(file,...args){const raw=read.call(this,file,...args);if(typeof file!=='string'||path.basename(file)!=='cardcount_profile.json')return raw;const p=JSON.parse(String(raw));if(p.bayes_markov_final_v1)p.bayes_markov_final_v1.enabled=false;if(p.road_primary_v1)p.road_primary_v1.enabled=false;const text=JSON.stringify(p);return Buffer.isBuffer(raw)?Buffer.from(text):text;};
