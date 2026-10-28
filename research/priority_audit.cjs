'use strict';
const fs=require('fs'),path=require('path'),cp=require('child_process'),assert=require('assert/strict'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),d=path.resolve(process.argv[2]||path.join(__dirname,'reports/priority_v1_holdout')),hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const protocol=JSON.parse(fs.readFileSync(path.join(d,'protocol.json')));for(const [file,sha]of Object.entries(protocol.source_sha256))assert.equal(hash(path.join(root,file)),sha,'Locked runtime changed: '+file);
const frozen=['app256forward.js','app256continuation.js','physics_multitask_model.json','final_probability_model.json','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','auth-config.js','bbb-license-storage.js','styles.css'];
cp.execFileSync('git',['diff','--exit-code',protocol.baseline,'--',...frozen],{cwd:root});
const auth=fs.readFileSync(path.join(root,'auth-client.js'),'utf8'),old=cp.execFileSync('git',['show',protocol.baseline+':auth-client.js'],{cwd:root,encoding:'utf8'}),strip=s=>s.replace(/\r\n/g,'\n').replace(/const scripts=\[[\s\S]*?\n\];/,'const scripts=[];');assert.equal(strip(auth),strip(old),'Licensing code outside loader array changed');
const out={passed:true,checked_at:new Date().toISOString(),frozen_core_models_pf_and_auth_logic_unchanged:true,runtime_hashes_match_lock:true};
if(fs.existsSync(path.join(d,'statistics.json'))){const r=JSON.parse(fs.readFileSync(path.join(d,'statistics.json')));assert.equal(r.shoes,protocol.shoes);assert.equal(r.reporter_sha256,hash(path.join(root,'research/priority_report.py')));assert.equal(r.raw_predictions_sha256,hash(path.join(d,'predictions.jsonl')));out.statistics_sha256=hash(path.join(d,'statistics.json'));out.research_gate=r.gate;}
fs.writeFileSync(path.join(d,'engineering-audit.json'),JSON.stringify(out,null,2));console.log(JSON.stringify(out));
