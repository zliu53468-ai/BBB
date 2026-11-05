'use strict';
// Rebuild: Add-Type ExactScoreBuild.cs; [ExactScoreBuild]::Run('exact.tsv')
// Then: node tools/stability/package_reference.js exact.tsv
const fs=require('fs');
if(!process.argv[2])throw Error('Pass the TSV produced by ExactScoreBuild.cs');
const rows=fs.readFileSync(process.argv[2],'utf8').trim().split(/\r?\n/).slice(1).map(line=>line.split('\t').slice(0,6).map(Number));
if(rows.length!==100||rows.some((r,i)=>r[0]*10+r[1]!==i||!r.every(Number.isFinite)||r[2]<=0||Math.abs(r[3]+r[4]+r[5]-1)>1e-10))throw Error('invalid reference');
const data={version:'BBB_EXACT_POINT_REFERENCE_V1',decks:8,assumption:'fresh_eight_deck_single_final_score',independent_evidence:false,
 source:'Complete 339400 legal value paths, 6958 removal compositions, two-hand no-replacement marginalization',rows};
fs.writeFileSync('stability_reference.js',`(function(r){const a=${JSON.stringify(data)};if(typeof module==='object'&&module.exports)module.exports=a;else r.BBBExactPointReference=a;})(typeof self!=='undefined'?self:this);\n`);
