'use strict';
const assert=require('node:assert/strict'),fs=require('fs');
const x=require('./pure_card_runtime.js');
assert.equal(x.CONFIG.particles,5000);assert.equal(x.CONFIG.mcTrials,10000);
assert(Math.abs(x.BASE.B-.458597422632763)<1e-10);
assert(Math.abs(x.BASE.P-.446246609343596)<1e-10);
for(const row of require('./validation/pure_card_exact_fixtures.json')){
 const q=x.exact(row.counts);for(const side of ['B','P','T'])assert(Math.abs(q[side]-row.probability[side])<1e-10);
}
assert.deepEqual(x.parse('J+5+2'),[0,5,2]);assert.deepEqual(x.parse('A 10'),[1,0]);
assert.throws(()=>x.parse('5'));assert.throws(()=>x.parse('X+3'));
assert.throws(()=>x.observation([4,4,1],[3,4]));
assert.throws(()=>x.observation([0,5],[3,4]));
assert.throws(()=>x.observation([4,3],[1,1]));
const o=x.observation([4,4],[3,4]);assert.equal(o.outcome,'P');
const snapshot=JSON.stringify(o),e=x.eor(o.used),pf=x.sample(o,e);
assert.equal(pf.particles.length,5000);assert(Math.abs(pf.particles.reduce((s,p)=>s+p.weight,0)-1)<1e-12);
for(const p of pf.particles)for(let j=0;j<10;j++)assert(p.counts[j]>=0&&p.counts[j]<=o.center[j]&&Number.isInteger(p.counts[j]));
const start=Date.now(),r=x.run([4,4],[3,4]);console.log('Full pipeline ms',Date.now()-start);
assert.equal(r.mc.samples,10000);assert.equal(r.mc.depletion,true);assert.equal(r.decision.direction,'SKIP');
assert.equal(JSON.stringify(o),snapshot);assert.deepEqual(x.sample(o,e),pf);
const zero={direction:'B'},theory={ev:{B:x.BASE_EV.B+.003,P:x.BASE_EV.P}},m={ev:{B:0,P:-.1},win_probability_se:{B:.0015,P:.0015}};
assert.equal(x.decide(zero,theory,m,{penalty:0}).direction,'B');
assert.equal(x.decide({direction:'P'},theory,m,{penalty:0}).direction,'SKIP');
assert.equal(x.decide(zero,theory,{...m,win_probability_se:{B:.0015001}},{penalty:0}).direction,'SKIP');
assert.equal(x.decide(zero,{ev:{B:x.BASE_EV.B+.00249,P:x.BASE_EV.P}},m,{penalty:0}).direction,'SKIP');
const opposed=Array.from({length:20},()=>({eor_direction:'B',actual:'P'}));assert.equal(x.diagnostic(opposed).penalty,.0005);
assert.equal(x.diagnostic(opposed.slice(0,3)).penalty,0);
const auth=fs.readFileSync('auth-client.js','utf8'),loaders=auth.match(/const scripts=\[[\s\S]*?\];/)[0];
assert(!/LinUCB|quant256|road|markov|physics_forecast/.test(loaders));
assert(loaders.includes('pure_card_panel.js'));
const canonical=auth.replace(/\r\n/g,'\n').replace(/const scripts=\[[\s\S]*?\];/,'const scripts=[];');
assert.equal(require('crypto').createHash('sha256').update(canonical).digest('hex'),
 '21485e0daaf2436ec343bc672b70a1587f093f374c2afc1175e2b80a7a33030a','license logic outside script inventory must be unchanged');
assert(!fs.readFileSync('pure_card_runtime.js','utf8').includes('localStorage'));
console.log('PASS local-card configuration, parsing, legal draws, 5000 bounded particles, exact baseline, MC, strict gates, diagnostic tightening, no legacy loaders');
