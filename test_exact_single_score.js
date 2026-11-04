const assert=require('node:assert/strict');
const {PhysicsForecastFirstEngine}=require('./physics_forecast_first_runtime.js');
const {BlindPhysicsEstimator}=require('./quant256_online_runtime.js');
const guard=require('./point_signal_quality_guard.js');
const config={independentFinalScoreMode:true,exactSingleScore:true};
const rows=[];
for(let player=0;player<10;player++)for(let banker=0;banker<10;banker++){
 const outcome=banker>player?'B':player>banker?'P':'T';
 const m=new PhysicsForecastFirstEngine(config);
 const r=m.syncHistory([outcome],[{player,banker}]);
 const pf=m.quant.physics;
 assert.equal(pf.particles.length,96);assert.equal(pf.lastAcceptance,1);
 assert(pf.weights.every(w=>w===1/96));
 for(const counts of pf.particles){
  assert(counts.every((n,i)=>Number.isInteger(n)&&n>=0&&n<=[128,32,32,32,32,32,32,32,32,32][i]));
  const removed=416-counts.reduce((a,b)=>a+b,0);assert(removed>=4&&removed<=6);
 }
 assert.equal(pf.snapshot().likelihood_method,'exact_single_score_rejection_v1');
 assert.equal(r.physical_ev.samples,9216);assert.equal(r.bet_allowed,false);
 assert(!r.physical_ev.simulation_exhausted);
 assert(Math.abs(Object.values(r.physical_ev.simulated_probability).reduce((a,b)=>a+b)-1)<1e-9);
 rows.push({player,banker,direction:guard.decide(r).direction,evB:r.ev_banker,evP:r.ev_player});
}
const failed=new BlindPhysicsEstimator({exactSingleScore:true});
failed.rng=()=>.999999;
const before=JSON.stringify(failed.particles);
assert.throws(()=>failed.observe('P',{player:9,banker:8}),/budget_exhausted/);
assert.equal(JSON.stringify(failed.particles),before);assert(failed.posteriorUnavailable);
const a=new PhysicsForecastFirstEngine(config),b=new PhysicsForecastFirstEngine(config);
assert.deepEqual(a.syncHistory(['P'],[{player:5,banker:3}]),b.syncHistory(['P'],[{player:5,banker:3}]));
assert.throws(()=>a.syncHistory(['P','B'],[{player:5,banker:3},{player:2,banker:4}]),/cannot_accumulate/);
console.log('PASS exact independent score particles; 100 pairs; atomic failure; fresh deterministic replay');
const {spawnSync}=require('node:child_process');
const path=require('node:path');
const worker=path.resolve(__dirname,'../BGS/bbb_engine/worker_v26.cjs');
if(require('node:fs').existsSync(worker)){
 const result=spawnSync(process.execPath,[worker],{input:rows.map(r=>JSON.stringify(r)).join('\n')+'\n',encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);
 const actual=result.stdout.trim().split('\n').map(line=>JSON.parse(line));
 assert.equal(actual.length,100);
 actual.forEach((r,i)=>{assert(r.ok);assert.equal(r.result.ev_banker,rows[i].evB);assert.equal(r.result.ev_player,rows[i].evP);assert.equal(r.result.direction,rows[i].direction);});
 console.log('PASS all 100 web / LINE exact output parity');
}
console.log('Measured display counts (not hit rate):',JSON.stringify(rows.reduce((a,r)=>(a[r.direction]++,a),{B:0,P:0,SKIP:0}))); 

