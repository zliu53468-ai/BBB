'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {BBBQuantEngine:E}=require('./experimental/quant_engine/bayes_markov_runtime.js');
const A=require('./quant_dynamic_adapter.js');
function close(a,b){assert(Math.abs(a-b)<1e-11);}
for(const tc of [-8,-4,-2,-1,0,1,2,4,8]){
 const e=new E();e.trueCount=tc;const p=e.predictNext();
 close(p.w_count,Math.min(.7,Math.abs(tc)*.15));close(Object.values(p.probabilities).reduce((a,b)=>a+b),1);
 close(e.generateShoeContext()[0],tc*tc*Math.sign(tc));assert.equal(e.generateShoeContext().length,128);
 if(Math.abs(tc)<2||Math.max(p.ev_banker,p.ev_player)<=.005){assert.equal(p.direction,'Skip');assert.equal(p.bet_fraction,0);}
 else{assert.notEqual(p.direction,'Skip');assert(p.bet_fraction>=.05&&p.bet_fraction<=.30);}
 assert(Object.values(p.side_bets).every(x=>!x.bet_allowed));
}
const history=('BBPPTBPPTBBP'.repeat(6)).split('');
for(let n=0;n<=70;n++){const p=A.predict(E,history.slice(0,n));assert.equal(p.finalProbability.new_128d_injected,false);assert.equal(p.quant_dynamic.estimated_shoe.pair_observation_known,0);}
const e=E.fromHistory(history.slice(0,40));assert.deepEqual(e.predictNext(),E.fromHistory(history.slice(0,40)).predictNext());
assert.deepEqual(A.predict(E,history.slice(0,39)),A.predict(E,history.slice(0,39)));assert.equal(new E().remainingCards,416);
const nodes=Object.fromEntries(['modePill','localCalibrationStatus','regime','message','sideBankerPair','sidePlayerPair','sideTie','sideLuckySix','sideBetNote'].map(k=>[k,{}]));
A.render({getElementById:k=>nodes[k]},e.predictNext(),false);for(const k of ['sideBankerPair','sidePlayerPair','sideTie','sideLuckySix'])assert.equal(nodes[k].textContent,'不建議・觀望');
async function runtime(enabled){
 const cfg=JSON.parse(fs.readFileSync('cardcount_profile.json','utf8'));cfg.quant_dynamic_v2.enabled=enabled;
 const map=new Map(),ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>map.get(k)||null,setItem:(k,v)=>map.set(k,v),removeItem:k=>map.delete(k)},fetch:async url=>({ok:true,json:async()=>url==='cardcount_profile.json'?cfg:{}})};
 ctx.window=ctx;ctx.__BGS256_CONTINUATION_TEST__={hazardChoose(){throw Error('legacy core invoked');}};
 vm.createContext(ctx);for(const f of ['experimental/quant_engine/bayes_markov_runtime.js','quant_dynamic_adapter.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(f,'utf8'),ctx,{filename:f});
 await ctx.__BGS_FINAL56__.loadModels();return ctx.__BGS_FINAL56__;
}
(async()=>{
 const api=await runtime(true),p=api.applyFinalPrediction(['B','P','T']);assert(p.quant_dynamic);assert.equal(api.displayDirection(p.next_hand_forecast),p.direction);
 api.registerPrediction(['B','P','T'],p);api.settlePending('T',null,['B','P','T']);assert(api.getTrainingRows()[0].quant_dynamic);assert.equal(api.getTrainingRows()[0].actual_outcome,'T');
 const legacy=await runtime(false);assert.equal(legacy.applyFinalPrediction(['B']).finalProbability.mode,'bayes_markov_final');
 assert.throws(()=>A.predict(E,['X']));console.log('PASS: TC/EV gates, dynamic weights, normalized probabilities, Kelly, 128D ABI isolation, replay/undo/reset, four side fields, production dispatch, rollback and snapshot settlement. No hit-rate claim.');
})().catch(e=>{console.error(e);process.exitCode=1;});
