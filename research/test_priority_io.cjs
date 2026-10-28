const assert=require('assert/strict'),fs=require('fs'),{runtime,settle}=require('./priority_runtime.cjs');
(async()=>{
 const shoes=fs.readFileSync('research/data/priority_v1/smoke-observations.jsonl','utf8').trim().split('\n').slice(0,3).map(JSON.parse);
 let cases=0;for(const legacy of [true,false]){
  const a=await runtime({legacy,fastIO:false}),b=await runtime({legacy,fastIO:true});
  for(const shoe of shoes){for(const r of [a,b]){r.data.clear();r.data.set('bgs_xgb_final_shoe_id_v5',shoe.shoe_id);}const history=[];
   for(const row of shoe.rows){const x=a.api.applyFinalPrediction(history);b.win.__labPosterior=a.win.__labPosterior;const y=b.api.applyFinalPrediction(history);
    for(const k of ['direction','probabilities','ev_banker','ev_player'])assert.deepEqual(x[k],y[k],'canonical journal IO parity '+k);
    assert.equal(x.finalProbability.finalPB,y.finalProbability.finalPB);
    if(!legacy)assert.deepEqual(x.physics_priority,y.physics_priority);
    for(const [r,p]of [[a,x],[b,y]])settle(r,history,p,row.outcome,{outcome:row.outcome});
    assert.equal(a.api.getTrainingCount(),b.api.getTrainingCount());history.push(row.outcome);cases++;
   }
  }
 }
 fs.writeFileSync('research/reports/priority_v1_smoke/io-parity.json',JSON.stringify({passed:true,cases,versions:['original full formal runtime','new gated runtime'],same_math:true,scope:'all recommendation/probability/EV/gate fields across3 complete shoes per version'}));console.log('Canonical IO exact parity',cases);
})().catch(e=>{console.error(e);process.exitCode=1;});
