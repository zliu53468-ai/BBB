const assert=require('assert/strict'),{runtime,settle}=require('./priority_runtime.cjs');
(async()=>{
 const old=await runtime({revision:'116ffc97b62f2519659d5ceb5ef5d717da0b8c7b'}),next=await runtime();
 const c=next.api.getCardcountProfile().physics_priority_v1.early_mid_conservative_v2;c.enabled=false;
 for(const h of ['BPTBP','BBPP'.repeat(8)]){const a=old.api.applyFinalPrediction(h.split('')),b=next.api.applyFinalPrediction(h.split(''));assert.equal(a.direction,b.direction);assert.equal(a.finalProbability.finalPB,b.finalProbability.finalPB);}
 c.enabled=true;next.data.clear();next.data.set('bgs_xgb_final_shoe_id_v5','v2-points-test');const history=[];
 for(let i=0;i<8;i++){
  const p=next.api.applyFinalPrediction(history);assert(p.physics_priority.alpha<=.4);assert.equal(p.physics_priority.point_weight,0);assert(p.finalProbability.particleDiagnostics.blend_weight<=.08+1e-12);
  settle(next,history,p,'B',{outcome:'B',player_score:2,banker_score:6});history.push('B');
 }
 console.log('Actual runtime: v2-off v1 parity; early PF budget includes optional points; alpha<=.40 passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
