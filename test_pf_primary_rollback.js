'use strict';
const assert=require('assert/strict'),{make}=require('./test_pf_primary_runtime');
(async()=>{
  // The old 1eb641e fixture remains archival. Its exact probabilities cannot be
  // compared to today's different Physics/PF model and profile parameters.
  const {api}=await make({legacy:false}),config=api.getCardcountProfile();
  assert(api.pfPrimaryConfig()?.enabled,'current default must be PF-primary');
  config.pf_primary_v1.enabled=false;
  config.anti_streak_v1.enabled=false;
  assert.equal(api.pfPrimaryConfig(),null);
  assert.equal(api.antiStreakConfig(),null);
  for(const h of ['', 'BPTBP', 'P'.repeat(5), 'B'.repeat(5), 'BP'.repeat(35)]){
    const p=api.applyFinalPrediction(h.split('')),r=p.finalProbability;
    assert.equal(r.mode,'physics_primary');
    assert.equal(r.anti_streak_enabled,false);
    assert.equal(p.opposite_blocked,false);
    assert(!r.execution_order.includes('anti_streak_qualification'));
    assert(r.execution_order.includes('direct_physics'));
    assert(!r.execution_order.includes('pf_primary_posterior'));
    assert(['B','P','Skip'].includes(p.bet_direction));
    assert.equal(p.bet_direction,p.direction);
    if(r.active){assert(Number.isFinite(r.finalPB));assert.equal(r.physicsDirect.length,48);assert.equal(r.extended.length,57);}
  }
  config.pf_primary_v1.enabled=true;
  const again=api.applyFinalPrediction('PPPPP'.split(''));
  assert.equal(again.finalProbability.mode,'pf_primary');
  assert.equal(again.finalProbability.anti_streak_enabled,false);
  console.log('PASS: rollback switches actually restore Physics-primary with legacy EV/Skip guards and no anti-streak block; PF re-enable works.');
})().catch(e=>{console.error(e);process.exitCode=1});
