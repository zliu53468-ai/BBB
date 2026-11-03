"use strict";
/* V2.4 WEB ONLY: preserve all real model logic, reset PF for EVERY point pair. */
const assert=require("node:assert/strict"),fs=require("node:fs");
const {PhysicsForecastFirstEngine,BASE_EV_GATE,OVERDRIVE_GATE,
  ONE_HAND_CUSHION}=require("./physics_forecast_first_runtime.js");
const {BlindPhysicsEstimator}=require("./quant256_online_runtime.js");
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const samples=[];
function single(player,banker){
  const fresh=new PhysicsForecastFirstEngine({
    threshold:.005,lossGate:.035,simulationsPerParticle:160,
    observedSamples:12000,quantOptions:{lambda:.82,alpha:.15,particles:256}
  });
  const outcome=banker>player?"B":player>banker?"P":"T";
  const report=fresh.syncHistory([outcome],[{player,banker}]);
  samples.push({fresh,report});
  return{fresh,report};
}
const a=single(7,8),b=single(8,2),again=single(7,8);
assert.notStrictEqual(a.fresh,b.fresh,"each scored hand owns a NEW model");
assert.notStrictEqual(a.fresh.quant.physics,b.fresh.quant.physics,
  "particles cannot carry into the next scored round");
assert.notStrictEqual(a.fresh.quant.physics.particles,b.fresh.quant.physics.particles);
assert.deepEqual(a.report,again.report,
  "same score always gets a reproducible new 256-particle posterior");
assert.equal(a.fresh.history.length,1);
assert.equal(b.fresh.history.length,1);
assert.equal(b.fresh.scoreRows.length,1);
assert.equal(b.fresh.scoreRows[0].player,8);
assert.equal(b.fresh.scoreRows[0].banker,2);
assert.equal(b.report.history_length,1);
assert.equal(a.report.point_score_observation_count,1);
assert.equal(b.report.point_score_observation_count,1);
assert.equal(a.report.physical_ev.samples,256*160);
assert.equal(b.report.physical_ev.samples,256*160);
assert.equal(a.fresh.quant.physics.size,256);
assert.equal(b.fresh.quant.physics.size,256);
assert.equal(new BlindPhysicsEstimator().size,256);
assert.equal(BASE_EV_GATE,.005);
assert.equal(OVERDRIVE_GATE,0);
assert.equal(ONE_HAND_CUSHION,.035);
assert.equal(b.report.research_signal.direction,
  b.report.ev_banker>=b.report.ev_player?"B":"P");
assert.equal(b.report.action,"SKIP",
  "final totals are not actual physical card counts and cannot qualify as BET");
near(b.report.ev_banker,b.report.physical_ev.ev_banker);
near(b.report.ev_player,b.report.physical_ev.ev_player);
assert.deepEqual(b.report.feature_weights,{physics:5,road:.1});
assert.equal(b.report.markov_residual.stats.selected_support,0,
  "independent one-round mode must never present historical transition evidence");
const panel=fs.readFileSync("quant256_online_panel.js","utf8");
const page=fs.readFileSync("index.html","utf8");
const css=fs.readFileSync("styles.css","utf8");
assert(panel.includes("const fresh=new PhysicsForecastFirstEngine(CONFIG)"));
assert(panel.includes("fresh.syncHistory([outcome]"));
assert(!panel.includes("model.syncHistory(seq"),
  "web panel must not replay cumulative historical sequences");
assert(page.includes('id="bbbPointEntry" aria-label'));
assert(page.indexOf('id="pointPlayerFinal"')<
  page.indexOf('id="pointBankerFinal"'),"left Player right Banker");
assert(css.includes("#predictionApp .bbb-results-block,")&&
  css.includes("#predictionApp #btnStart,"),"hide historical input/Start");
console.log("PASS V2.4 independent PF256 x3: one final-score pair per new model, N/L/M retained, deterministic same score, no historical Markov or BET override");
