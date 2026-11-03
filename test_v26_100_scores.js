"use strict";
/**
 * Full 0..9 x 0..9 final-score diagnostic, one new PF256 per independent case.
 * No direction balancing; print the actual Banker/Player distribution.
 * Compare prior V2.5 1/20-shoe smoothing against V2.6 from the SAME posterior
 * and MC draws, avoiding an unfair RNG comparison.
 */
const assert=require("node:assert/strict");
const {PhysicsForecastFirstEngine,BASE}=require("./physics_forecast_first_runtime.js");
const rows=[];
const fields={total:0,bankerNew:0,playerNew:0,bankerLegacy:0,playerLegacy:0,
  directionChanges:0,minBlend:Infinity,maxBlend:0,sumBlend:0,
  minESS:Infinity,maxESS:0,minGap:Infinity,maxGap:-Infinity};
for(let player=0;player<=9;player++)for(let banker=0;banker<=9;banker++){
  const outcome=banker>player?"B":player>banker?"P":"T";
  const engine=new PhysicsForecastFirstEngine({
    simulationsPerParticle:160,threshold:.005,lossGate:.035,
    quantOptions:{lambda:.82,alpha:.15,particles:256},
    independentFinalScoreMode:true
  });
  const r=engine.syncHistory([outcome],[{player,banker}]);
  assert.equal(r.history_length,1);
  assert.equal(r.point_score_observation_count,1);
  assert.equal(engine.quant.physics.size,256);
  assert.equal(r.physical_ev.samples,40960);
  assert.equal(r.bet_allowed,false);
  assert.equal(r.action,"SKIP","final scores are not direct observed-card evidence");
  assert(!r.physical_ev.simulation_exhausted);
  assert.equal(r.early_shoe_guard?.mode,"independent_final_score_quality");
  const d=r.early_shoe_guard;
  assert(d.blend>=.28-1e-12&&d.blend<=.82+1e-12);
  const prob=r.physical_ev.simulated_probability;
  assert(Math.abs(prob.B+prob.P+prob.T-1)<1e-8);
  assert(Math.abs(r.ev_banker-(.95*prob.B-prob.P))<1e-10);
  assert(Math.abs(r.ev_player-(prob.P-prob.B))<1e-10);
  const rawB=BASE.B+(prob.B-BASE.B)/d.blend;
  const rawP=BASE.P+(prob.P-BASE.P)/d.blend;
  const oldProgress=.05;
  const oldBase=.12+.88*oldProgress**1.25;
  const oldBlend=oldBase*(1-.25*(1-oldProgress)*(1-d.quality));
  assert(d.blend>oldBlend,
    "quality-adaptive mode should relieve the fixed first-hand 86% prior blend");
  const oldB=BASE.B+oldBlend*(rawB-BASE.B);
  const oldP=BASE.P+oldBlend*(rawP-BASE.P);
  const oldDir=.95*oldB-oldP >=oldP-oldB?"B":"P";
  const newDir=r.ev_banker>=r.ev_player?"B":"P";
  // Reconstruct unsmoothed physical EV from the same posterior and draws,
  // without treating final totals as observed card ranks.
  const rawEvB=.95*rawB-rawP,rawEvP=rawP-rawB;
  const oldEvB=.95*oldB-oldP,oldEvP=oldP-oldB;
  assert(Number.isFinite(rawEvB)&&Number.isFinite(rawEvP));
  assert(Number.isFinite(oldEvB)&&Number.isFinite(oldEvP));
  const round=x=>Number(x.toFixed(10));
  rows.push({player,banker,outcome,
    raw_ev_banker:round(rawEvB),raw_ev_player:round(rawEvP),
    v25_ev_banker:round(oldEvB),v25_ev_player:round(oldEvP),
    v26_ev_banker:round(r.ev_banker),v26_ev_player:round(r.ev_player),
    raw_direction:rawEvB>=rawEvP?"B":"P",
    v25_direction:oldDir,final_direction:newDir,
    blend:round(d.blend),update_ess:round(r.physical_ev.posterior_update_ess??256)});

  fields.total++;
  fields[newDir==="B"?"bankerNew":"playerNew"]++;
  fields[oldDir==="B"?"bankerLegacy":"playerLegacy"]++;
  if(oldDir!==newDir)fields.directionChanges++;
  fields.minBlend=Math.min(fields.minBlend,d.blend);
  fields.maxBlend=Math.max(fields.maxBlend,d.blend);
  fields.sumBlend+=d.blend;
  fields.minESS=Math.min(fields.minESS,r.physical_ev.posterior_update_ess??256);
  fields.maxESS=Math.max(fields.maxESS,r.physical_ev.posterior_update_ess??0);
  const gap=r.ev_banker-r.ev_player;
  fields.minGap=Math.min(fields.minGap,gap);
  fields.maxGap=Math.max(fields.maxGap,gap);
}
assert.equal(fields.total,100);
assert.equal(rows.length,100);
assert.equal(fields.bankerNew+fields.playerNew,100);
assert.equal(fields.bankerLegacy+fields.playerLegacy,100);
fields.meanBlend=fields.sumBlend/fields.total;
console.log("PASS V2.6 all 100 independent Player×Banker scores:",
  JSON.stringify(fields,null,2));
// Publish 100 comparable rows in CI logs, not just aggregate counts.
// EV values are decimal per-unit stakes and are not hit rates.
const columns=["player","banker","outcome","raw_ev_banker","raw_ev_player",
  "v25_ev_banker","v25_ev_player","v26_ev_banker","v26_ev_player",
  "raw_direction","v25_direction","final_direction","blend","update_ess"];
console.log("AUDIT V2.6 all 100 scores: raw EV, old smoothed EV, new smoothed EV and final direction (CSV)");
console.log(columns.join(","));
for(const row of rows)console.log(columns.map(k=>row[k]).join(","));
// A more balanced 50:50 direction mix is NOT a goal or passing criterion.
