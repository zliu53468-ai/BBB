#!/usr/bin/env node
"use strict";
// Run against an untouched holdout replay comparing current and original main.
const fs=require("node:fs");
function assess(report,{minimumShoes=100,minimumBets=500,minSkipReduction=0,maxSkipIncrease=.05}={}){
  const failures=[];
  if(!Number.isInteger(minimumShoes)||minimumShoes<1||!Number.isInteger(minimumBets)||minimumBets<1||
    !Number.isFinite(minSkipReduction)||minSkipReduction<0||minSkipReduction>1||!Number.isFinite(maxSkipIncrease)||maxSkipIncrease<0||maxSkipIncrease>.05)throw Error("Invalid acceptance configuration");
  if(report.data_kind!=="real_holdout_observations")failures.push("requires real_holdout_observations; synthetic/unknown data cannot establish quality");
  if(!(report.shoes>=minimumShoes))failures.push("insufficient holdout shoes");
  if(report.current?.no_flip_violations!==0)failures.push("no-flip audit failed");
  for(const stage of ["all","early","middle","late"]){
    const a=report.current?.metrics?.[stage],b=report.baseline?.metrics?.[stage];
    if(!a||!b){failures.push(stage+": baseline/current metrics missing");continue;}
    if(!(a.bets>=minimumBets&&b.bets>=minimumBets))failures.push(stage+": insufficient bets");
    for(const metric of ["brier","hit_rate_on_bets","realized_ev_per_bet","skip_rate"])
      if(!Number.isFinite(a[metric])||!Number.isFinite(b[metric]))failures.push(stage+": invalid "+metric);
    if(a.hit_rate_on_bets<b.hit_rate_on_bets)failures.push(stage+": hit rate declined");
    if(a.realized_ev_per_bet<b.realized_ev_per_bet)failures.push(stage+": realized EV declined");
    if(a.brier>b.brier+.0015)failures.push(stage+": Brier regressed beyond tolerance");
    if(a.skip_rate-b.skip_rate>maxSkipIncrease+1e-12)failures.push(stage+": Skip increased beyond 5pp ceiling");
    if(stage==="all"&&minSkipReduction>0&&!(b.skip_rate-a.skip_rate>=minSkipReduction))failures.push("overall Skip reduction below target");
  }
  return {passed:failures.length===0,failures,minimumShoes,minimumBets,minSkipReduction,maxSkipIncrease,
    limitation:"Observed non-regression only; data provenance and an untouched holdout must be independently verified. No future-performance guarantee."};
}
if(require.main===module){
  const result=assess(JSON.parse(fs.readFileSync(process.argv[2],"utf8")));
  console.log(JSON.stringify(result,null,2));process.exitCode=result.passed?0:1;
}
module.exports={assess};
