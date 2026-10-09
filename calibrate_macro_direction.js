#!/usr/bin/env node
"use strict";
/* Causal, shoe-disjoint Physical EV macro direction calibration.
 * node calibrate_macro_direction.js --synthetic 3000 [report.json]
 * node calibrate_macro_direction.js completed_hands.json [report.json]
 * Inference coefficients are proposals ONLY, never auto-deployed.
 */
const fs = require("node:fs");
const Macro = require("./macro_ema.js");
const NAMES = [["Six_Card", "six_card"], ["Low_Score", "low_score"], ["Point_Diff", "point_diff"]];
function randomSource(seed=20261009) {
  let s=seed>>>0;
  return ()=>{s=(s+0x6D2B79F5)|0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;};
}
function synthetic(shoeCount,seed=20261009) {
  if(!Number.isInteger(shoeCount)||shoeCount<5||shoeCount>100000)throw new Error("synthetic shoes must be 5..100000");
  const rand=randomSource(seed),base=[];
  for(let deck=0;deck<8;deck++)for(let suit=0;suit<4;suit++)for(let rank=1;rank<=13;rank++)base.push(rank>=10?0:rank);
  const rows=[];
  for(let id=0;id<shoeCount;id++) {
    const cards=base.slice();
    for(let i=cards.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[cards[i],cards[j]]=[cards[j],cards[i]];}
    let cursor=0,round=0;
    while(cards.length-cursor>66&&round<70) {
      let p=(cards[cursor]+cards[cursor+2])%10,b=(cards[cursor+1]+cards[cursor+3])%10;
      cursor+=4;let third=null,count=4;
      if(p<8&&b<8) {
        if(p<=5){third=cards[cursor++];p=(p+third)%10;count++;}
        const draws=third===null?b<=5:
          b<=2||(b===3&&third!==8)||(b===4&&third>=2&&third<=7)||
          (b===5&&third>=4&&third<=7)||(b===6&&third>=6&&third<=7);
        if(draws){b=(b+cards[cursor++])%10;count++;}
      }
      rows.push({shoe_id:"sim-"+String(id).padStart(6,"0"),round_index:++round,
        outcome:b>p?"B":p>b?"P":"T",card_count:count,player_score:p,banker_score:b});
    }
  }
  return rows;
}
function moments(){return {n:0,sx:0,sy:0,sxx:0,syy:0,sxy:0};}
function add(m,x,y){m.n++;m.sx+=x;m.sy+=y;m.sxx+=x*x;m.syy+=y*y;m.sxy+=x*y;}
function combine(m,z){m.n+=z.n;m.sx+=z.sx;m.sy+=z.sy;m.sxx+=z.sxx;m.syy+=z.syy;m.sxy+=z.sxy;}
function correlation(m){
  if(m.n<3)return null;
  const vx=m.sxx-m.sx*m.sx/m.n,vy=m.syy-m.sy*m.sy/m.n,cov=m.sxy-m.sx*m.sy/m.n;
  return vx>1e-12&&vy>1e-12?cov/Math.sqrt(vx*vy):null;
}
function percentile(sorted,q){if(!sorted.length)return null;const x=q*(sorted.length-1),i=Math.floor(x),t=x-i;
  return sorted[i]*(1-t)+sorted[Math.min(i+1,sorted.length-1)]*t;}
function bootstrap(items,seed=20261009,iterations=400){
  if(items.length<2)return [null,null];
  const random=randomSource(seed),rs=[];
  for(let i=0;i<iterations;i++){
    const total=moments();
    for(let k=0;k<items.length;k++)combine(total,items[Math.floor(random()*items.length)]);
    const r=correlation(total);if(r!==null)rs.push(r);
  }
  rs.sort((a,b)=>a-b);
  return [percentile(rs,.025),percentile(rs,.975)];
}
function summarize(groups,feature,seed){
  const entries=groups.map(g=>g.stats[feature]).filter(m=>m.n>0),all=moments();
  for(const m of entries)combine(all,m);
  const r=correlation(all),ci=bootstrap(entries,seed);
  return {shoes:entries.length,observations:all.n,pearson_r:r,cluster_bootstrap_95pct:ci,
    direction:r===null?"unavailable":r>0?"B_positive":"P_negative"};
}
function build(rows) {
  if(!Array.isArray(rows)||!rows.length)throw new Error("completed-hands rows required");
  const groups=new Map();
  for(const row of rows){
    if(row.shoe_id===null||row.shoe_id===undefined)throw new Error("shoe_id required");
    const id=String(row.shoe_id),group=groups.get(id)||{id,rows:[],stats:NAMES.map(()=>moments())};
    if(row.round_index!==group.rows.length+1)throw new Error(id+" requires consecutive round_index including ties");
    group.rows.push(Macro.normalizeHand(row));groups.set(id,group);
  }
  for(const group of groups.values()) {
    const state=Macro.createState(group.id);
    for(const hand of group.rows) {
      const pre=Macro.snapshot(state);
      if(hand.outcome!=="T"){
        const label=hand.outcome==="B"?1:0;
        for(let k=0;k<NAMES.length;k++){
          const key=NAMES[k][0],spread=pre[key+"_Spread"];
          if(Number.isFinite(spread)&&pre.counts[key]>=12&&pre.ages[key]<=3)
            add(group.stats[k],k===2?spread/9:spread,label);
        }
      }
      Macro.update(state,hand); // Never use this hand's observations before prediction.
    }
  }
  return [...groups.values()];
}
function eligible(stats){
  const {observations:n,shoes,pearson_r:r,cluster_bootstrap_95pct:ci}=stats;
  return n>=2000&&shoes>=30&&r!==null&&Math.abs(r)>=.01&&ci.every(Number.isFinite)&&
    (ci[0]>0&&r>0||ci[1]<0&&r<0);
}
function report(rows,dataKind) {
  const groups=build(rows),count=groups.length;
  if(count<5)throw new Error("At least 5 complete independent shoes are needed; 30+ per split recommended");
  const cut1=Math.floor(count*.6),cut2=Math.floor(count*.8);
  const splits={train:groups.slice(0,cut1),calibration:groups.slice(cut1,cut2),holdout:groups.slice(cut2)};
  const analysis={},candidate={},monitor={};
  for(let k=0;k<3;k++){
    const key=NAMES[k][1];
    const train=summarize(splits.train,k,20261009+k*11);
    const calibration=summarize(splits.calibration,k,20261019+k*11);
    const holdout=summarize(splits.holdout,k,20261029+k*11);
    const agreed=eligible(train)&&eligible(calibration)&&Math.sign(train.pearson_r)===Math.sign(calibration.pearson_r);
    const strength=agreed?(Math.min(Math.abs(train.pearson_r),Math.abs(calibration.pearson_r))>=.03?.02:.01):0;
    candidate[key]=agreed?Math.sign(train.pearson_r)*strength:0;
    monitor[key]=agreed?
      (Math.sign(holdout.pearson_r)===Math.sign(candidate[key])?"holdout_direction_agrees":"holdout_direction_disagrees"):
      "not_selected_on_train_calibration";
    analysis[key]={train,calibration,holdout,train_calibration_eligible:agreed};
  }
  return {schema:"macro_direction_calibration_v1",data_kind:dataKind,
    timing:"snapshot prior to current outcome; current hand updates EMA afterwards",
    splits:"entire shoes; chronological first 60% train, next 20% calibration, latest 20% holdout",
    shoes:count,completed_hands:rows.length,
    feature_normalization:{six_card:"spread",low_score:"spread",point_diff:"spread / 9"},
    bootstrap:"400 seeded shoe-cluster samples per feature per split; exploratory uncertainty intervals",
    selection:"train/calibration only: >=30 shoes and >=2000 samples each, |r|>=0.01, 95% cluster CI excludes 0, same sign; .01 or .02 magnitude",
    analysis,candidate_coefficients:candidate,holdout_monitor:monitor,
    production_activation_recommended:false,
    warning:"Simulation/observational correlation does not establish causal or profitable prediction. Leave production coefficients zero until unseen complete-shoe replay and economic guardrails pass. Do not tune on holdout."};
}
function main(){
  const args=process.argv.slice(2);let rows,kind,out;
  if(args[0]==="--synthetic"){const n=Number(args[1]||3000);rows=synthetic(n);kind="synthetic_standard_8_deck_seed_20261009";out=args[2];}
  else if(args.length){const payload=JSON.parse(fs.readFileSync(args[0],"utf8"));
    rows=Array.isArray(payload)?payload:payload.rows;kind=payload.data_kind||"completed_observed_hands";out=args[1];}
  else {console.error("Usage: node calibrate_macro_direction.js [--synthetic N | completed_hands.json] [report.json]");process.exit(2);}
  const result=report(rows,kind),json=JSON.stringify(result,null,2)+"\n";
  if(out)fs.writeFileSync(out,json);console.log(json.trim());
}
if(require.main===module)main();
module.exports={synthetic,report,build,correlation};
