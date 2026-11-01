"use strict";
// Quant256 V3 完整線上回放與瀏覽器按鈕接線測試。
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const {OnlineLinUCB256,BlindPhysicsEstimator,context256,windowStats,ACTIVE_INDICES,VERSION}=
  require("./quant256_online_runtime");
const near=(a,b,tol=1e-9)=>assert(Math.abs(a-b)<=tol,"expected "+a+" ~ "+b);
assert.equal(VERSION,"BBB_QUANT256_ONLINE_V3_HYPER_MICRO_PAIRWISE");
assert.equal(ACTIVE_INDICES.length,36);
assert.equal(new OnlineLinUCB256().lambda,.82);
for(const l of [.75,.82,.94,.96])near(new OnlineLinUCB256({lambda:l}).lambda,l);
assert.throws(()=>new OnlineLinUCB256({lambda:.74}),/lambda/);
assert.throws(()=>new OnlineLinUCB256({lambda:.97}),/lambda/);
assert.throws(()=>new OnlineLinUCB256({lossGate:-1}),/loss_gate/);
const q=new OnlineLinUCB256(),p0=q.predict();
assert.equal(p0.context.length,256);assert.equal(p0.shoe_context.length,128);
assert.equal(p0.road_context.length,128);assert.equal(p0.dynamic_gate,0);
assert.equal(p0.direction,"B");assert.equal(p0.bet_allowed,true);
assert.equal(p0.reason,"neutral_default_signal");
assert.equal(p0.calibrated_probability,false);assert.equal(p0.claimed_edge,false);
assert.equal(p0.blind_physics.actual_draw_count_known,false);
assert.equal(p0.blind_physics.actual_remaining_known,false);
assert.equal(p0.blind_physics.source,"simulated_unobserved");
assert.throws(()=>q.observe("X"),/invalid_outcome/);
// 初始無資訊，初次 +1/-2 更新的 x 向量為零；下一次才有可學習的牌路。
const x0=p0.context.map((x,i)=>ACTIVE_INDICES.includes(i)?x:0);
q.observe("P"); // predict("B") 錯誤，下局應暫時閘門 .04
assert.equal(q.gateCooldown,.04);assert.equal(q.last_settlement,"loss");
assert.equal(q.trainedRounds,1);assert.equal(q.prequential.wrong,1);
assert.equal(q.predict().dynamic_gate,.04);
const aBefore=q.arms.B.A.map(row=>Array.from(row));
const bBefore=Array.from(q.arms.B.b);
const x1=Float64Array.from(ACTIVE_INDICES,i=>q.predict().context[i]);
assert.equal(q.predict().direction,"Skip"); // 上局僅一筆資料，分數仍相等
q.observe("B"); // 觀望也按 B/P 結算，以正負標籤更新，但不視為真實命中
assert.equal(q.gateCooldown,0);
assert.equal(q.last_settlement,"skipped");
assert.equal(q.trainedRounds,2);
for(let i=0;i<36;i++){
  near(q.arms.B.b[i],.82*bBefore[i]+x1[i]);
  near(q.arms.P.b[i],-2*x1[i]); // 第一局輸入的零向量不產生非零 b
  for(let j=0;j<36;j++){
    near(q.arms.B.A[i][j],.82*aBefore[i][j]+(i===j?.18:0)+x1[i]*x1[j]);
    near(q.arms.B.A[i][j],q.arms.P.A[i][j]);
  }
}
assert.notEqual(q.arms.B.b[12],-q.arms.P.b[12],
  "雙倍懲罰令 b 非反對稱；不能誤稱 b_B=-b_P");
// 驗證開莊正回饋 +1/開閒 -2；非和局先衰減而且雙臂 A 同步。
const x2=Float64Array.from(ACTIVE_INDICES,i=>q.predict().context[i]);
const beforeB=Array.from(q.arms.B.b),beforeP=Array.from(q.arms.P.b);
q.observe("P");
for(let i=0;i<36;i++){
  near(q.arms.B.b[i],.82*beforeB[i]-2*x2[i]);
  near(q.arms.P.b[i],.82*beforeP[i]+x2[i]);
}
const trained=q.trainedRounds,prevB=Array.from(q.arms.B.b);
q.observe("T");
assert.equal(q.trainedRounds,trained,"Tie 不可用 B/P 假標籤更新");
for(let i=0;i<36;i++)near(q.arms.B.b[i],.82*prevB[i]);
assert.equal(q.prequential.eligible,
  q.prequential.correct+q.prequential.wrong+q.prequential.ties);
const result=q.syncHistory(["P","B","P","T"]);
assert.equal(result.history_length,4);assert.equal(q.trainedRounds,3);
q.syncHistory(["P","B"]);
assert.equal(q.trainedRounds,2,"返回上一局需回放而不重複學習");
q.syncHistory([]);
assert.equal(q.trainedRounds,0);assert.equal(q.gateCooldown,0);
const blind=new BlindPhysicsEstimator({particles:12});
assert.equal(blind.snapshot().actual_draw_count_known,false);
for(const r of ["B","P","T","B","P"])blind.observe(r);
assert(blind.snapshot().virtual_remaining<416);
const h="BBBPBBPPTBPPBBTPBPBBPP".split("");
const c=context256(h,blind);
assert.equal(c.vector.length,256);
assert(c.vector.every(v=>Number.isFinite(v)&&v>=-1&&v<=1));
assert(c.vector.slice(12,128).every(v=>v===0));
assert(c.vector.slice(128+24).every(v=>v===0));
assert.equal(c.features.maximum_window,16);
near(c.features.turn_rate_delta,windowStats(h,2).turn-windowStats(h,5).turn);
near(c.features.regularity_delta,windowStats(h,2).regularity-windowStats(h,5).regularity);
near(c.features.entropy_delta,windowStats(h,2).entropy-windowStats(h,5).entropy);
const initial=context256([],new BlindPhysicsEstimator());
assert.equal(initial.vector.every(v=>v===0),true,"不可注入固定截距/單邊先驗");
const inverted=h.map(v=>v==="B"?"P":v==="P"?"B":"T");
const ci=context256(inverted,blind);
for(const i of [0,11,12,13,14,21])near(c.road[i],-ci.road[i]);
near(c.shoe[0],-ci.shoe[0]);
near(c.shoe[1],-ci.shoe[1]);
// 人工固定 UCB 差 .02，驗證失誤後只有一個非和局觸發 .04 観望。
const scripted=new OnlineLinUCB256();
scripted._armScore=arm=>({mean:arm==="B"?.02:0,uncertainty:0,ucb:arm==="B"?.02:0});
assert.equal(scripted.predict().direction,"B");assert.equal(scripted.predict().dynamic_gate,0);
scripted.observe("P"); // B 錯
assert.equal(scripted.predict().dynamic_gate,.04);
assert.equal(scripted.predict().direction,"Skip");
scripted.observe("T"); // 和局不消耗 B/P 錯誤冷卻
assert.equal(scripted.predict().dynamic_gate,.04);
scripted.observe("B"); // 觀望當局結束，冷卻解除
assert.equal(scripted.predict().dynamic_gate,0);
assert.equal(scripted.predict().direction,"B");
scripted.observe("B");
assert.equal(scripted.predict().dynamic_gate,0,"命中立即維持零門檻");
scripted.observe("P");
assert.equal(scripted.predict().dynamic_gate,.04,"下一次失誤立即變嚴格");
scripted.syncHistory(["P","T","B","B","P"]);
assert.equal(scripted.gateCooldown,.04,"單純重讀歷史不能改動動態門檻");
const fresh=OnlineLinUCB256.fromHistory(h);
near(fresh.ucb_banker,OnlineLinUCB256.fromHistory(h).ucb_banker);
near(fresh.delta,OnlineLinUCB256.fromHistory(h).delta);
// 高密度下即使表現不好也不允許出現多個連續無方向的非和局。
const dense=new OnlineLinUCB256();
const rounds=Array.from({length:60},(_,i)=>i%3===0?"B":"P");
dense.syncHistory(rounds);
assert(dense.prequential.eligible>=30,"實驗性高頻決策至少半數有方向");
assert.equal(dense.prequential.eligible+dense.prequential.skipped,60);
assert.equal(dense.hitRate,undefined,"不應把測試輸出宣稱為保證勝率");
assert(dense.predict().signal_rate>=.5);

// DOM 操作測試：原本 B/P/T 儲存先執行，然後 V3 接管顯示；Start 防止 Frozen click。
const listeners={},dom={};
const ids=["btnStart","btnB","btnP","btnT","btnBack","btnEnd","directionText",
  "directionCode","directionOrb","confidence","regime","strength","modePill","message"];
for(const id of ids){
  listeners[id]=[];dom[id]={textContent:"",className:"",
    addEventListener(type,fn,capture){listeners[id].push({fn,capture:!!capture});}};
}
const KEY="bgs256d_frozen_6x15_forward_v18",
 LEGACY="bgs256d_128plus128_frozen_direct_tech_panel_v1";
let saved=[],frozenClicks=0;
const data=new Map([[KEY,JSON.stringify({history:saved})]]);
const storage={getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)};
for(const v of ["B","P","T"])dom["btn"+v].addEventListener("click",()=>{
  saved.push(v);storage.setItem(KEY,JSON.stringify({history:saved}));
});
dom.btnStart.addEventListener("click",()=>frozenClicks++);
dom.btnBack.addEventListener("click",()=>{
  saved.pop();storage.setItem(KEY,JSON.stringify({history:saved}));
});
function click(id){
  let stop=false;
  const e={preventDefault(){},stopImmediatePropagation(){stop=true;}};
  for(const handler of [...listeners[id]].sort((a,b)=>Number(b.capture)-Number(a.capture))){
    if(stop)break;handler.fn(e);
  }
}
const win={document:{getElementById:id=>dom[id]||null},localStorage:storage,
  location:{search:""},BBBQuant256Online:require("./quant256_online_runtime"),
  BBBPhysicsForecastFirst:require("./physics_forecast_first_runtime")};
vm.runInNewContext(fs.readFileSync("./quant256_online_panel.js","utf8"),
  {window:win,document:win.document,URLSearchParams,console});
const ui=win.__BBB_QUANT256_ONLINE_PANEL__;
assert.equal(ui.configuration.quantOptions.lambda,.82);
assert.equal(ui.configuration.threshold,.010);
assert.equal(ui.configuration.lossGate,.035);
assert.equal(ui.version,"BBB_PHYSICS_FORECAST_FIRST_V3_MICRO_FUSION");
click("btnB");click("btnP");click("btnT");
assert.equal(ui.model.history.join(""),"BPT");
click("btnStart");
assert.equal(frozenClicks,0);
assert(dom.modePill.textContent.includes("V3 BALANCED"));
assert(["莊","閒","觀望"].includes(dom.directionText.textContent));
assert(!dom.directionText.textContent.includes("研究"),
  "最終方向須移除後面的（研究）兩個字");
assert(["莊","閒","觀望"].includes(dom.directionText.textContent));
if(["莊","閒"].includes(dom.directionText.textContent))
  assert(/BANKER|PLAYER/.test(dom.directionCode.textContent));
assert(dom.regime.textContent.includes("未確認正 EV"));
assert(dom.confidence.textContent.includes("非勝率"));
click("btnBack");assert.equal(ui.model.history.join(""),"BP");
data.delete(KEY);data.set(LEGACY,JSON.stringify({history:["T","P","B"]}));
assert.equal(ui.history().join(""),"TPB","must migrate legacy 128+128 local key");
assert.equal(ui.run().history_length,3);
click("btnEnd");assert.equal(ui.model.trainedRounds,0);
assert.equal(ui.run().action,"SKIP");
assert(["B","P"].includes(ui.run().research_signal.direction));
assert([0,.010,.035].includes(ui.run().dynamic_gate));
// 盲估每局根據合法補牌結果抽樣，不可憑和局任意兩倍扣除特定牌。
const pair=new BlindPhysicsEstimator({particles:12});
const original=pair.particles.map(c=>c.slice());
pair.observe("T");
const pairStats=pair.snapshot().pairwise_depletion;
assert.equal(pairStats.observed_cards,false);
assert.equal(pairStats.source,"unconditional_shoe_sampling");
assert(pairStats.estimated_draws>=4&&pairStats.estimated_draws<=6);
assert(pairStats.estimated_pairwise_key_copresence>=0);
assert(pairStats.estimated_pairwise_key_copresence<=1);
assert(pairStats.tie_or_long_hands<=pairStats.matched_particles);
for(let i=0;i<pair.particles.length;i++){
  const removed=original[i].reduce((s,x,j)=>s+x-pair.particles[i][j],0);
  assert(removed===0||(removed>=4&&removed<=6),"unconditional legal hand must remove exactly 4-6 cards");
  assert(pair.particles[i].every(x=>Number.isInteger(x)&&x>=0));
}
const pairLength=new BlindPhysicsEstimator({particles:8});
pairLength.observe("B",{observedDraws:4});
const info=pairLength.snapshot().pairwise_depletion;
assert.equal(info.observed_draw_count,true);
assert(info.estimated_draws>=4&&info.estimated_draws<=6,
  "observedDraws must be metadata, never a depletion filter");
assert.throws(()=>pairLength.observe("T",{observedDraws:7}),/invalid_observed_draw_count/);
assert.equal(ui.run().pairwise_depletion.observed_cards,false);
assert.equal(ui.run().fusion.modifies_real_ev,false);
 // 相同初始牌靴 + RNG，輸入任意 B/P/T 或補牌觀測，不得改變虛擬扣牌。
 const seqB=new BlindPhysicsEstimator({particles:16,seed:1729});
 const seqP=new BlindPhysicsEstimator({particles:16,seed:1729});
 const seqT=new BlindPhysicsEstimator({particles:16,seed:1729});
 const inputOptions=new BlindPhysicsEstimator({particles:16,seed:1729});
 for(let i=0;i<24;i++){
   seqB.observe("B");seqP.observe("P");seqT.observe("T");
   inputOptions.observe(i%2?"P":"B",{observedDraws:i%2?6:4});
   assert.deepEqual(seqB.particles,seqP.particles,"B/P must have identical virtual depletion");
   assert.deepEqual(seqB.particles,seqT.particles,"T must not force extra depletion");
   assert.deepEqual(seqB.particles,inputOptions.particles,"observedDraws must not condition depletion");
   assert.equal(seqB.snapshot().virtual_remaining,seqP.snapshot().virtual_remaining);
 }
 assert.equal(seqB.pairwiseHistory[0].source,"unconditional_shoe_sampling");
console.log("PASS V3 UI: original gates and display; unconditional virtual depletion invariant to B/P/T and observedDraws");