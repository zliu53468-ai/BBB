/**
 * BBB PHYSICS_FORECAST_FIRST V1
 *
 * 物理主導決策，路單僅 0.1× 診斷，物理特徵 5×。
 * B/P/T 只可反推「可能」的 8 副牌路徑，無法知道實際扣了哪些牌。
 * 因此 B/P/T-only 模式可輸出模擬 EV，但永遠不能把它標為真實正 EV 或 BET。
 * 選用真實逐局完整牌值時才可經 EV + 不確定度閘門產生 BET 研究訊號。
 */
(function(root,factory){
  "use strict";
  const quant=(root&&root.BBBQuant256Online)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./quant256_online_runtime.js"):null);
  const api=factory(quant);
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBPhysicsForecastFirst=api;
})(typeof window!=="undefined"?window:null,function(quant){
"use strict";
if(!quant||typeof quant.OnlineLinUCB256!=="function")throw Error("missing_quant256_online_runtime");
const VERSION="BBB_PHYSICS_FORECAST_FIRST_V1",DECKS=8;
const INITIAL=Object.freeze([128,32,32,32,32,32,32,32,32,32]);
const BASE=Object.freeze({B:.458597,P:.446247,T:.095156});
const BASE_EV=Object.freeze({B:.95*BASE.B-BASE.P,P:BASE.P-BASE.B});
const BASE_GAP=BASE_EV.B-BASE_EV.P;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const valid=s=>["B","P","T"].includes(s);
const sum=a=>a.reduce((n,v)=>n+v,0);
function seeded(seed){
  let x=seed>>>0||0x9E3779B9;
  return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
}
function hashHistory(seq){
  let x=2166136261;
  for(const c of seq){x^=c.charCodeAt(0);x=Math.imul(x,16777619);}
  return x>>>0;
}
function validateCounts(counts,{observed=false}={}){
  if(!Array.isArray(counts)||counts.length!==10||counts.some((v,i)=>
    !Number.isInteger(v)||v<0||v>INITIAL[i]))throw Error("invalid_point_counts");
  const remaining=sum(counts);
  if(remaining>416||remaining<6)throw Error("remaining_shoe_not_playable");
  if(observed&&remaining===416)throw Error("no_observed_depletion");
  return remaining;
}
/** 抽一手合法的標準百家樂第三張牌（P1,B1,P2,B2,P3?,B3?）。 */
function deal(counts,rng){
  const available=counts.slice(),total=validateCounts(counts);
  let seen=0;
  function draw(){
    let pick=Math.floor(rng()*(total-seen));
    for(let i=0;i<10;i++){
      pick-=available[i];
      if(pick<0){available[i]--;seen++;return i;}
    }
    throw Error("virtual_deal_exhausted");
  }
  const cards=[draw(),draw(),draw(),draw()];
  const initialP=(cards[0]+cards[2])%10,initialB=(cards[1]+cards[3])%10;
  let p=initialP,b=initialB,p3=null;
  if(p<8&&b<8){
    if(p<=5){p3=draw();cards.push(p3);p=(p+p3)%10;}
    const bankerDraw=p3===null?b<=5:
      b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
      b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false;
    if(bankerDraw){const b3=draw();cards.push(b3);b=(b+b3)%10;}
  }
  return{cards,outcome:b>p?"B":p>b?"P":"T",player:p,banker:b};
}
/** 按出牌順序驗證完整觀測：不能用僅 B/P/T 偽造補牌數。 */
function observedHand(cards){
  if(!Array.isArray(cards)||cards.length<4||cards.length>6||
    cards.some(x=>!Number.isInteger(x)||x<0||x>9))
    throw Error("invalid_observed_hand");
  const a=(cards[0]+cards[2])%10,b=(cards[1]+cards[3])%10;
  const natural=a>=8||b>=8;
  const drawP=!natural&&a<=5;
  const p3=drawP?cards[4]:null;
  const drawB=!natural&&(p3===null?b<=5:
    b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
    b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false);
  const expectedLength=4+Number(drawP)+Number(drawB);
  if(cards.length!==expectedLength)throw Error("observed_hand_draw_rule_mismatch");
  const p=(a+(drawP?p3:0))%10;
  const bankerThird=drawB?cards[drawP?5:4]:0;
  const banker=(b+bankerThird)%10;
  return banker>p?"B":p>banker?"P":"T";
}
function observedCounts(rows,history){
  if(!Array.isArray(rows)||!Array.isArray(history)||rows.length!==history.length)
    throw Error("observed_rows_must_cover_entire_shoe");
  const counts=INITIAL.slice();
  for(let i=0;i<rows.length;i++){
    const row=rows[i],outcome=history[i],cards=row?.observation?.cards;
    if(!row||row.outcome!==outcome||!valid(outcome)||!Array.isArray(cards))
      throw Error("observed_rows_outcome_mismatch");
    if(observedHand(cards)!==outcome)throw Error("observed_result_mismatch");
    for(const card of cards){
      counts[card]--;
      if(counts[card]<0)throw Error("observed_card_exceeds_shoe");
    }
  }
  validateCounts(counts,{observed:true});
  return counts;
}
/** Monte Carlo 只從剩餘牌組成計算 B/P/T，沒有引入牌路權重。 */
function simulate(counts,{samples=2048,seed=0x13877}={}){
  validateCounts(counts);
  if(!Number.isInteger(samples)||samples<64||samples>50000)throw Error("invalid_mc_sample_count");
  const rng=seeded(seed),wins={B:0,P:0,T:0};
  let expectedCards=0;
  for(let i=0;i<samples;i++){
    const hand=deal(counts,rng);
    wins[hand.outcome]++;expectedCards+=hand.cards.length;
  }
  const p={B:wins.B/samples,P:wins.P/samples,T:wins.T/samples};
  const evB=.95*p.B-p.P,evP=p.P-p.B;
  // 莊/閒下注逢和退回原注。SE 用每局盈虧平方的樣本估計。
  const seB=Math.sqrt(Math.max(0,(.95*.95*p.B+p.P-evB*evB)/samples));
  const seP=Math.sqrt(Math.max(0,(p.B+p.P-evP*evP)/samples));
  const gap=evB-evP;
  const seGap=Math.sqrt(Math.max(0,(1.95*1.95*p.B+4*p.P-gap*gap)/samples));
  return{samples,counts:counts.slice(),remaining:sum(counts),
    win_probabilities:p,ev_banker:evB,ev_player:evP,ev_gap:gap,
    ev_gap_deviation:gap-BASE_GAP,ev_banker_delta:evB-BASE_EV.B,
    ev_player_delta:evP-BASE_EV.P,se_banker:seB,se_player:seP,
    se_gap:seGap,draws_mean:expectedCards/samples,
    payout:{banker_win:.95,player_win:1,tie_push:true}};
}
function summarizeHypotheses(physics,history,{samplesPerParticle=64}={}){
  if(!physics||!Array.isArray(physics.particles))throw Error("invalid_physics_particles");
  if(!Number.isInteger(samplesPerParticle)||samplesPerParticle<8||samplesPerParticle>1024)
    throw Error("invalid_shadow_sample_count");
  const validParticles=physics.particles.filter(c=>Array.isArray(c)&&sum(c)>=6);
  const remaining=physics.particles.map(c=>sum(c));
  const estimates=[];
  const stableSeed=hashHistory(history);
  for(let i=0;i<validParticles.length;i++){
    const counts=validParticles[i],rng=seeded(stableSeed^Math.imul(i+1,2654435761));
    const tally={B:0,P:0,T:0};let draws=0;
    for(let j=0;j<samplesPerParticle;j++){
      const result=deal(counts,rng);
      tally[result.outcome]++;draws+=result.cards.length;
    }
    estimates.push({...tally,draws});
  }
  const n=estimates.length*samplesPerParticle;
  if(!n)return{...baselineResult(),particle_support:0,simulation_exhausted:true};
  const p={
    B:sum(estimates.map(e=>e.B))/n,
    P:sum(estimates.map(e=>e.P))/n,
    T:sum(estimates.map(e=>e.T))/n
  };
  const evB=.95*p.B-p.P,evP=p.P-p.B,gap=evB-evP;
  const seGap=Math.sqrt(Math.max(0,(1.95**2*p.B+4*p.P-gap**2)/n));
  return{remaining:sum(remaining)/Math.max(remaining.length,1),
    samples:n,particle_support:validParticles.length,
    simulated_probability:p,ev_banker:evB,ev_player:evP,
    ev_gap:gap,ev_gap_deviation:gap-BASE_GAP,
    se_gap:seGap,draws_mean:sum(estimates.map(e=>e.draws))/n,
    source:"hypothetical_particle_posterior",simulation_exhausted:false};
}
function baselineResult(){
  return{samples:0,remaining:416,particle_support:0,simulated_probability:{...BASE},
    ev_banker:BASE_EV.B,ev_player:BASE_EV.P,ev_gap:BASE_GAP,
    ev_gap_deviation:0,se_gap:null,draws_mean:null,
    source:"unobserved_baseline",simulation_exhausted:false};
}
function decisionFromEV(s,{observed=false,threshold=.015,z=1.96}={}){
  if(!finite(threshold)||threshold<0)throw Error("invalid_ev_gate");
  const direction=s.ev_banker>=s.ev_player?"B":"P";
  const chosenEV=direction==="B"?s.ev_banker:s.ev_player;
  const se=direction==="B"?s.se_banker:s.se_player;
  const enough=observed&&finite(se)&&finite(s.se_gap);
  const significant=enough&&Math.abs(s.ev_gap_deviation)>threshold&&
    Math.abs(s.ev_gap_deviation)>z*s.se_gap&&
    chosenEV-z*se>0;
  return{action:significant?"BET":"SKIP",direction:significant?direction:"Skip",
    candidate:direction,bet_allowed:significant,
    reason:!observed?"unobserved_card_composition":
      !significant?"physical_ev_gate_not_confirmed":"observed_physical_ev_confirmed",
    ev_threshold:threshold,ev_gap_deviation:s.ev_gap_deviation,
    selected_ev:chosenEV,ev_lower_bound:enough?chosenEV-z*se:null,
    evidence_source:observed?"supplied_observed_cards":"bpt_virtual_only"};
}
function weightedContext(model,physical){
  const raw=model.context;
  if(!Array.isArray(raw)||raw.length!==256)throw Error("invalid_quant256_context");
  const shoe=Array(128).fill(0),road=raw.slice(128).map(x=>clamp(x,-1,1)*.1);
  const remaining=physical.remaining;
  const phase=clamp((416-remaining)/416,0,1),p=physical.simulated_probability||physical.win_probabilities;
  // 5× 物理主體：進程、三次方、點數虛擬殘量的中心差、模型 EV 與基準偏移。
  shoe[0]=5*phase;shoe[1]=5*phase**2;shoe[2]=5*phase**3;
  shoe[3]=5*(1-phase);shoe[4]=5*(1-phase)**2;shoe[5]=5*(1-phase)**3;
  shoe[6]=5*clamp(p.B-BASE.B,-1,1);
  shoe[7]=5*clamp(p.P-BASE.P,-1,1);
  shoe[8]=5*clamp(p.T-BASE.T,-1,1);
  shoe[9]=5*clamp(physical.ev_gap_deviation,-1,1);
  shoe[10]=5*clamp(physical.ev_banker-BASE_EV.B,-1,1);
  shoe[11]=5*clamp(physical.ev_player-BASE_EV.P,-1,1);
  if(Array.isArray(physical.counts)){
    for(let i=0;i<10;i++)
      shoe[16+i]=5*(physical.counts[i]/Math.max(6,remaining)-INITIAL[i]/416);
  }
  return{vector:shoe.concat(road),shoe,road,physics_weight:5,
    road_weight:.1,road_has_direction_authority:false};
}
class PhysicsForecastFirstEngine{
  constructor({simulationsPerParticle=64,observedSamples=12000,threshold=.015,quantOptions={}}={}){
    if(!finite(threshold)||threshold<0||threshold>.5)throw Error("invalid_ev_gate");
    this.simulationsPerParticle=simulationsPerParticle;
    this.observedSamples=observedSamples;
    this.threshold=threshold;this.quant=new quant.OnlineLinUCB256(quantOptions);
    this.observedRows=null;this.lastReport=null;
  }
  get history(){return this.quant.history;}
  get trainedRounds(){return this.quant.trainedRounds;}
  reset(){this.quant.reset();this.observedRows=null;this.lastReport=null;return this;}
  setObservedRounds(rows){
    if(rows===null){this.observedRows=null;this.lastReport=null;return this;}
    if(!Array.isArray(rows)||rows.some(r=>!r||!valid(r.outcome)||
      !Array.isArray(r.observation?.cards)))throw Error("invalid_observed_rows");
    this.observedRows=JSON.parse(JSON.stringify(rows));this.lastReport=null;return this;
  }
  syncHistory(history){
    this.quant.syncHistory(history);
    if(this.observedRows!==null&&this.observedRows.length!==history.length)
      throw Error("observed_rows_must_cover_entire_shoe");
    this.lastReport=null;return this.predict();
  }
  predict(){
    if(this.lastReport&&this.lastReport.history_length===this.history.length)
      return this.lastReport;
    const quantPrediction=this.quant.predict();
    const observed=this.observedRows!==null;
    const counts=observed?observedCounts(this.observedRows,this.history):null;
    const physics=observed?simulate(counts,{
      samples:this.observedSamples,seed:hashHistory(this.history)^0xb15f13}):
      (this.history.length?summarizeHypotheses(this.quant.physics,this.history,{
        samplesPerParticle:this.simulationsPerParticle}):baselineResult());
    const normalized=observed?
      {...physics,simulated_probability:physics.win_probabilities,source:"observed_card_count"}:
      physics;
    const gate=decisionFromEV(normalized,{observed,threshold:this.threshold});
    const features=weightedContext(quantPrediction,normalized);
    const report={version:VERSION,history_length:this.history.length,
      action:gate.action,direction:gate.direction,bet_allowed:gate.bet_allowed,
      reason:gate.reason,card_count_source:normalized.source,
      observed_card_evidence:observed,card_count_gate:gate,
      context:features.vector,shoe_context:features.shoe,road_context:features.road,
      feature_weights:{physics:5,road:.1},
      road_has_direction_authority:false,model_road_debug:quantPrediction,
      physical_ev:normalized,ev_banker:normalized.ev_banker,ev_player:normalized.ev_player,
      ev_gap_deviation:normalized.ev_gap_deviation,ev_threshold:this.threshold,
      baseline_ev:{...BASE_EV},hypothetical_candidate:gate.candidate,
      proxy_only:!observed,validated_edge:false,
      notice:observed?
        "已提供逐局實際牌值；MC 的機率仍有抽樣誤差，非保證獲利。":
        "B/P/T 無法得知實際殘牌；本頁僅顯示假設條件模擬 EV，強制 Skip，不會偽稱真實算牌優勢。"};
    this.lastReport=report;return report;
  }
}
return{PhysicsForecastFirstEngine,observedCounts,observedHand,
  deal,simulate,summarizeHypotheses,decisionFromEV,weightedContext,
  BASE,BASE_EV,INITIAL,DECKS,VERSION};
});