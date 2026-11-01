/**
 * BBB PHYSICS_FORECAST_FIRST V2 — dynamic research signal gate
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
const VERSION="BBB_PHYSICS_FORECAST_FIRST_V3_MICRO_FUSION",DECKS=8;
const BASE_EV_GATE=.005,OVERDRIVE_GATE=0,ONE_HAND_CUSHION=.035;
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
/**
 * 靴尾非線性敏感度只調整「研究訊號分數」，不重新定義莊/閒真實 EV。
 * 以局數 / 70 估算靴進程只是代理量；因 4/5/6 張未知，不是實際耗牌進度。
 */
function shoePhaseScaling(currentHand){
  if(!Number.isInteger(currentHand)||currentHand<0)throw Error("invalid_hand_index");
  const penetration_ratio=clamp(currentHand/70,0,1);
  return{penetration_ratio,scale:1+Math.pow(penetration_ratio,2)*1.5,
    source:"hand_index_proxy_not_observed_card_penetration",
    modifies_real_ev:false};
}
/**
 * 過往 B/P 非和局構造 Beta-Binomial 路單方向傾向。
 * 每次索引 t 的標籤，只使用 t 之前的路單狀態，避免用未來答案訓練。
 * Beta(4,4) 先驗將少量連龍、單跳回歸 50/50，拒絕假校準。
 */
function roadRegime(nonTie){
  if(nonTie.length<2)return null;
  const last=nonTie[nonTie.length-1];
  let run=1;
  for(let i=nonTie.length-2;i>=0&&nonTie[i]===last;i--)run++;
  const two=nonTie.slice(-2);
  const five=nonTie.slice(-5);
  let changes=0;
  for(let i=1;i<five.length;i++)if(five[i]!==five[i-1])changes++;
  const turnRate=five.length>1?changes/(five.length-1):.5;
  const regime=(two[0]===two[1]?"pair":"alternating")+
    ":"+(run>=3?"long":"short")+
    ":"+(turnRate>=.5?"turning":"stable");
  return{regime,last,run,turn_rate:turnRate,
    signature:regime+":"+last};
}
function roadPosterior(history,{prior=4,minimumSupport=8}={}){
  if(!Array.isArray(history)||history.some(x=>!valid(x)))
    throw Error("invalid_history");
  if(!finite(prior)||prior<1||!Number.isInteger(minimumSupport)||minimumSupport<1)
    throw Error("invalid_road_posterior_options");
  const clean=[],byKey=new Map();
  // 結算下一局前，只根據前一局已出現的二元結構記錄「是否延續」。
  for(const outcome of history){
    if(outcome==="T")continue;
    const state=roadRegime(clean);
    if(state){
      const item=byKey.get(state.regime)||{stay:0,change:0};
      if(outcome===state.last)item.stay++;else item.change++;
      byKey.set(state.regime,item);
    }
    clean.push(outcome);
  }
  const current=roadRegime(clean);
  if(!current)return{direction:null,road_bias:0,road_probability_b:.5,
    posterior_stay:.5,posterior_se:.5,samples:0,support:0,
    reliable:false,regime:"insufficient",signature:"insufficient"};
  const counts=byKey.get(current.regime)||{stay:0,change:0};
  const samples=counts.stay+counts.change;
  const a=prior+counts.stay,b=prior+counts.change;
  const stay=a/(a+b);
  const variance=(a*b)/((a+b)**2*(a+b+1));
  const roadProbabilityB=current.last==="B"?stay:1-stay;
  const posteriorSE=Math.sqrt(variance);
  const support=clamp(samples/(samples+16),0,1);
  // 僅可當作先驗平滑的微弱路單輔助，非真正已校準的貝氏預測。
  const reliable=samples>=minimumSupport&&
    Math.abs(stay-.5)>1.28*posteriorSE;
  return{direction:roadProbabilityB>=.5?"B":"P",
    road_bias:roadProbabilityB-.5,road_probability_b:roadProbabilityB,
    posterior_stay:stay,posterior_se:posteriorSE,samples,support,reliable,
    regime:current.regime,signature:current.signature,
    last:current.last,run:current.run,turn_rate:current.turn_rate,
    prior_beta:[prior,prior],calibrated:false};
}
/**
 * 具溫度 .1 的產品專家融合：物理結果是主體，路單僅加入小幅 log-odds。
 * 對 B/P/T 歷史沒有真正物理 likelihood，故這是 Bayes-inspired 診斷值，
 * 不是經驗證的後驗勝率，也不可修改純物理 EV 或下注方向。
 */
function microFusion(physical,history){
  const road=roadPosterior(history);
  const phase=shoePhaseScaling(history.length);
  const p=physical.simulated_probability||physical.win_probabilities;
  const pB=clamp(p.B/Math.max(1e-12,p.B+p.P),1e-6,1-1e-6);
  const logit=x=>Math.log(x/(1-x));
  const logistic=x=>1/(1+Math.exp(-x));
  const roadLogit=clamp(logit(clamp(road.road_probability_b,1e-5,1-1e-5)),-.6,.6);
  const roadWeight=.1*road.support;
  const posteriorB=logistic(logit(pB)+roadWeight*roadLogit);
  const physicsDirection=physical.ev_banker>=physical.ev_player?"B":"P";
  const aligned=road.reliable&&road.direction===physicsDirection;
  const conflicted=road.reliable&&road.direction!==physicsDirection;
  // 只放大代理訊號幅度（非 EV），一致加最多 10%，衝突降低最多 10%。
  const resonance=aligned?1+Math.min(.10,.2*road.support):
    conflicted?1-Math.min(.10,.2*road.support):1;
  const raw_deviation=physical.ev_gap_deviation;
  const scaled_deviation=raw_deviation*phase.scale;
  const weighted_score=Math.abs(scaled_deviation)*resonance;
  return{source:"bayes_inspired_beta_binomial_road_diagnostic",
    physics_direction:physicsDirection,road,physics_conditional_b:pB,
    fused_conditional_b:posteriorB,road_weight:roadWeight,
    aligned,conflicted,confidence_multiplier:resonance,
    raw_ev_gap_deviation:raw_deviation,scaled_deviation,
    research_score:weighted_score,phase,
    modifies_real_ev:false,calibrated:false};
}

/** 衝突風險只攔住一個新 regime 的首局，避免又演變成無限 Skip。 */
function conflictGuard(fusion,gate,lastSignature){
  const signature=fusion.road.signature+":"+fusion.physics_direction;
  const applied=fusion.conflicted&&lastSignature!==signature;
  return{signature,applied,effective_gate:applied?Math.max(gate,.035):gate,
    source:"one_time_road_physics_conflict"};
}

/**
 * 研究訊號狀態機：
 * BASE(0.005) → 上一局實際提示正確：OVERDRIVE(0)
 * BASE/OVERDRIVE → 上一局實際提示錯誤：CUSHION(0.035)
 * CUSHION → 下一個非和局若未通過門檻：BASE(0.005)
 *
 * 狀態只在 observe()/syncHistory() 的逐局結算更新；重複 predict() 不會
 * 虛增連勝或重複結算。預測錯誤是從上一局「已提示的研究方向」計算，
 * 不能偷看當前輸入開獎結果。B/P/T 盲估的高密度方向不等於 BET。
 */
class PhysicsForecastFirstEngine{
  constructor({simulationsPerParticle=64,observedSamples=12000,
    threshold=BASE_EV_GATE,lossGate=ONE_HAND_CUSHION,quantOptions={}}={}){
    if(!finite(threshold)||threshold<0||threshold>.5)throw Error("invalid_ev_gate");
    if(!finite(lossGate)||lossGate<0||lossGate>.5)throw Error("invalid_loss_gate");
    this.simulationsPerParticle=simulationsPerParticle;
    this.observedSamples=observedSamples;
    this.threshold=threshold;this.lossGate=lossGate;
    this.quant=new quant.OnlineLinUCB256(quantOptions);
    this.reset();
  }
  get history(){return this.quant.history;}
  get trainedRounds(){return this.quant.trainedRounds;}
  reset(){
    this.quant.reset();this.observedRows=null;this.lastReport=null;
    this.phase="base";this.winStreak=0;this.lastSettlement="none";
    this.lastConflictSignature=null;
    this.settlements={signals:0,correct:0,wrong:0,ties:0,filtered:0,turns:0};
    return this;
  }
  _gate(){return this.phase==="overdrive"?OVERDRIVE_GATE:
    this.phase==="cushion"?this.lossGate:this.threshold;}
  setObservedRounds(rows){
    if(rows===null){this.observedRows=null;this.lastReport=null;return this;}
    if(!Array.isArray(rows)||rows.some(r=>!r||!valid(r.outcome)||
      !Array.isArray(r.observation?.cards)))throw Error("invalid_observed_rows");
    // 匯入前先確認整靴完整，避免先顯示真實數據後才發現驗證失敗。
    if(rows.length!==this.history.length)throw Error("observed_rows_must_cover_entire_shoe");
    observedCounts(rows,this.history);
    this.observedRows=JSON.parse(JSON.stringify(rows));
    this.lastReport=null;return this;
  }
  /**
   * 每次真正開獎只結算一次研究方向；T 不宣稱方向輸贏。
   * 上一局若是 Cushion 且 research signal 不達 .035，只消耗下一個 B/P
   * 觀測事件就恢復 BASE，不會因和局卡住。若下一局仍有訊號，
   * 正確→OVERDRIVE，錯誤→再次一局 Cushion。
   */
  _settle(prior,outcome){
    this.lastReport=null; // 結算狀態改變立即使預測快取失效
    if(outcome==="T"){this.settlements.ties++;this.lastSettlement="tie";return;}
    this.settlements.turns++;
    const signal=prior.research_signal;
    // 同一種衝突只阻擋一次，不因連續相同路單無限觀望。
    if(signal?.conflict_guard_applied)
      this.lastConflictSignature=signal.conflict_signature;
    else if(!signal?.conflict_detected)this.lastConflictSignature=null;
    if(!signal?.available){
      this.settlements.filtered++;this.phase="base";
      this.winStreak=0;this.lastSettlement="filtered";
      return;
    }
    this.settlements.signals++;
    if(signal.direction===outcome){
      this.settlements.correct++;
      this.winStreak++;this.phase="overdrive";this.lastSettlement="win";
    }else{
      this.settlements.wrong++;
      this.winStreak=0;this.phase="cushion";this.lastSettlement="loss";
    }
  }
  syncHistory(seq){
    if(!Array.isArray(seq)||seq.some(v=>!valid(v)))throw Error("invalid_history");
    if(this.observedRows!==null&&this.observedRows.length!==seq.length)
      throw Error("observed_rows_must_cover_entire_shoe");
    let prefix=this.history.length<=seq.length;
    if(prefix)for(let i=0;i<this.history.length;i++)
      if(this.history[i]!==seq[i]){prefix=false;break;}
    if(!prefix){
      // 返回上一局或歷史改寫必須全程重播，杜絕重複計算連勝。
      const remembered=this.observedRows;
      this.reset();this.observedRows=remembered;
    }
    for(let i=this.history.length;i<seq.length;i++){
      const prior=this.predict();
      this._settle(prior,seq[i]);
      this.quant.observe(seq[i]);
      this.lastReport=null;
    }
    return this.predict();
  }
  predict(){
    if(this.lastReport&&this.lastReport.history_length===this.history.length)
      return this.lastReport;
    const quantPrediction=this.quant.predict();
    const observed=this.observedRows!==null&&this.history.length>0;
    const counts=observed?observedCounts(this.observedRows.slice(0,this.history.length),this.history):null;
    const physics=observed?simulate(counts,{
      samples:this.observedSamples,seed:hashHistory(this.history)^0xb15f13}):
      (this.history.length?summarizeHypotheses(this.quant.physics,this.history,{
        samplesPerParticle:this.simulationsPerParticle}):baselineResult());
    const normalized=observed?
      {...physics,simulated_probability:physics.win_probabilities,source:"observed_card_count"}:
      physics;
    const dynamicGate=this._gate();
    const fusion=microFusion(normalized,this.history);
    const score=fusion.research_score;
    const direction=fusion.physics_direction; // 路單不可反轉物理 EV 方向
    // 若 Beta(4,4) 後驗與物理主方向顯著衝突，對該 regime 首次遇到時
    // 啟用「一次性衝突觀望」，後續相同 regime 不重複卡住。
    const conflict=conflictGuard(fusion,dynamicGate,this.lastConflictSignature);
    const conflictSignature=conflict.signature;
    const conflictSkip=conflict.applied;
    const effectiveGate=conflict.effective_gate;
    const usable=!normalized.simulation_exhausted&&finite(score)&&
      finite(normalized.ev_banker)&&finite(normalized.ev_player);
    const available=usable&&!conflictSkip&&score>=effectiveGate;
    const researchSignal={
      action:available?"SIGNAL":"SKIP",
      direction,available,score,dynamic_gate:dynamicGate,
      effective_gate:effectiveGate,
      phase:this.phase,win_streak:this.winStreak,
      last_settlement:this.lastSettlement,
      source:observed?"observed_cards":"simulated_bpt_only",
      evidence_qualified:observed,
      conflict_detected:fusion.conflicted,
      conflict_guard_applied:conflictSkip,
      conflict_signature:conflictSignature,
      reason:!usable?"simulation_unavailable":conflictSkip?"short_conflict_guard":
        !available?"dynamic_signal_filter":"model_direction_not_validated"
    };
    // 正式 BET 仍要求觀測到每局真實點數、正 EV 和抽樣誤差下界。
    // 動態門檻會影響真實 EV 閘門，但任何連勝都無法捏造實際牌值。
    const gate=decisionFromEV(normalized,{observed,threshold:dynamicGate});
    const action=available?gate.action:"SKIP";
    const reason=!available?researchSignal.reason:gate.reason;
    const features=weightedContext(quantPrediction,normalized);
    const report={version:VERSION,history_length:this.history.length,
      action,direction:action==="BET"?direction:"Skip",bet_allowed:action==="BET",
      reason,card_count_source:normalized.source,
      observed_card_evidence:observed,card_count_gate:gate,
      research_signal:researchSignal,display_direction:direction,
      pairwise_depletion:quantPrediction.blind_physics.pairwise_depletion,
      fusion,shoe_phase: fusion.phase,
      scaled_research_score: fusion.research_score,
      raw_ev_gap_deviation:normalized.ev_gap_deviation,
      dynamic_gate:dynamicGate,base_ev_gate:this.threshold,
      cushion_gate:this.lossGate,gate_phase:this.phase,
      win_streak:this.winStreak,last_settlement:this.lastSettlement,
      prequential:{...this.settlements},
      signal_rate:this.settlements.turns?
        this.settlements.signals/this.settlements.turns:null,
      hit_rate:this.settlements.signals?
        this.settlements.correct/this.settlements.signals:null,
      context:features.vector,shoe_context:features.shoe,road_context:features.road,
      feature_weights:{physics:5,road:.1},
      road_has_direction_authority:false,model_road_debug:quantPrediction,
      physical_ev:normalized,ev_banker:normalized.ev_banker,ev_player:normalized.ev_player,
      ev_gap_deviation:normalized.ev_gap_deviation,ev_threshold:dynamicGate,
      baseline_ev:{...BASE_EV},hypothetical_candidate:direction,
      proxy_only:!observed,validated_edge:false,
      phantom_95pct_claim:false,
      notice:observed?
        "已輸入完整實際點數；仍需確認 MC 抽樣誤差與真實正 EV，連勝不保證下局贏。":
        "高頻莊/閒僅為虛擬殘牌的研究方向，不是已驗證的正 EV 或投注指令。"};
    this.lastReport=report;return report;
  }
}
return{PhysicsForecastFirstEngine,observedCounts,observedHand,
  deal,simulate,summarizeHypotheses,decisionFromEV,weightedContext,
  BASE,BASE_EV,INITIAL,DECKS,VERSION,BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION,
  shoePhaseScaling,roadRegime,roadPosterior,microFusion,conflictGuard};
});