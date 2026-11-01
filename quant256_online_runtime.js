/**
 * BBB 256D Online LinUCB V3 — hyper-sensitive high-frequency experimental B/P/T engine.
 * 本模組不讀取 A~K，也不把盲估牌張當真實算牌資料。
 * 128D 牌靴 + 128D 牌路固定介面，內部只訓練經挑選的 36 個非零特徵，
 * 其餘維度為 0；活躍維度使用完整正定 Gram 矩陣，而非凍結係數。
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBQuant256Online=api;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const DIM=256,SHOE=128,DECK=416,BASE_POINTS=[128,32,32,32,32,32,32,32,32,32];
const ACTIVE=[...Array.from({length:12},(_,i)=>i),...Array.from({length:24},(_,i)=>128+i)];
const K=ACTIVE.length,VERSION="BBB_QUANT256_ONLINE_V4_CONDITIONAL_MARGINAL";
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const valid=s=>s==="B"||s==="P"||s==="T";
const bp=h=>h.filter(v=>v!=="T");
const safeRatio=(n,d,f=0)=>d?n/d:f;
function windowStats(history,n){
  const a=bp(history).slice(-n),count=a.length;
  if(!count)return{count:0,banker:.5,turn:.5,regularity:.5,entropy:1,run:0};
  let turns=0;
  for(let i=1;i<count;i++)if(a[i]!==a[i-1])turns++;
  const turn=count>=2?turns/(count-1):.5;
  const entropy=turn===0||turn===1?0:-turn*Math.log2(turn)-(1-turn)*Math.log2(1-turn);
  let run=1;for(let i=count-1;i>0&&a[i]===a[i-1];i--)run++;
  return{count,banker:a.filter(x=>x==="B").length/count,turn,regularity:Math.abs(2*turn-1),entropy,run};
}
function seededRng(seed){
  let x=seed>>>0||0x6d2b79f5;
  return function(){x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
}
/** 正確的百家樂第三張牌規則；每組虛擬粒子保存 0~9 點的假設殘牌。 */
function sampleHand(counts,rng){
  const c=counts.slice(),total=c.reduce((a,b)=>a+b,0);
  if(total<6)return null;
  let cards=0;
  const draw=()=>{
    let r=Math.floor(rng()*(total-cards)),idx=9;
    for(let i=0;i<10;i++){r-=c[i];if(r<0){idx=i;break;}}
    if(c[idx]<=0)throw Error("virtual_deck_empty");
    c[idx]--;cards++;return idx;
  };
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();
  let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
    let p3=null;
    if(p<=5){p3=draw();p=(p+p3)%10;}
    let bankerDraw=false;
    if(p3===null)bankerDraw=b<=5;
    else if(b<=2)bankerDraw=true;
    else if(b===3)bankerDraw=p3!==8;
    else if(b===4)bankerDraw=p3>=2&&p3<=7;
    else if(b===5)bankerDraw=p3>=4&&p3<=7;
    else if(b===6)bankerDraw=p3>=6&&p3<=7;
    if(bankerDraw)b=(b+draw())%10;
  }
  return{counts:c,draws:cards,result:b>p?"B":p>b?"P":"T",
    consumed:BASE_POINTS.map((v,i)=>counts[i]-c[i])};
}
/**
 * 不可辨識性聲明：B/P/T 不提供實際 4/5/6 張或點數。
 * 這裡僅從合法發牌模擬抽取「符合已知結果的可能路徑」做 latent ensemble；
 * 它不是對真實牌靴的測量，亦不代表已知任何 A~K 殘量。
 */
/**
 * 條件序貫蒙地卡羅：
 * 1. 每個粒子在原有殘牌上抽多個合法 4/5/6 張候選局；
 * 2. 每個候選局的點數都依真實不放回張數抽，且套用第三張牌規則；
 * 3. 只有符合本局 B/P/T（和已觀測張數）的候選局可成為扣牌轉移；
 * 4. 以「符合結果的候選比例」當觀測 likelihood，再按此權重系統重採樣。
 *
 * 這會避免原本每個粒子必定找一個符合結果的樣本、卻完全忽略
 * P(實際結果 | 可能殘牌) 的偏差。當只有 B/P/T，仍不能知道真實牌張。
 */
class BlindPhysicsEstimator{
  constructor({particles=24,seed=0xBBB256,proposalTrials=64}={}){
    this.size=clamp(Math.floor(particles)||24,8,96);
    if(!Number.isInteger(proposalTrials)||proposalTrials<16||proposalTrials>256)
      throw Error("invalid_particle_proposals");
    this.proposalTrials=proposalTrials;this.seed=seed>>>0;this.reset();
  }
  reset(){
    this.rng=seededRng(this.seed);
    this.particles=Array.from({length:this.size},()=>BASE_POINTS.slice());
    this.drawHistory=[];this.densityHistory=[];this.pairwiseHistory=[];
    this.lastAcceptance=0;this.lastEss=this.size;this.lastLikelihood=0;
    this.lastMarginal={
      method:"conditional_observation_likelihood",
      point_values:[0,1,2,3,4,5,6,7,8,9],
      // 權重是條件耗牌的相對過量，不是大小牌對莊/閒固定方向的規則。
      relative_excess:Array(10).fill(0),
      expected_conditional:Array(10).fill(0),
      expected_unconditional:Array(10).fill(0),
      small_1_4_excess:0,large_5_8_excess:0,
      likelihood:0,samples:0,identified_cards:false
    };
    this.lastPairwise={source:"conditional_joint_sampling",observed_cards:false,
      observed_draw_count:false,matched_particles:0,tie_or_long_hands:0,
      estimated_key_consumption:0,estimated_key_consumption_ratio:0,
      estimated_pairwise_key_copresence:0,estimated_draws:0};
    return this;
  }
  observe(outcome,options={}){
    if(!valid(outcome))throw Error("invalid_outcome");
    const observedDraws=options.observedDraws??null;
    if(observedDraws!==null&&![4,5,6].includes(observedDraws))
      throw Error("invalid_observed_draw_count");
    const candidateBanks=[],likelihoods=[];
    const allDraw=Array(10).fill(0),matchedDraw=Array(10).fill(0);
    let proposalCount=0,matchedCount=0;
    for(let i=0;i<this.size;i++){
      const original=this.particles[i];
      const acceptedBank=[];
      let matches=0,proposals=0;
      // Reservoir sampling：從符合條件的所有候選局無偏隨機選一條。
      for(let j=0;j<this.proposalTrials;j++){
        const trial=sampleHand(original,this.rng);
        if(!trial)break;
        proposals++;proposalCount++;
        for(let k=0;k<10;k++)allDraw[k]+=trial.consumed[k];
        if(trial.result!==outcome||
          (observedDraws!==null&&trial.draws!==observedDraws))continue;
        matches++;matchedCount++;
        for(let k=0;k<10;k++)matchedDraw[k]+=trial.consumed[k];
        // 多候選 reservoir：重採樣同一父粒子時仍可抽取不同合法發牌路徑。
        if(acceptedBank.length<8)acceptedBank.push(trial);
        else{
          const j=Math.floor(this.rng()*matches);
          if(j<8)acceptedBank[j]=trial;
        }
      }
      // 在罕見的零命中情況增加候選，不能直接把不符合結果的手硬扣掉。
      if(!matches&&proposals){
        for(let extra=0;extra<256;extra++){
          const trial=sampleHand(original,this.rng);
          if(!trial)break;
          proposals++;proposalCount++;
          for(let k=0;k<10;k++)allDraw[k]+=trial.consumed[k];
          if(trial.result!==outcome||
            (observedDraws!==null&&trial.draws!==observedDraws))continue;
          matches++;matchedCount++;
          for(let k=0;k<10;k++)matchedDraw[k]+=trial.consumed[k];
          acceptedBank.push(trial);break;
        }
      }
      // 匹配機率在此粒子下估計；不把方向固定成莊/閒兩個人為群組。
      likelihoods.push(proposals?matches/proposals:0);
      candidateBanks.push(acceptedBank);
    }
    const total=likelihoods.reduce((a,b)=>a+b,0);
    if(total<=0){
      // 資訊不足時不得虛構轉移；保留原牌並回報不可識別。
      this.lastAcceptance=0;this.lastLikelihood=0;this.lastEss=0;
      this.lastMarginal={...this.lastMarginal,likelihood:0,samples:proposalCount};
      this.lastPairwise={...this.lastPairwise,matched_particles:0,
        tie_or_long_hands:0,estimated_draws:0};
      this.drawHistory.push(0);this.pairwiseHistory.push({...this.lastPairwise});
      const snapshot=this.snapshot();this.densityHistory.push({
        high:snapshot.virtual_high_ratio,low:snapshot.virtual_low_ratio});
      return snapshot;
    }
    // SMC 系統重採樣：對 P(結果|粒子殘牌) 高的粒子加權，避免
    // 舊方法「每個粒子都以同一權重接受一次」造成長期推估漂移。
    const ess=total**2/likelihoods.reduce((sum,w)=>sum+w*w,0);
    const step=total/this.size,offset=this.rng()*step;
    const next=[];
    let index=0,cumulative=likelihoods[0];
    for(let i=0;i<this.size;i++){
      const target=offset+i*step;
      while(target>=cumulative&&index<this.size-1){
        index++;cumulative+=likelihoods[index];
      }
      const bank=candidateBanks[index];
      next.push(bank.length?
        bank[Math.floor(this.rng()*bank.length)].counts.slice():
        this.particles[index].slice());
    }
    // 所有實際粒子僅由合法匹配的出牌轉移；所有耗牌維持非負整數。
    this.particles=next;this.lastEss=ess;this.lastLikelihood=matchedCount/Math.max(1,proposalCount);
    this.lastAcceptance=candidateBanks.filter(bank=>bank.length>0).length/this.size;
    const cond=matchedDraw.map(v=>v/Math.max(1,matchedCount));
    const uncond=allDraw.map(v=>v/Math.max(1,proposalCount));
    const excess=cond.map((v,i)=>(v-uncond[i])/Math.max(.05,uncond[i]));
    this.lastMarginal={
      method:"conditional_observation_likelihood",point_values:[0,1,2,3,4,5,6,7,8,9],
      relative_excess:excess,expected_conditional:cond,expected_unconditional:uncond,
      small_1_4_excess:excess.slice(1,5).reduce((a,b)=>a+b,0)/4,
      large_5_8_excess:excess.slice(5,9).reduce((a,b)=>a+b,0)/4,
      likelihood:this.lastLikelihood,samples:proposalCount,identified_cards:false
    };
    let keyDraws=0,pairCoPresence=0,longHands=0,sumDraws=0;
    for(const bank of candidateBanks){
      if(!bank.length)continue;
      // 用同一父粒子條件樣本估算共同消耗；不是實際抽到的牌。
      const selected=bank[0];
      sumDraws+=selected.draws;
      if(outcome==="T"||selected.draws>=5){
        longHands++;
        const group=[0,1,4,7].map(k=>selected.consumed[k]);
        keyDraws+=group.reduce((a,b)=>a+b,0);
        if(group.filter(v=>v>0).length>=2)pairCoPresence++;
      }
    }
    this.lastPairwise={
      source:"conditional_joint_sampling",observed_cards:false,
      observed_draw_count:observedDraws!==null,
      matched_particles:candidateBanks.filter(bank=>bank.length>0).length,
      tie_or_long_hands:longHands,
      estimated_key_consumption:longHands?keyDraws/longHands:0,
      estimated_key_consumption_ratio:sumDraws?keyDraws/sumDraws:0,
      estimated_pairwise_key_copresence:longHands?pairCoPresence/longHands:0,
      estimated_draws:candidateBanks.some(bank=>bank.length)?
        sumDraws/candidateBanks.filter(bank=>bank.length>0).length:0
    };
    this.drawHistory.push(this.lastPairwise.estimated_draws);
    this.pairwiseHistory.push({...this.lastPairwise});
    const snapshot=this.snapshot();
    this.densityHistory.push({high:snapshot.virtual_high_ratio,low:snapshot.virtual_low_ratio});
    return snapshot;
  }
  snapshot(){
    const mean=BASE_POINTS.map((_,k)=>this.particles.reduce((s,p)=>s+p[k],0)/this.size);
    const remain=mean.reduce((s,v)=>s+v,0);
    const high=mean[7]+mean[8]+mean[9],low=mean[1]+mean[2]+mean[3]+mean[4];
    const highBase=96/416,lowBase=128/416;
    const highRatio=remain?high/remain:highBase,lowRatio=remain?low/remain:lowBase;
    const highs=this.particles.map(p=>(p[7]+p[8]+p[9])/Math.max(1,p.reduce((s,v)=>s+v,0)));
    const variance=highs.reduce((s,v)=>s+(v-highRatio)**2,0)/this.size;
    const recent=this.drawHistory.slice(-8).filter(x=>x>0);
    const avg=(values,n,fallback)=>{
      const tail=values.slice(-n);return tail.length?
        tail.reduce((sum,v)=>sum+v,0)/tail.length:fallback;
    };
    const densityHigh=this.densityHistory.map(x=>x.high),
      lows=this.densityHistory.map(x=>x.low),draws=this.drawHistory.filter(v=>v>0);
    const highMomentum=avg(densityHigh,3,highRatio)-avg(densityHigh,8,highRatio);
    const lowMomentum=avg(lows,3,lowRatio)-avg(lows,8,lowRatio);
    const drawMomentum=(avg(draws,3,4.94)-avg(draws,8,4.94))/6;
    return{
      source:"simulated_unobserved",estimated_only:true,observed_cards:0,
      virtual_remaining:remain,virtual_high_ratio:highRatio,virtual_low_ratio:lowRatio,
      virtual_high_delta:highRatio-highBase,virtual_low_delta:lowRatio-lowBase,
      virtual_high_momentum:highMomentum,virtual_low_momentum:lowMomentum,
      virtual_draw_momentum:drawMomentum,
      virtual_high_std:Math.sqrt(variance),conditional_acceptance:this.lastAcceptance,
      conditional_outcome_likelihood:this.lastLikelihood,effective_particle_size:this.lastEss,
      expected_draws_recent:recent.length?recent.reduce((s,v)=>s+v,0)/recent.length:4.94,
      marginal_card_weights:{...this.lastMarginal},
      pairwise_depletion:{...this.lastPairwise},
      actual_draw_count_known:false,actual_remaining_known:false
    };
  }
}
function context256(history,physics){
  if(!Array.isArray(history)||history.some(v=>!valid(v)))throw Error("invalid_history");
  const shoe=Array(SHOE).fill(0),road=Array(SHOE).fill(0);
  const s2=windowStats(history,2),s5=windowStats(history,5),
    s8=windowStats(history,8),s16=windowStats(history,16);
  const nonTie=bp(history),tieRatio=n=>{
    const tail=history.slice(-n);return tail.length?
      tail.filter(x=>x==="T").length/tail.length:0;
  };
  const sim=physics.snapshot(),last=nonTie.at(-1)||"";
  const turn25=s2.turn-s5.turn,regular25=s2.regularity-s5.regularity;
  const entropy25=s2.entropy-s5.entropy;
  // 固定 128D 牌靴/進程槽：只用可得的短窗相對差與虛擬模擬動量。
  // 不將盲估的高低張殘牌當成實測事實，也不加入單調累積局數。
  shoe[0]=clamp(s2.banker-s5.banker,-1,1);
  shoe[1]=clamp(s5.banker-s8.banker,-1,1);
  shoe[2]=clamp(tieRatio(2)-tieRatio(5),-1,1);
  shoe[3]=clamp(tieRatio(5)-tieRatio(8),-1,1);
  shoe[4]=clamp(sim.virtual_high_momentum*2,-.2,.2);
  shoe[5]=clamp(sim.virtual_low_momentum*2,-.2,.2);
  shoe[6]=clamp(sim.virtual_high_momentum-sim.virtual_low_momentum,-.2,.2);
  shoe[7]=clamp(sim.virtual_draw_momentum,-.2,.2);
  shoe[8]=0; // 盲抽不提供真實剩餘牌或預測的確定性
  shoe[9]=clamp(turn25,-1,1);
  shoe[10]=clamp(s5.entropy-s8.entropy,-1,1);
  shoe[11]=clamp(entropy25,-1,1);

  // 後 128D：核心改為 2 vs 5 局。所有有效視窗 <=16 局（無 32+ 長窗）。
  // Entropy 在只有兩筆非和局時常為 0，它仍是變化檢測而非預知未來。
  road[0]=last==="B"?.5:last==="P"?-.5:0;
  road[1]=clamp(s2.run/Math.max(1,s2.count)-s5.run/Math.max(1,s5.count),-1,1);
  road[2]=clamp(s2.turn-.5,-.5,.5);
  road[3]=clamp(s5.turn-.5,-.5,.5);
  road[4]=clamp(turn25,-1,1);
  road[5]=clamp((s2.banker-.5)*(s2.regularity-.5),-1,1);
  road[6]=clamp((s5.banker-.5)*(s5.regularity-.5),-1,1);
  road[7]=clamp(regular25,-1,1);
  road[8]=clamp((s2.banker-.5)*(s2.entropy-.5),-1,1);
  road[9]=clamp(s5.entropy-s8.entropy,-1,1);
  road[10]=clamp(entropy25,-1,1);
  road[11]=s2.banker-.5;
  road[12]=s5.banker-.5;
  road[13]=clamp(s2.banker-s5.banker,-1,1);
  road[14]=clamp(s8.banker-.5,-.5,.5);
  road[15]=clamp(s5.turn-s8.turn,-1,1);
  road[16]=clamp(s5.run/Math.max(1,s5.count)-s8.run/Math.max(1,s8.count),-1,1);
  road[17]=clamp((s8.banker-.5)*(s8.regularity-.5),-1,1);
  road[18]=clamp((s5.banker-s8.banker)*(s5.entropy-s8.entropy),-1,1);
  road[19]=clamp(tieRatio(2)-tieRatio(5),-1,1);
  road[20]=nonTie.length<2?0:nonTie.at(-1)===nonTie.at(-2)?.4:-.4;
  road[21]=clamp(s2.banker-s8.banker,-1,1);
  road[22]=clamp(s5.regularity-s8.regularity,-1,1);
  road[23]=clamp(s2.turn-s8.turn,-1,1);
  const vector=shoe.concat(road);
  if(vector.length!==DIM||vector.some(x=>!finite(x)))throw Error("invalid_256D_context");
  return{vector,shoe,road,physics:sim,features:{
    regularity2:s2.regularity,regularity5:s5.regularity,regularity_delta:regular25,
    turn2:s2.turn,turn5:s5.turn,turn_rate_delta:turn25,
    entropy2:s2.entropy,entropy5:s5.entropy,entropy_delta:entropy25,
    entropy_volatility:Math.abs(entropy25),
    non_tie_sample:nonTie.length,maximum_window:16
  }};
}
function makeArm(){
  const A=Array.from({length:K},(_,i)=>Float64Array.from({length:K},(_,j)=>i===j?1:0));
  return{A,b:new Float64Array(K)};
}
/** 精確的正定 Cholesky 解方；求解 A^(-1)b 與 A^(-1)x。 */
function solveSPD(A,v){
  const L=Array.from({length:K},()=>new Float64Array(K));
  for(let i=0;i<K;i++)for(let j=0;j<=i;j++){
    let t=A[i][j];
    for(let k=0;k<j;k++)t-=L[i][k]*L[j][k];
    if(i===j){
      if(!(t>1e-9))throw Error("linucb_matrix_not_positive_definite");
      L[i][j]=Math.sqrt(t);
    }else L[i][j]=t/L[j][j];
  }
  const y=new Float64Array(K),x=new Float64Array(K);
  for(let i=0;i<K;i++){let v0=v[i];for(let j=0;j<i;j++)v0-=L[i][j]*y[j];y[i]=v0/L[i][i];}
  for(let i=K-1;i>=0;i--){let v0=y[i];for(let j=i+1;j<K;j++)v0-=L[j][i]*x[j];x[i]=v0/L[i][i];}
  return x;
}
function dot(a,b){let s=0;for(let i=0;i<K;i++)s+=a[i]*b[i];return s;}
class OnlineLinUCB256{
  constructor({lambda=.82,alpha=.15,minSamples=0,confidenceThreshold=0,
    lossGate=.04,particles=24}={}){
    if(!finite(lambda)||lambda<.75||lambda>.96)throw Error("lambda_must_be_0.75_to_0.96");
    if(!finite(alpha)||alpha<0||alpha>2)throw Error("invalid_alpha");
    if(!Number.isInteger(minSamples)||minSamples<0)throw Error("invalid_min_samples");
    if(!finite(confidenceThreshold)||confidenceThreshold<0||confidenceThreshold>.5)
      throw Error("invalid_confidence_threshold");
    if(!finite(lossGate)||lossGate<0||lossGate>.5)throw Error("invalid_loss_gate");
    this.lambda=lambda;this.alpha=alpha;this.minSamples=minSamples;
    this.confidenceThreshold=confidenceThreshold;this.lossGate=lossGate;
    this.particleCount=particles;this.reset();
  }
  reset(){
    this.history=[];this.arms={B:makeArm(),P:makeArm()};
    this.physics=new BlindPhysicsEstimator({particles:this.particleCount});
    this.trainedRounds=0;
    this.prequential={eligible:0,correct:0,wrong:0,ties:0,skipped:0};
    this.last_signal=null;this.last_settlement="none";
    this.gateCooldown=0;this.recommendationArm=null;this.recommendationStreak=0;
    return this;
  }
  _x(vector){return Float64Array.from(ACTIVE,i=>vector[i]);}
  _armScore(arm,x){
    const m=this.arms[arm],theta=solveSPD(m.A,m.b),projection=solveSPD(m.A,x);
    const mean=dot(x,theta),uncertainty=Math.sqrt(Math.max(0,dot(x,projection)));
    return{mean,uncertainty,ucb:mean+this.alpha*uncertainty};
  }
  predict(){
    const c=context256(this.history,this.physics),x=this._x(c.vector);
    const banker=this._armScore("B",x),player=this._armScore("P",x);
    const delta=Math.abs(banker.ucb-player.ucb);
    // 高密度訊號政策：平常 dynamic_gate=0；預測失誤只提高下一局至 .04。
    // 這是研究用信號強度門檻，不是下注具有正 EV 的保證。
    let dynamic_gate=Math.max(this.confidenceThreshold,this.gateCooldown);
    const enough=this.trainedRounds>=this.minSamples;
    const allowed=enough&&delta>=dynamic_gate;
    // 分數同為零時先展示 B 為固定決策，不得標記為高信心或高勝率。
    const rawDirection=banker.ucb>=player.ucb?"B":"P";
    const reason=!enough?"warmup":!allowed?"one_round_loss_gate":
      delta===0?"neutral_default_signal":"hyper_micro_signal";
    return{
      version:VERSION,direction:allowed?rawDirection:"Skip",raw_direction:rawDirection,
      bet_allowed:allowed,reason,dynamic_gate,
      ucb_banker:banker.ucb,ucb_player:player.ucb,
      delta,confidence_threshold:dynamic_gate,
      arms:{B:banker,P:player},history_length:this.history.length,
      trained_rounds:this.trainedRounds,lambda:this.lambda,alpha:this.alpha,
      context:c.vector,shoe_context:c.shoe,road_context:c.road,
      active_dimensions:K,features:c.features,blind_physics:c.physics,
      gate_cooldown_rounds:this.gateCooldown>0?1:0,
      last_settlement:this.last_settlement,
      recommendation_streak:this.recommendationStreak,
      calibrated_probability:false,claimed_edge:false,
      prequential:{...this.prequential},
      evaluated_hands:this.prequential.eligible,
      hit_rate:this.prequential.eligible-this.prequential.ties>0?
        this.prequential.correct/(this.prequential.eligible-this.prequential.ties):null,
      signal_rate:this.history.length?
        this.prequential.eligible/this.history.length:null,
      notice:"高密度推測方向不是保證命中；B/P/T 無法提供實際牌張與可驗證正EV。"
    };
  }
  _decay(m){
    // 每局 A = lambda*A+(1-lambda)*I；b = lambda*b
    const lam=this.lambda;
    for(let i=0;i<K;i++){
      m.b[i]*=lam;
      for(let j=0;j<K;j++)m.A[i][j]*=lam;
      m.A[i][i]+=1-lam;
    }
  }
  /** Prequential：先取得上一局預測，等使用者按 B/P/T 後才結算、再訓練。 */
  observe(outcome){
    if(!valid(outcome))throw Error("invalid_outcome");
    const prior=this.predict(),x=this._x(prior.context);
    this.last_signal=prior.direction;
    if(prior.bet_allowed){
      this.recommendationStreak=prior.direction===this.recommendationArm?
        this.recommendationStreak+1:1;
      this.recommendationArm=prior.direction;
      this.prequential.eligible++;
      if(outcome==="T")this.prequential.ties++;
      else if(prior.direction===outcome)this.prequential.correct++;
      else this.prequential.wrong++;
    }else{
      this.recommendationArm=null;this.recommendationStreak=0;
      this.prequential.skipped++;
    }
    // T 無法判定 B/P 預測對錯。保留剩餘的一局震盪過濾資格直到 B/P 開獎。
    if(outcome!=="T"){
      const failed=prior.bet_allowed&&prior.direction!==outcome;
      this.gateCooldown=failed?this.lossGate:0; // 成功立即歸零；Skip 亦只維持一局
      this.last_settlement=prior.bet_allowed?(failed?"loss":"win"):"skipped";
    }else{
      this.last_settlement="tie";
    }
    for(const arm of ["B","P"])this._decay(this.arms[arm]);
    if(outcome!=="T"){
      // 同一筆已知結果揭露兩個 arm 的 full-information 標籤。
      // 非數學反對稱：獲勝 +1、失敗 -2，A 雙邊相同但 b 不互為負數。
      for(const arm of ["B","P"]){
        const m=this.arms[arm],reward=arm===outcome?1:-2;
        for(let i=0;i<K;i++){
          m.b[i]+=reward*x[i];
          for(let j=0;j<K;j++)m.A[i][j]+=x[i]*x[j];
        }
      }
      this.trainedRounds++;
    }
    this.history.push(outcome);
    this.physics.observe(outcome);
    return this.predict();
  }
  /** 歷史/返回上一局採可重現回放；不會因點擊分析而重複訓練。 */
  syncHistory(seq){
    if(!Array.isArray(seq)||seq.some(v=>!valid(v)))throw Error("invalid_history");
    let same=this.history.length<=seq.length;
    if(same)for(let i=0;i<this.history.length;i++)
      if(this.history[i]!==seq[i]){same=false;break;}
    if(!same)this.reset();
    for(let i=this.history.length;i<seq.length;i++)this.observe(seq[i]);
    return this.predict();
  }
  static fromHistory(seq,options={}){return new OnlineLinUCB256(options).syncHistory(seq);}
}
return{OnlineLinUCB256,BlindPhysicsEstimator,context256,windowStats,
  DIM,SHOE_DIM:SHOE,ACTIVE_INDICES:ACTIVE.slice(),VERSION};
});