/**
 * BBB 256D Online LinUCB V1 — B/P/T-only, browser-safe.
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
const K=ACTIVE.length,VERSION="BBB_QUANT256_ONLINE_V1";
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
  return{counts:c,draws:cards,result:b>p?"B":p>b?"P":"T"};
}
/**
 * 不可辨識性聲明：B/P/T 不提供實際 4/5/6 張或點數。
 * 這裡僅從合法發牌模擬抽取「符合已知結果的可能路徑」做 latent ensemble；
 * 它不是對真實牌靴的測量，亦不代表已知任何 A~K 殘量。
 */
class BlindPhysicsEstimator{
  constructor({particles=24,seed=0xBBB256}={}){
    this.size=clamp(Math.floor(particles)||24,8,96);
    this.seed=seed>>>0;this.reset();
  }
  reset(){
    this.rng=seededRng(this.seed);
    this.particles=Array.from({length:this.size},()=>BASE_POINTS.slice());
    this.drawHistory=[];this.lastAcceptance=0;return this;
  }
  observe(outcome){
    if(!valid(outcome))throw Error("invalid_outcome");
    let accepted=0,sum=0;
    for(let i=0;i<this.size;i++){
      const original=this.particles[i];let chosen=null;
      // 條件抽樣，拒絕不符合本局已知 B/P/T 的虛擬發牌結果。
      // 不編造實測卡牌；未抽中時不改寫粒子狀態。
      for(let tries=0;tries<128;tries++){
        const trial=sampleHand(original,this.rng);
        if(trial&&trial.result===outcome){chosen=trial;break;}
      }
      if(chosen){this.particles[i]=chosen.counts;sum+=chosen.draws;accepted++;}
    }
    this.lastAcceptance=accepted/this.size;
    this.drawHistory.push(accepted?sum/accepted:0);
    return this.snapshot();
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
    return{
      source:"simulated_unobserved",estimated_only:true,observed_cards:0,
      virtual_remaining:remain,virtual_high_ratio:highRatio,virtual_low_ratio:lowRatio,
      virtual_high_delta:highRatio-highBase,virtual_low_delta:lowRatio-lowBase,
      virtual_high_std:Math.sqrt(variance),conditional_acceptance:this.lastAcceptance,
      expected_draws_recent:recent.length?recent.reduce((s,v)=>s+v,0)/recent.length:4.94,
      actual_draw_count_known:false,actual_remaining_known:false
    };
  }
}
function context256(history,physics){
  if(!Array.isArray(history)||history.some(v=>!valid(v)))throw Error("invalid_history");
  const shoe=Array(SHOE).fill(0),road=Array(SHOE).fill(0);
  const s3=windowStats(history,3),s8=windowStats(history,8),s16=windowStats(history,16);
  const nonTie=bp(history),n=history.length,ties=history.slice(-8).filter(x=>x==="T").length;
  const sim=physics.snapshot(),last=nonTie.at(-1)||"";
  // 128D: 僅可觀測的進程 + 明確標示虛擬/低權重的物理先驗。
  shoe[0]=.7;shoe[1]=clamp((n-30)/70,-.5,.6);
  shoe[2]=clamp(ties/8-.095,-.2,.5);
  shoe[3]=clamp(n/75,0,1);
  shoe[4]=clamp((sim.virtual_remaining/DECK-.5)*.6,-.3,.3);
  shoe[5]=clamp(sim.virtual_high_delta*2,-.1,.1); // 虛擬，保守夾限
  shoe[6]=clamp(sim.virtual_low_delta*2,-.1,.1);  // 虛擬，保守夾限
  shoe[7]=clamp((sim.expected_draws_recent-4.94)/5,-.2,.2); // 非實測4/6張
  shoe[8]=0; // 無觀測卡牌，不可把估計視為信心
  shoe[9]=clamp((s8.banker-.5)*.6,-.3,.3);
  shoe[10]=clamp(s8.entropy-.5,-.5,.5);
  shoe[11]=clamp((s3.entropy-s8.entropy)*.5,-.5,.5);
  // 128D: B/P/T 路單，所有有效視窗 <= 32 局。短窗 3 vs 8 差分。
  road[0]=last==="B"?.5:last==="P"?-.5:0;
  road[1]=clamp(s8.run/10,0,1);
  road[2]=clamp(s3.turn-.5,-.5,.5);
  road[3]=clamp(s8.turn-.5,-.5,.5);
  road[4]=clamp(s3.turn-s8.turn,-1,1);
  road[5]=s3.regularity-.5;
  road[6]=s8.regularity-.5;
  road[7]=clamp(s3.regularity-s8.regularity,-1,1);
  road[8]=s3.entropy-.5;
  road[9]=s8.entropy-.5;
  road[10]=clamp(s3.entropy-s8.entropy,-1,1);
  road[11]=s3.banker-.5;
  road[12]=s8.banker-.5;
  road[13]=clamp(s3.banker-s8.banker,-1,1);
  road[14]=clamp(s16.banker-.5,-.5,.5);
  road[15]=clamp(s8.turn-s16.turn,-1,1);
  road[16]=clamp(s8.run/6-.5,-.5,.5);
  road[17]=clamp(s16.regularity-.5,-.5,.5);
  road[18]=clamp((s8.entropy-s16.entropy),-1,1);
  road[19]=clamp(ties/8-.095,-.2,.5);
  road[20]=nonTie.length>=2&&nonTie.at(-1)===nonTie.at(-2)?.4:-.4;
  road[21]=clamp(s3.count/3,0,1);
  road[22]=clamp(s8.count/8,0,1);
  road[23]=clamp(s16.count/16,0,1);
  const vector=shoe.concat(road);
  if(vector.length!==DIM||vector.some(x=>!finite(x)))throw Error("invalid_256D_context");
  return{vector,shoe,road,physics:sim,features:{regularity3:s3.regularity,
    regularity8:s8.regularity,regularity_delta:s3.regularity-s8.regularity,
    entropy3:s3.entropy,entropy8:s8.entropy,entropy_delta:s3.entropy-s8.entropy,
    entropy_volatility:Math.abs(s3.entropy-s8.entropy),
    observed_bp:nonTie.length}};
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
  constructor({lambda=.94,alpha=.15,minSamples=12,confidenceThreshold=.025,particles=24}={}){
    if(!finite(lambda)||lambda<.90||lambda>.96)throw Error("lambda_must_be_0.90_to_0.96");
    if(!finite(alpha)||alpha<0||alpha>2)throw Error("invalid_alpha");
    if(!finite(confidenceThreshold)||confidenceThreshold<=0||confidenceThreshold>.5)throw Error("invalid_confidence_threshold");
    this.lambda=lambda;this.alpha=alpha;this.minSamples=minSamples;
    this.confidenceThreshold=confidenceThreshold;this.particleCount=particles;
    this.reset();
  }
  reset(){
    this.history=[];this.arms={B:makeArm(),P:makeArm()};
    this.physics=new BlindPhysicsEstimator({particles:this.particleCount});
    this.trainedRounds=0;this.prequential={eligible:0,correct:0,wrong:0,ties:0};
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
    // 3 vs 8 局震盪越明顯，提高 Skip 閘門，禁止假定跳路轉折必勝。
    const shock=c.features.entropy_volatility;
    const windowShock=Math.abs(windowStats(this.history,8).turn-windowStats(this.history,16).turn);
    const threshold=clamp(this.confidenceThreshold*(1+shock*.9+windowShock*.4),
      this.confidenceThreshold,this.confidenceThreshold*2.3);
    const enough=this.trainedRounds>=this.minSamples;
    const allowed=enough&&delta>=threshold;
    const rawDirection=banker.ucb>=player.ucb?"B":"P";
    return{
      version:VERSION,direction:allowed?rawDirection:"Skip",raw_direction:rawDirection,
      bet_allowed:allowed,reason:!enough?"warmup":!allowed?"volume_guard":"model_score_gap",
      ucb_banker:banker.ucb,ucb_player:player.ucb,delta,confidence_threshold:threshold,
      arms:{B:banker,P:player},history_length:this.history.length,
      trained_rounds:this.trainedRounds,lambda:this.lambda,alpha:this.alpha,
      context:c.vector,shoe_context:c.shoe,road_context:c.road,
      active_dimensions:K,features:c.features,blind_physics:c.physics,
      calibrated_probability:false,claimed_edge:false,
      prequential:{...this.prequential},
      notice:"UCB 分數差並非真實勝率或正 EV；僅 B/P/T 不能推算實際剩餘牌張。"
    };
  }
  _decay(m){
    // A <- lambda*A + (1-lambda)*I；b <- lambda*b
    const lam=this.lambda;
    for(let i=0;i<K;i++){
      m.b[i]*=lam;
      for(let j=0;j<K;j++)m.A[i][j]*=lam;
      m.A[i][i]+=1-lam;
    }
  }
  /** 先對上一局作 prequential 評估，再用結算結果更新；不偷看未來。 */
  observe(outcome){
    if(!valid(outcome))throw Error("invalid_outcome");
    const prior=this.predict(),x=this._x(prior.context);
    if(prior.bet_allowed){
      this.prequential.eligible++;
      if(outcome==="T")this.prequential.ties++;
      else if(prior.direction===outcome)this.prequential.correct++;
      else this.prequential.wrong++;
    }
    for(const arm of ["B","P"])this._decay(this.arms[arm]);
    if(outcome!=="T"){
      // B/P 結算可同時揭示兩臂的全回饋標籤，這是 full-information LinUCB。
      for(const arm of ["B","P"]){
        const m=this.arms[arm],reward=arm===outcome?1:0;
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
  /** 本機不持久化 A/b：刷新重播歷史，返回上一局會重建，不會重複學習。 */
  syncHistory(seq){
    if(!Array.isArray(seq)||seq.some(v=>!valid(v)))throw Error("invalid_history");
    let same=this.history.length<=seq.length;
    if(same)for(let i=0;i<this.history.length;i++)if(this.history[i]!==seq[i]){same=false;break;}
    if(!same)this.reset();
    for(let i=this.history.length;i<seq.length;i++)this.observe(seq[i]);
    return this.predict();
  }
  static fromHistory(seq,options={}){return new OnlineLinUCB256(options).syncHistory(seq);}
}
return{OnlineLinUCB256,BlindPhysicsEstimator,context256,windowStats,
  DIM,SHOE_DIM:SHOE,ACTIVE_INDICES:ACTIVE.slice(),VERSION};
});