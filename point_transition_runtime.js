/** 點數時間序列模型：僅處理雙方最終總點，沒有任何牌組或扣牌輸入。 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 else root.BBBPointTransitions=api;
})(typeof self!=='undefined'?self:this,function(){
 'use strict';
 const VERSION='BBB_POINT_TRANSITIONS_ADAPTIVE_V1';
 // 保留既有儲存版本，避免已累積的總點數資料因 V4 升級而無法恢復。
 const MODEL_REVISION='TOTAL_POINTS_V4_DYNAMIC_1024';
 const DEFAULTS=Object.freeze({
  // 最低樣本限制不會因連續觀望而取消，避免「保底」憑空製造高信心。
  minSamples:12,minDecisive:8,priorPerOutcome:.5,
  // 依「累積輸入筆數」調整，並非牌張數或任何物理發牌狀態。
  baseGap:.12,minGap:.04,gapDecayPerRound:.0015,
  // 已連續觀望超過六次後，下一次評估暫時降低為正常門檻的七折。
  relaxation:.7,relaxAfterSkips:6,
  particleCount:1024,particleStdMax:.08,particleOverrideGap:.045,
  windows:Object.freeze([240,720,2160,0]),retainedEvents:2161,
  targetWindow:6,targetSignalsMin:1,targetSignalsMax:2
 });
 const valid=x=>Number.isInteger(x)&&x>=0&&x<=9;
 const outcome=state=>state%10>Math.floor(state/10)?'B':state%10<Math.floor(state/10)?'P':'T';
 function encode(player,banker){if(!valid(player)||!valid(banker))throw Error('閒、莊必須各為 0～9 的整數總點');return player*10+banker;}

 class StateTransitionMatrix{
  constructor(saved={}){
   this.rows=Object.create(null);
   for(const [key,values] of Object.entries(saved)){
    if(!/^(0|[1-9][0-9]?)$/.test(key)||!Array.isArray(values)||values.length!==100||
       values.some(v=>!Number.isSafeInteger(v)||v<0))throw Error('歷史轉移矩陣格式錯誤');
    this.rows[key]=values.slice();
   }
  }
  // 第一維是前局的 100 種點數組合，第二維是下一局的 100 種組合。
  increment(from,to,amount=1){
   if(!Number.isInteger(from)||from<0||from>99||!Number.isInteger(to)||to<0||to>99)throw Error('轉移狀態錯誤');
   const row=this.rows[from]||(this.rows[from]=Array(100).fill(0));
   if(!Number.isSafeInteger(row[to]+amount)||row[to]+amount<0)throw Error('轉移次數錯誤');
   row[to]+=amount;
   if(amount<0&&row.every(n=>n===0))delete this.rows[from];
  }
  row(state){return (this.rows[state]||Array(100).fill(0)).slice();}
  probabilities(state){const row=this.row(state),n=row.reduce((a,b)=>a+b,0);return n?row.map(v=>v/n):null;}
  snapshot(){return Object.fromEntries(Object.entries(this.rows).map(([k,v])=>[k,v.slice()]));}
 }


 // 重現性亂數來源：相同觀測頻率每次重繪得到相同的可信度數值。
 // 不借助牌數、牌組組成或任何發牌物理資訊。
 function seededRandom(seed){
  let s=seed>>>0;
  return function(){
   s=(s+0x6D2B79F5)>>>0;
   let t=s;t=Math.imul(t^(t>>>15),t|1);
   t^=t+Math.imul(t^(t>>>7),t|61);
   return ((t^(t>>>14))>>>0)/4294967296;
  };
 }
 function frequencySeed(counts){
  let seed=2166136261;
  for(const value of [counts.B,counts.P,counts.T]){
   seed=Math.imul(seed^(value>>>0),16777619)>>>0;
  }
  return seed||1;
 }

 class ParticleConfidenceSampler{
  constructor(config){this.config=config;}
  // 1024 個「條件機率後驗粒子」，不是實體牌組粒子。
  // 將 100 個下一局總點狀態的頻率依莊／閒／和聚合，再以
  // Dirichlet(B+0.5, P+0.5, T+0.5) 產生 1024 個可能的下一局機率向量。
  // 標準差代表歷史估計的不確定性，不代表真實未來局的勝率保證。
  sample(counts){
   const next=seededRandom(frequencySeed(counts));
   const normal=()=>{
    const u=Math.max(next(),1e-12),v=next();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
   };
   const gamma=shape=>{
    if(shape<1)return gamma(shape+1)*Math.pow(Math.max(next(),1e-12),1/shape);
    const d=shape-1/3,c=1/Math.sqrt(9*d);
    for(let tries=0;tries<1000;tries++){
     const x=normal(),v0=1+c*x;
     if(v0<=0)continue;
     const v=v0*v0*v0,u=Math.max(next(),1e-12);
     if(u<1-.0331*x*x*x*x||Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return d*v;
    }
    throw Error('後驗粒子採樣未收斂');
   };
   let meanB=0,meanP=0,meanGap=0,m2B=0,m2P=0,m2Gap=0;
   for(let i=1;i<=this.config.particleCount;i++){
    const b=gamma(counts.B+this.config.priorPerOutcome),
     p=gamma(counts.P+this.config.priorPerOutcome),
     t=gamma(counts.T+this.config.priorPerOutcome),sum=b+p+t;
    const pb=b/sum,pp=p/sum,gap=pb-pp;
    const db=pb-meanB;meanB+=db/i;m2B+=db*(pb-meanB);
    const dp=pp-meanP;meanP+=dp/i;m2P+=dp*(pp-meanP);
    const dg=gap-meanGap;meanGap+=dg/i;m2Gap+=dg*(gap-meanGap);
   }
   const denominator=Math.max(1,this.config.particleCount-1);
   return {count:this.config.particleCount,stdBanker:Math.sqrt(m2B/denominator),
    stdPlayer:Math.sqrt(m2P/denominator),stdGap:Math.sqrt(m2Gap/denominator)};
  }
 }

 class StatisticalV4Filter{
  constructor(config){this.config=config;this.sampler=new ParticleConfidenceSampler(config);}
  evaluate(row,relaxed,currentRound=0){
   const c=this.config,n=row.reduce((a,b)=>a+b,0),counts={B:0,P:0,T:0};
   row.forEach((v,state)=>counts[outcome(state)]+=v);
   const decisive=counts.B+counts.P,prior=c.priorPerOutcome,total=n+3*prior;
   // 平滑三種結果，避免小樣本出現 0%／100% 的誤導性機率。
   const p={B:(counts.B+prior)/total,P:(counts.P+prior)/total,T:(counts.T+prior)/total};
   const delta=p.B-p.P,gap=Math.abs(delta);
   const se=Math.sqrt(Math.max(0,p.B+p.P-delta*delta)/total);
   // currentRound 只看已累積的總點數輸入次數，不涉及實體局序或殘牌。
   const round=Math.max(0,Math.floor(Number.isFinite(currentRound)?currentRound:0));
   const standardThreshold=Math.max(c.minGap,c.baseGap-round*c.gapDecayPerRound);
   const threshold=standardThreshold*(relaxed?c.relaxation:1);
   const sufficient=n>=c.minSamples&&decisive>=c.minDecisive;
   const particles=sufficient?this.sampler.sample(counts):null;
   const lowVariance=Boolean(particles&&
    Math.max(particles.stdBanker,particles.stdPlayer)<c.particleStdMax);
   const standardPass=sufficient&&gap>threshold;
   // 低變異例外仍需樣本充分、差值超過 4.5%，不可將低標準差當成必贏。
   const variancePass=sufficient&&lowVariance&&gap>c.particleOverrideGap;
   const pass=standardPass||variancePass;
   const direction=pass?(delta>0?'B':'P'):'SKIP';
   const signalType=!pass?null:standardPass?'dynamic_gap':'low_variance_override';
   return {direction,counts,sampleSize:n,decisiveSamples:decisive,probability:p,
    gap,standardError:se,standardThreshold,threshold,currentRound:round,
    particleStd:particles,particleStdMax:c.particleStdMax,lowVariance,signalType,relaxed,
    reason:!sufficient?'insufficient_samples':!pass?'gap_below_threshold':
      standardPass?'dynamic_gap_signal':'low_variance_signal',
    statisticallyCalibrated:false,validatedAccuracyGain:false};
  }
 }

 class AdaptiveObservationController{
  constructor(config){this.config=config;}
  choose(matrix,events,state,skips){
   const c=this.config,relaxed=skips>c.relaxAfterSkips;
   // 先掃短窗口。連續觀望時起點擴至下一層；不足樣本仍繼續擴大。
   const start=relaxed?1:0;
   let selected;
   for(let w=start;w<c.windows.length;w++){
    const depth=c.windows[w],row=depth===0?matrix.row(state):Array(100).fill(0);
    if(depth!==0){
     const first=Math.max(1,events.length-depth);
     for(let j=first;j<events.length;j++){
      if(!events[j].breakBefore&&events[j-1].state===state)row[events[j].state]++;
     }
    }
    const sampleSize=row.reduce((a,b)=>a+b,0);
    selected={row,depth,relaxed};
    // 只使用完全相同的起點總點組合，不偷偷退回全域勝負頻率。
    if(sampleSize>=c.minSamples)break;
   }
   return selected;
  }
 }

 class PointTransitionPredictor{
  constructor(saved=null){
   this.config=DEFAULTS;this.filter=new StatisticalV4Filter(DEFAULTS);
   this.controller=new AdaptiveObservationController(DEFAULTS);
   this.matrix=new StateTransitionMatrix();this.events=[];this.totalRounds=0;
   this.consecutiveSkips=0;this.nextIsBreak=false;
   if(saved!==null)this.restore(saved);
  }
  restore(saved){
   if(saved.version!==VERSION||!Number.isSafeInteger(saved.totalRounds)||saved.totalRounds<0||
    !Array.isArray(saved.events)||saved.events.length>DEFAULTS.retainedEvents||
    saved.events.length>saved.totalRounds||!Number.isSafeInteger(saved.consecutiveSkips)||saved.consecutiveSkips<0)
    throw Error('點數歷史格式或版本錯誤，請匯出檢查後再清除');
   if(saved.events.some(e=>!Number.isInteger(e.state)||e.state<0||e.state>99||typeof e.breakBefore!=='boolean'||
    !Number.isSafeInteger(e.skipsAfter)||e.skipsAfter<0||!e.forecast||!['B','P','SKIP'].includes(e.forecast.direction)))throw Error('逐局歷史格式錯誤');
   if(saved.totalRounds>0&&!saved.events.length)throw Error('歷史紀錄缺少最近狀態');
   this.matrix=new StateTransitionMatrix(saved.matrix);this.events=JSON.parse(JSON.stringify(saved.events));
   this.totalRounds=saved.totalRounds;this.consecutiveSkips=saved.consecutiveSkips;this.nextIsBreak=Boolean(saved.nextIsBreak);
  }
  // 這是純讀取方法。重新整理或重繪不會增加觀望次數，也不會重複學習。
  forecast(){
   if(!this.events.length)return null;
   const current=this.events[this.events.length-1];
   const chosen=this.controller.choose(this.matrix,this.events,current.state,this.consecutiveSkips);
   const result=this.filter.evaluate(chosen.row,chosen.relaxed,this.totalRounds);
   const predictions=this.events.slice(-this.config.targetWindow).map(e=>e.forecast?.direction);
   const recentSignals=predictions.filter(d=>d==='B'||d==='P').length;
   return {...result,version:VERSION,state:{player:Math.floor(current.state/10),banker:current.state%10},
    scanDepth:chosen.depth,consecutiveSkipsBefore:this.consecutiveSkips,
    recentSignals,targetWindow:this.config.targetWindow,targetSignals:[1,2],
    frequencyTargetGuaranteed:false,totalRounds:this.totalRounds};
  }
  observe(player,banker){
   const state=encode(player,banker),previous=this.events[this.events.length-1],breakBefore=this.nextIsBreak||!previous;
   if(previous&&!breakBefore){
    // 先結算「上一筆對這一筆」的方向，再加入已發生的轉移；不使用未知下一局。
    previous.actualNext=outcome(state);this.matrix.increment(previous.state,state);
   }
   this.events.push({state,breakBefore,forecast:{direction:'SKIP'},skipsAfter:0,actualNext:null});
   if(this.events.length>this.config.retainedEvents)this.events.shift();
   this.totalRounds++;this.nextIsBreak=false;
   const result=this.forecast();
   this.consecutiveSkips=result.direction==='SKIP'?this.consecutiveSkips+1:0;
   const event=this.events[this.events.length-1];event.forecast=result;event.skipsAfter=this.consecutiveSkips;
   result.recentSignals=this.events.slice(-this.config.targetWindow).filter(e=>e.forecast.direction!=='SKIP').length;
   return {...result,consecutiveSkipsAfter:this.consecutiveSkips};
  }
  latest(){return this.events.length?JSON.parse(JSON.stringify(this.events[this.events.length-1].forecast)):null;}
  undo(){
   if(!this.events.length)return null;
   // 最舊明細已裁掉時不允許回退到無法恢復的邊界。
   if(this.events.length===1&&this.totalRounds>1)throw Error('已達可返回的歷史邊界');
   const last=this.events.pop(),previous=this.events[this.events.length-1];
   if(previous&&!last.breakBefore){this.matrix.increment(previous.state,last.state,-1);previous.actualNext=null;}
   this.totalRounds--;this.consecutiveSkips=previous?previous.skipsAfter:0;this.nextIsBreak=false;
   return this.latest();
  }
  startNewSequence(){this.nextIsBreak=true;this.consecutiveSkips=0;}
  snapshot(){return {version:VERSION,matrix:this.matrix.snapshot(),events:JSON.parse(JSON.stringify(this.events)),
   totalRounds:this.totalRounds,consecutiveSkips:this.consecutiveSkips,nextIsBreak:this.nextIsBreak};}
 }
 return Object.freeze({VERSION,MODEL_REVISION,DEFAULTS,encode,outcome,StateTransitionMatrix,ParticleConfidenceSampler,StatisticalV4Filter,
  AdaptiveObservationController,PointTransitionPredictor});
});
