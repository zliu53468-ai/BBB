/** 點數時間序列模型：僅處理雙方最終總點，沒有任何牌組或扣牌輸入。 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 else root.BBBPointTransitions=api;
})(typeof self!=='undefined'?self:this,function(){
 'use strict';
 const VERSION='BBB_POINT_TRANSITIONS_ADAPTIVE_V1';
 const DEFAULTS=Object.freeze({minSamples:12,minDecisive:8,baseGap:.06,z:1.96,
  relaxation:.75,relaxAfterSkips:4,windows:Object.freeze([240,720,2160,0]),
  retainedEvents:2161,priorPerOutcome:.5,targetWindow:6,targetSignalsMin:1,targetSignalsMax:2});
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

 class StatisticalV4Filter{
  constructor(config){this.config=config;}
  evaluate(row,relaxed){
   const c=this.config,n=row.reduce((a,b)=>a+b,0),counts={B:0,P:0,T:0};
   row.forEach((v,state)=>counts[outcome(state)]+=v);
   const decisive=counts.B+counts.P,prior=c.priorPerOutcome,total=n+3*prior;
   // Jeffreys 型每結果 0.5 偽計數，避免全勝小樣本產生零誤差。
   const p={B:(counts.B+prior)/total,P:(counts.P+prior)/total,T:(counts.T+prior)/total};
   const delta=p.B-p.P,gap=Math.abs(delta);
   // X=+1（莊）、-1（閒）、0（和）：Var(X)=pB+pP-(pB-pP)^2。
   const se=Math.sqrt(Math.max(0,p.B+p.P-delta*delta)/total);
   const standardThreshold=Math.max(c.baseGap,c.z*se);
   const threshold=standardThreshold*(relaxed?c.relaxation:1);
   const sufficient=n>=c.minSamples&&decisive>=c.minDecisive;
   const direction=sufficient&&gap>threshold?(delta>0?'B':'P'):'SKIP';
   return {direction,counts,sampleSize:n,decisiveSamples:decisive,probability:p,
    gap,standardError:se,standardThreshold,threshold,relaxed,
    reason:!sufficient?'insufficient_samples':gap<=threshold?'gap_below_threshold':'transition_signal',
    // 自適應選窗與放寬會影響檢定覆蓋率；不能宣稱通過就是校準的 95% 保證。
    statisticallyCalibrated:false,validatedAccuracyGain:false};
  }
 }

 class AdaptiveObservationController{
  constructor(config){this.config=config;}
  choose(matrix,events,state,skips){
   const c=this.config,relaxed=skips>=c.relaxAfterSkips;
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
   const result=this.filter.evaluate(chosen.row,chosen.relaxed);
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
 return Object.freeze({VERSION,DEFAULTS,encode,outcome,StateTransitionMatrix,StatisticalV4Filter,
  AdaptiveObservationController,PointTransitionPredictor});
});
