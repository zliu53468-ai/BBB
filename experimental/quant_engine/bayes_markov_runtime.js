/**
 * BBBQuantEngine — 使用者指定公式的獨立、未校準實驗核心。
 * B/P/T 不足以識別真實牌階；本類的 counts/ratios/TC 全是啟發式估計。
 * 不自動覆寫 Frozen Core，也不將未知對子當成「沒有對子」。
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.BBBQuantEngine=api.BBBQuantEngine;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const INITIAL=Object.freeze({low:128,mid:96,six:32,neutral:32,zero:128}),GROUPS=Object.keys(INITIAL),BASE=Object.freeze({B:.4586,P:.4462,T:.0952}),SCHEMA='BBB_QUANT_DYNAMIC_128_V2';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),finite=Number.isFinite;
class BBBQuantEngine {
 constructor(options={}){
  this.config={sensitivity:.0045,max_count_weight:.7,count_weight_gain:.15,min_abs_count:2,min_ev_exclusive:.005,...options};
  for(const key of ['sensitivity','max_count_weight','count_weight_gain','min_abs_count','min_ev_exclusive'])if(!finite(this.config[key])||this.config[key]<0)throw Error('invalid_quant_config:'+key);
  if(this.config.max_count_weight>.7||this.config.sensitivity>.0045||this.config.min_abs_count<2||this.config.min_ev_exclusive<.005)throw Error('quant_config_outside_release_bounds');
  this.shoeNumber=0;this.reset();
 }
 reset(){this.shoeNumber++;this.estimatedCounts={...INITIAL};this.remainingCards=416;this.runningCount=0;this.trueCount=0;this.pairCount=0;this.l6Index=0;this.history=[];this.lastSide=null;this.streak=0;this.knownPairRounds=0;this.unknownPairRounds=0;this.lastUpdate=null;this._ratios();return this;}
 _ratios(){this.remainingCards=GROUPS.reduce((s,k)=>s+this.estimatedCounts[k],0);this.cardRatios=Object.fromEntries(GROUPS.map(k=>[k,this.estimatedCounts[k]/INITIAL[k]]));this.cardDensities=Object.fromEntries(GROUPS.map(k=>[k,this.remainingCards?this.estimatedCounts[k]/this.remainingCards:0]));}
 /** 有容量限制的期望扣牌：不允許負牌數，溢出按剩餘容量重分配。 */
 _consume(draws,weights){let left=Math.min(draws,this.remainingCards);const deducted=Object.fromEntries(GROUPS.map(k=>[k,0]));for(let step=0;step<12&&left>1e-10;step++){const available=GROUPS.filter(k=>this.estimatedCounts[k]>1e-12);if(!available.length)break;let total=available.reduce((s,k)=>s+weights[k],0);const fallback=total<=0;if(fallback)total=available.reduce((s,k)=>s+this.estimatedCounts[k],0);const before=left;for(const k of available){const share=before*(fallback?this.estimatedCounts[k]:weights[k])/total,take=Math.min(this.estimatedCounts[k],share);this.estimatedCounts[k]-=take;deducted[k]+=take;left-=take;}}if(left>1e-7)throw Error('estimated_card_conservation_failed');return deducted;}
 /** pair 可選：只有實際知道 true/false 時才更新對子計數；未知不作標籤。 */
 updateRound(outcome,observation={}){
  if(!['B','P','T'].includes(outcome))throw Error('outcome must be B, P or T');
  if(observation.pair!==undefined&&observation.pair!==null&&typeof observation.pair!=='boolean')throw Error('pair must be boolean or unknown');
  if(outcome!=='T'){this.streak=outcome===this.lastSide?this.streak+1:1;this.lastSide=outcome;}
  const draws=outcome==='T'?4.5:4.94,impact=outcome!=='T'&&this.streak>3?.7:1,conditional=outcome==='B'?{low:.45,mid:.35,six:0,neutral:0,zero:.20}:outcome==='P'?{low:.40,mid:.45,six:0,neutral:0,zero:.15}:{low:.30,mid:.30,six:0,neutral:0,zero:.40};
  // 退火的是條件組成影響，不把實際期望消耗4.94誤減成3.458張。
  const neutral=Object.fromEntries(GROUPS.map(k=>[k,this.estimatedCounts[k]/this.remainingCards])),weights=Object.fromEntries(GROUPS.map(k=>[k,impact*conditional[k]+(1-impact)*neutral[k]])),before=this.remainingCards,deducted=this._consume(draws,weights);
  this.runningCount+=draws*(conditional.low-conditional.mid)*impact;
  if(typeof observation.pair==='boolean'){this.pairCount+=observation.pair?-2:.1;this.knownPairRounds++;}else this.unknownPairRounds++;
  if(outcome==='P')this.l6Index+=.2;else if(outcome==='B')this.l6Index-=.1;
  this.history.push(outcome);this._ratios();this.trueCount=clamp(this.runningCount/(this.remainingCards/52),-8,8);
  const update={outcome,expected_draws:draws,composition_impact:impact,deducted,remaining_before:before,remaining_after:this.remainingCards,estimated_only:true,reset_triggered:this.remainingCards<=60};
  if(update.reset_triggered){update.reason='estimated_threshold_not_observed_shuffle';this.reset();}
  this.lastUpdate=update;return update;
 }
 /**
  * 完整128D候選schema。它與既有Core的128D順序不同，不能冒充相容。
  * 16個主狀態＋5群×8基底＋8個TC區域×3＋8個進度區域×3＋8窗口×3=128。
  */
 generateShoeContext(){
  const rr=this.remainingCards/416,progress=1-rr,n=this.history.length,tc=this.trueCount,v=[Math.pow(tc,2)*Math.sign(tc),rr,progress,clamp(n/70,0,1),tc/8,Math.tanh(this.runningCount/32),Math.tanh(this.pairCount/10),Math.tanh(this.l6Index/8),this.lastSide==='B'?1:this.lastSide==='P'?-1:0,this.history.at(-1)==='T'?1:0,clamp(this.streak/70,0,1),n?this.knownPairRounds/n:0,n?this.unknownPairRounds/n:0,1,this.cardDensities.low-this.cardDensities.zero,this.cardDensities.six-this.cardDensities.neutral];
  for(const k of GROUPS){const r=this.cardRatios[k],d=this.cardDensities[k],initial=INITIAL[k]/416;v.push(r,d,d/initial,this.estimatedCounts[k]/416,r*r,d*d,d*d*d,d-initial);}
  for(const center of [-7,-5,-3,-1,1,3,5,7]){const z=(tc-center)/2;v.push(clamp(1-Math.abs(z),0,1),clamp((tc-center)/16,0,1),Math.exp(-.5*z*z));}
  for(let i=0;i<8;i++){const center=(i+.5)/8,z=(progress-center)*8;v.push(clamp(1-Math.abs(z),0,1),Math.exp(-.5*z*z),clamp((progress-center)*2,0,1));}
  for(const window of [3,5,8,12,16,24,48,70]){const h=this.history.slice(-window);for(const side of ['B','P','T'])v.push(h.length?h.filter(x=>x===side).length/h.length:0);}
  if(v.length!==128||!v.every(finite))throw Error('invalid128D');return v;
 }
 /** 顯式實驗轉接器；預設拒絕把新schema塞給舊權重。 */
 injectShoeContext(context256,{allowUncalibrated=false}={}){if(!allowUncalibrated)throw Error('128D schema differs from Frozen Core; explicit experimental opt-in and validation required');if(!Array.isArray(context256)||context256.length!==256||!context256.every(finite))throw Error('expected256D');return [...this.generateShoeContext(),...context256.slice(128)];}
 static fractionalKelly(pWin,pLoss,payout){if(![pWin,pLoss,payout].every(finite)||pWin<0||pLoss<0||payout<=0||pWin+pLoss<=0||pWin+pLoss>1+1e-12)throw Error('invalid Kelly inputs');const ev=payout*pWin-pLoss,raw=ev>0?.25*ev/(payout*(pWin+pLoss)):0;return {ev,raw_fraction:raw,bet_fraction:ev>0?clamp(raw,.05,.30):0,minimum_clip_applied:ev>0&&raw<.05};}
 _side(signal,probability,payout){const known=finite(probability)&&probability>=0&&probability<=1&&finite(payout)&&payout>0,ev=known?probability*payout-(1-probability):null;return {signal,ev,bet_allowed:signal&&known&&ev>0,stake_fraction:0,reason:!known?'requires_calibrated_probability_and_payout':!signal?'index_not_triggered':ev<=0?'nonpositive_ev':'positive_ev_signal_stake_not_allocated'};}
 predictNext(sideInputs={}){
  const tc=this.trueCount,last=this.history.slice(-3).join(''),weight=Math.min(this.config.max_count_weight,Math.abs(tc)*this.config.count_weight_gain);let shift=tc*this.config.sensitivity;
  if(last==='BBB')shift+=tc>0?.005:tc<0?-.002:0;
  if(last==='PPP')shift+=tc<0?-.005:tc>0?.002:0;
  const adjustedB=clamp(BASE.B+shift,0,1-BASE.T),adjustedP=1-BASE.T-adjustedB;
  const probabilities={B:(1-weight)*BASE.B+weight*adjustedB,P:(1-weight)*BASE.P+weight*adjustedP,T:BASE.T};
  const banker=BBBQuantEngine.fractionalKelly(probabilities.B,probabilities.P,.95),player=BBBQuantEngine.fractionalKelly(probabilities.P,probabilities.B,1),best=banker.ev>=player.ev?'B':'P',chosen=best==='B'?banker:player,eligible=Math.abs(tc)>=this.config.min_abs_count&&chosen.ev>this.config.min_ev_exclusive,action=eligible?best:'Skip';
  // B/P/T不能識別莊對、閒對或幸運六；不以啟發式指數冒充校準機率。
  const side=(signal,input)=>input?.calibrated===true?this._side(signal,input.probability,input.netPayout):this._side(signal,null,null);
  return {version:'BBB_QUANT_DYNAMIC_V2',experimental:true,validated:false,probabilities,w_count:weight,count_shift_before_mix:shift,ev_banker:banker.ev,ev_player:player.ev,direction:action,direction_text:action==='Skip'?'觀望':action==='B'?'莊':'閒',bet_fraction:eligible?chosen.bet_fraction:0,kelly_raw_fraction:eligible?chosen.raw_fraction:0,minimum_clip_applied:eligible&&chosen.minimum_clip_applied,eligibility:{count_ok:Math.abs(tc)>=this.config.min_abs_count,ev_ok:chosen.ev>this.config.min_ev_exclusive,min_abs_count:this.config.min_abs_count,min_ev_exclusive:this.config.min_ev_exclusive},
   side_bets:{banker_pair:side(this.pairCount>3.5,sideInputs.bankerPair),player_pair:side(this.pairCount>3.5,sideInputs.playerPair),tie:side(tc>=5,sideInputs.tie),lucky_six:side(this.l6Index>4&&tc>=-4&&tc<=-2,sideInputs.luckySix)},
   estimated_shoe:{remaining_cards:this.remainingCards,cardRatios:{...this.cardRatios},cardDensities:{...this.cardDensities},runningCount:this.runningCount,trueCount:tc,pairCount:this.pairCount,l6Index:this.l6Index,pair_observation_known:this.knownPairRounds,estimated_only:true},context_schema:SCHEMA,notice:'B/P/T-only heuristic; probabilities and EV are unvalidated estimates, not measured betting edges'};
 }
 static fromHistory(history,options={}){const e=new BBBQuantEngine(options);for(const x of history)e.updateRound(x);return e;}
}
return {BBBQuantEngine,CONTEXT_SCHEMA:SCHEMA,INITIAL_COUNTS:INITIAL,BASE_PROBABILITIES:BASE};});
