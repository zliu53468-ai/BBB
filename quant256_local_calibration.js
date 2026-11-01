/**
 * Quant256 近 12 局局部線上 A/b 校準（不改寫既有 Frozen / Physics 主模型）。
 * 只使用實際結算前已知的 256D context，並在 B/P 結算後更新；
 * 和局不作 B/P 標籤。相同歷史可逐局回放，沒有未來資訊洩漏。
 * 這是未校準分數而非客觀勝率，也無法保證連敗次數上限。
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBLocalQuant256=api;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const VERSION="BBB_LOCAL_256D_WINDOW12_V1",WINDOW=12;
// 全部以 256D 向量取值，包含新增的 same-side / alternating 二階交互。
const ACTIVE=[...Array.from({length:12},(_,i)=>i),
  ...Array.from({length:30},(_,i)=>128+i)];
const K=ACTIVE.length;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function project(vector){
  if(!Array.isArray(vector)||vector.length!==256)throw Error("invalid_256D_feature");
  const scale=1/Math.sqrt(K);
  return Float64Array.from(ACTIVE,i=>{
    const v=Number(vector[i]);
    if(!Number.isFinite(v))throw Error("nonfinite_256D_feature");
    return clamp(v,-1,1)*scale;
  });
}
function makeMatrix(ridge){
  return Array.from({length:K},(_,i)=>Float64Array.from({length:K},(_,j)=>i===j?ridge:0));
}
/** 完整 42×42 SPD Cholesky；兩臂共享 full-information A，b 符號相反。 */
function solveSPD(A,v){
  const L=Array.from({length:K},()=>new Float64Array(K));
  for(let i=0;i<K;i++)for(let j=0;j<=i;j++){
    let s=A[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];
    if(i===j){if(!(s>1e-12))throw Error("local_cholesky_invalid");L[i][j]=Math.sqrt(s);}
    else L[i][j]=s/L[j][j];
  }
  const y=new Float64Array(K),x=new Float64Array(K);
  for(let i=0;i<K;i++){let s=v[i];for(let j=0;j<i;j++)s-=L[i][j]*y[j];y[i]=s/L[i][i];}
  for(let i=K-1;i>=0;i--){let s=y[i];for(let j=i+1;j<K;j++)s-=L[j][i]*x[j];x[i]=s/L[i][i];}
  return x;
}
class WindowLocalLinUCB{
  constructor({windowSize=WINDOW,ridge=1,alpha=.15,maxBlend=.35}={}){
    if(!Number.isInteger(windowSize)||windowSize<4||windowSize>64)throw Error("invalid_window_size");
    if(!Number.isFinite(ridge)||ridge<=0)throw Error("invalid_ridge");
    if(!Number.isFinite(alpha)||alpha<0||alpha>2)throw Error("invalid_alpha");
    if(!Number.isFinite(maxBlend)||maxBlend<0||maxBlend>.6)throw Error("invalid_blend_weight");
    this.windowSize=windowSize;this.ridge=ridge;this.alpha=alpha;this.maxBlend=maxBlend;
    this.reset();
  }
  reset(){this.observations=[];this.totalResults=0;return this;}
  /** 接收本局開獎前的 x 與結算 outcome，確保局部 A/b 絕不使用未來資訊。 */
  observe(preRound256,outcome){
    if(!["B","P","T"].includes(outcome))throw Error("invalid_outcome");
    const x=project(preRound256);
    this.observations.push({x,outcome});
    if(this.observations.length>this.windowSize)this.observations.shift();
    this.totalResults++;return this;
  }
  /** 用近 12 局的預測前 context 重新建立 A/b（真正滑動窗，過期資料完全移除）。 */
  matrices(){
    const A=makeMatrix(this.ridge),bB=new Float64Array(K),bP=new Float64Array(K);
    let samples=0,ties=0;
    for(const entry of this.observations){
      if(entry.outcome==="T"){ties++;continue;}
      samples++;
      const sign=entry.outcome==="B"?1:-1;
      for(let i=0;i<K;i++){
        const xi=entry.x[i];bB[i]+=sign*xi;bP[i]-=sign*xi;
        for(let j=0;j<K;j++)A[i][j]+=xi*entry.x[j];
      }
    }
    return{A,bB,bP,samples,ties};
  }
  predict(vector,staticPrediction){
    const x=project(vector),m=this.matrices();
    const variance=Math.max(0,dot(x,solveSPD(m.A,x)));
    const uncertainty=Math.sqrt(variance);
    const bMean=dot(x,solveSPD(m.A,m.bB));
    const pMean=dot(x,solveSPD(m.A,m.bP));
    const localB=bMean+this.alpha*uncertainty,localP=pMean+this.alpha*uncertainty;
    // "靜態" 是現有 256D 模型當局的基準分數，並非重新假造原 Frozen 係數。
    const staticB=Number(staticPrediction?.ucb_banker),
      staticP=Number(staticPrediction?.ucb_player);
    if(!Number.isFinite(staticB)||!Number.isFinite(staticP))
      throw Error("invalid_static_quant_scores");
    const staticUncB=Number(staticPrediction?.arms?.B?.uncertainty)||0,
      staticUncP=Number(staticPrediction?.arms?.P?.uncertainty)||0;
    const blend=this.maxBlend*Math.min(1,m.samples/Math.max(4,this.windowSize*.67));
    const scoreB=(1-blend)*staticB+blend*localB;
    const scoreP=(1-blend)*staticP+blend*localP;
    // 兩部分的未知協方差無法假設獨立：採保守的線性加權不確定度。
    const uncertaintyB=(1-blend)*Math.max(0,staticUncB)+blend*uncertainty;
    const uncertaintyP=(1-blend)*Math.max(0,staticUncP)+blend*uncertainty;
    const combinedUncertainty=Math.sqrt(uncertaintyB**2+uncertaintyP**2);
    const gap=scoreB-scoreP;
    const uncertaintyRatio=Math.abs(gap)/(Math.abs(gap)+combinedUncertainty+1e-9);
    return{version:VERSION,window_size:this.windowSize,sample_count:m.samples,
      tied_rounds:m.ties,recorded_rounds:this.observations.length,
      total_results:this.totalResults,active_dimensions:K,blend_weight:blend,
      A:m.A,b:{B:Array.from(m.bB),P:Array.from(m.bP)},
      arms:{B:{mean:bMean,ucb:localB,uncertainty},
        P:{mean:pMean,ucb:localP,uncertainty}},
      static_ucb:{B:staticB,P:staticP},
      blended_ucb:{B:scoreB,P:scoreP},
      direction:gap>=0?"B":"P",gap,
      uncertainty_banker:uncertaintyB,uncertainty_player:uncertaintyP,
      combined_uncertainty:combinedUncertainty,
      uncertainty_ratio:clamp(uncertaintyRatio,0,1),
      calibrated_probability:false};
  }
}
/**
 * 風險曝險的純函數試算：不將未校準的 UCB 信心當成真實勝率。
 * 無合格研究方向 => 金額 0；有方向 => 僅供示範的公式金額，
 * 不代表有正 EV 或建議實際下注，10–50% 本金風險非常高。
 */
function calculateStakeExposure({bankroll=5000,percent=20,
  confidenceWeight=.5,signal=false}={}){
  const allowed=[5000,10000,20000,30000,50000];
  if(!allowed.includes(bankroll))throw Error("invalid_bankroll_preset");
  if(!Number.isFinite(percent)||percent<10||percent>50)
    throw Error("invalid_stake_percent");
  if(!Number.isFinite(confidenceWeight)||confidenceWeight<.5||confidenceWeight>1)
    throw Error("invalid_confidence_weight");
  const ratio=signal?clamp(percent/100*confidenceWeight,.10,.50):0;
  return{bankroll,percent,confidenceWeight,signal,exposureRatio:ratio,
    amount:signal?Math.min(bankroll*.50,Math.round(bankroll*ratio)):0,
    is_real_wager_advice:false};
}
return{WindowLocalLinUCB,project,calculateStakeExposure,ACTIVE_INDICES:ACTIVE.slice(),VERSION};
});