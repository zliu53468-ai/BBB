/**
 * BBB 資金曝險試算 UI（完全獨立於預測引擎）。
 * 依現行 research_signal 是否通過，顯示參考金額或 NT$0。
 * 不修改莊/閒方向、EV、連勝狀態、觀望門檻或原始牌靴資料。
 */
(function(root,factory){
  "use strict";
  const api=factory(root);
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.__BBB_V3_CAPITAL_UI__=api;
})(typeof window!=="undefined"?window:null,function(root){
  "use strict";
  const VERSION="BBB_V3_CAPITAL_UI_V1",KEY="bbb_stake_exposure_panel_v1";
  const PRESETS=[5000,10000,20000,30000,50000];
  let capital=5000,stakePercent=20,lastReport=null;
  const el=id=>root?.document?.getElementById(id)||null;
  const money=v=>"NT$"+Math.round(v).toLocaleString("en-US");
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  function calculateStakeExposure({bankroll=5000,percent=20,confidenceWeight=.5,signal=false}={}){
    if(!PRESETS.includes(bankroll))throw Error("invalid_bankroll_preset");
    if(!Number.isInteger(percent)||percent<10||percent>50)throw Error("invalid_stake_percent");
    if(!Number.isFinite(confidenceWeight)||confidenceWeight<.5||confidenceWeight>1)
      throw Error("invalid_confidence_weight");
    const ratio=signal?clamp(percent/100*confidenceWeight,.10,.50):0;
    return{amount:signal?Math.round(bankroll*ratio):0,exposureRatio:ratio,
      confidenceWeight,signal,verified_edge:false,version:VERSION};
  }
  function load(){
    try{
      const raw=JSON.parse(root?.localStorage?.getItem(KEY)||"null");
      if(raw&&PRESETS.includes(raw.capital))capital=raw.capital;
      if(raw&&Number.isInteger(raw.stakePercent)&&raw.stakePercent>=10&&raw.stakePercent<=50)
        stakePercent=raw.stakePercent;
    }catch(_){}
  }
  function save(){
    try{root?.localStorage?.setItem(KEY,JSON.stringify({capital,stakePercent}));}catch(_){}
  }
  function renderStake(report){
    lastReport=report||null;
    const signal=report?.research_signal?.available===true;
    // 正規化研究分數，不是客觀勝率；決不回寫預測引擎。
    const score=Number(report?.research_signal?.score)||0;
    const gate=Number(report?.research_signal?.dynamic_gate)||0;
    const relative=signal?clamp((score-gate)/Math.max(.01,score+.01),0,1):0;
    const weight=signal?clamp(.5+.5*relative,.5,1):.5;
    const exposure=calculateStakeExposure({bankroll:capital,percent:stakePercent,
      confidenceWeight:weight,signal});
    const select=el("bankrollSelect"),slider=el("stakePercent");
    if(select)select.value=String(capital);
    if(slider)slider.value=String(stakePercent);
    const values={
      bankrollCurrent:money(capital),
      stakePercentValue:stakePercent+"%",
      stakeConfidenceWeight:signal?(weight*100).toFixed(1)+"%":"—",
      stakeRecommendation:money(exposure.amount),
      stakeLocalStatus:report?
        "V2 已分析 "+report.history_length+" 局":"V2 準備中",
      stakeStatus:!signal?"觀望／訊號未通過：試算金額 NT$0":
        "V2 模擬方向："+(report.research_signal.direction==="B"?"莊":"閒")+
        "｜曝險試算 "+money(exposure.amount)+"｜非已驗證正期望值"
    };
    for(const [id,value] of Object.entries(values)){
      const node=el(id);if(node)node.textContent=value;
    }
    return exposure;
  }
  function init(){
    load();
    const select=el("bankrollSelect"),slider=el("stakePercent");
    if(select)select.addEventListener("change",()=>{
      const next=Number(select.value);
      if(PRESETS.includes(next))capital=next;
      save();renderStake(lastReport);
    });
    if(slider)for(const event of ["input","change"])
      slider.addEventListener(event,()=>{
        const next=Number(slider.value);
        if(Number.isFinite(next))stakePercent=clamp(Math.round(next),10,50);
        save();renderStake(lastReport);
      });
    renderStake(null);
  }
  if(root?.document)init();
  return{VERSION,calculateStakeExposure,renderStake,
    get capital(){return capital;},get stake_percent(){return stakePercent;},
    get report(){return lastReport;}};
});