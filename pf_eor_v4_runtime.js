/**
 * BBB V4 正式推論核心：僅輸入當局閒／莊 0–9 最終總點。
 * 方向唯一來自條件式 PF 組成機率與 EV；EOR 量測敏感度；
 * Deplete MC 使用相同資料獨立重新抽樣以交叉檢查，不當成額外證據。
 * 不建立 LinUCB、不計算 256D 上下文、不使用牌路特徵修改方向。
 */
(function(root,factory){
 "use strict";
 const quant=root?.BBBQuant256Online||
  (typeof module==="object"&&module.exports?require("./quant256_online_runtime.js"):null);
 const physics=root?.BBBPhysicsForecastFirst||
  (typeof module==="object"&&module.exports?require("./physics_forecast_first_runtime.js"):null);
 const deplete=root?.BBBDepleteMC||
  (typeof module==="object"&&module.exports?require("./deplete_mc_runtime.js"):null);
 const api=factory(quant,physics,deplete);
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBPhysicalV4=api;
})(typeof window!=="undefined"?window:null,function(quant,physics,deplete){
 "use strict";
 if(!quant?.BlindPhysicsEstimator||!physics?.summarizeHypotheses||
   !physics?.earlyShoeGuard||!deplete?.attachShadow||!deplete?.dealWithUniforms)
   throw Error("missing_physical_v4_dependencies");
 const VERSION="BBB_PF_EOR_DEPLETE_V4_SCORE_ONLY";
 const CONFIG=Object.freeze({
  particles:96,simulationsPerParticle:160,
  // 使用 16 個分層抽取的假想粒子，每個點數影響各做 96 次配對模擬。
  eorParticleCount:16,eorSamples:96,
  minEvGap:.0028,maxEvGap:.012,
  mcNoiseWeight:.09,posteriorSpreadWeight:.013,eorSensitivityWeight:.05,
  poorQualityWeight:.0015,
  crossCheckZ:2.5,crossCheckFloor:.008
 });
 function valid(x){return Number.isInteger(x)&&x>=0&&x<=9;}
 function outcome(p,b){return b>p?"B":p>b?"P":"T";}
 const abs=Math.abs,finite=Number.isFinite;
 const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
 function seeded(seed){
  let s=seed>>>0||0x9e3779b9;
  return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};
 }
 const gapOf=prob=>1.95*prob.B-2*prob.P;
 const gapScore=x=>x==="B"?1.95:x==="P"?-2:0;

 function computeEOR(estimator,pair,{particles=CONFIG.eorParticleCount,
  samples=CONFIG.eorSamples}={}){
  const rows=estimator.particles;
  if(!Array.isArray(rows)||rows.length<1||
   !Number.isInteger(particles)||particles<1||particles>rows.length||
   !Number.isInteger(samples)||samples<16||samples>1024)
    throw Error("invalid_eor_samples");
  const delta=Array(10).fill(0),supported=Array(10).fill(0);
  let scenarios=0;
  // 分層選取後驗粒子；每次在同一假想組成中扣掉一張 0–9 點數牌，
  // 與未再扣牌版本使用相同亂數，降低 EOR 差分的蒙地卡羅雜訊。
  for(let i=0;i<particles;i++){
   const idx=Math.min(rows.length-1,Math.floor((i+.5)*rows.length/particles));
   const counts=rows[idx],total=counts.reduce((a,b)=>a+b,0);
   if(total<7)continue;
   const rng=seeded(0x45d3e77^Math.imul(pair.player+1,0x9e3779b1)^
    Math.imul(pair.banker+1,0x85ebca6b)^Math.imul(idx+1,0x27d4eb2f));
   // 同一個假想剩餘組成計算 10 種點數移除的配對差值；
   // 這些是敏感度導數的近似，不是額外觀察到的真實牌。
   for(let j=0;j<samples;j++){
    const u=Array.from({length:6},()=>rng());
    const control=gapScore(deplete.dealWithUniforms(counts,u));
    for(let rank=0;rank<10;rank++){
     if(counts[rank]<1)continue;
     const removed=counts.slice();removed[rank]--;
     delta[rank]+=gapScore(deplete.dealWithUniforms(removed,u))-control;
     supported[rank]++;scenarios++;
    }
   }
  }
  const effects=delta.map((v,i)=>supported[i]?v/supported[i]:null);
  if(effects.some(v=>v===null))throw Error("eor_insufficient_support");
  const maxAbs=Math.max(...effects.map(abs));
  const weightedMean=effects.reduce((v,e,i)=>v+e*(rows[0]?.[i]??0),0)/
    (rows[0]?.reduce((a,b)=>a+b,0)||1);
  return{effects_by_point:effects,max_abs_ev_gap_delta:maxAbs,
   weighted_ev_gap_delta:weightedMean,simulated_paired_scenarios:scenarios,
   hypothetically_removed_cards:1,diagnostic_only:true,
   actual_remaining_counts_known:false,independent_evidence:false};
 }

 class QualityFilterV4{
  constructor(config=CONFIG){this.config=config;}
  decide(report){
   const c=this.config,d=report?.quality,eor=report?.eor,shadow=report?.deplete_mc;
   const evB=report?.ev_banker,evP=report?.ev_player;
   const candidate=finite(evB)&&finite(evP)?(evB>=evP?"B":"P"):null;
   const mandatory=[
    evB,evP,d?.quality,d?.ess_ratio,d?.ancestor_diversity,
    d?.mc_gap_se,d?.posterior_gap_spread,eor?.max_abs_ev_gap_delta,
    shadow?.ev_banker,shadow?.ev_player,shadow?.paired_mc_gap_se
   ];
   if(!candidate||mandatory.some(v=>!finite(v))||
    report?.physical_ev?.simulation_exhausted||
    shadow?.status!=="complete"||
    !report?.point_observation_independent)
    return{direction:"SKIP",candidate,reason:"diagnostics_unavailable",
     validated_hit_rate:false,verified_edge:false};
   const quality=clamp(d.quality,0,1);
   const noise=clamp(d.mc_gap_se,0,.10);
   const posterior=clamp(d.posterior_gap_spread,0,.3);
   const sensitivity=clamp(eor.max_abs_ev_gap_delta,0,.2);
   // 在 EV 單位下比較，而不是把 EV 差距誤稱為勝率差距。
   // 這是可回測、可微調的保守啟發式，並非經驗證的勝率置信區間。
   const threshold=clamp(c.minEvGap+
    c.mcNoiseWeight*noise+
    c.posteriorSpreadWeight*posterior+
    c.eorSensitivityWeight*sensitivity+
    c.poorQualityWeight*(1-quality),c.minEvGap,c.maxEvGap);
   const gap=abs(evB-evP),pfSigned=evB-evP;
   const mcSigned=shadow.ev_banker-shadow.ev_player;
   const deviation=abs(pfSigned-mcSigned);
   // Deplete 僅負責相容性檢查：同一筆證據不可當成兩份獨立觀測融合。
   const tolerance=Math.max(c.crossCheckFloor,c.crossCheckZ*
    Math.hypot(noise,shadow.paired_mc_gap_se));
   const stable=deviation<=tolerance;
   const conflict=pfSigned*mcSigned<0&&abs(mcSigned)>threshold*.5;
   const pass=gap>threshold&&stable&&!conflict;
   return{direction:pass?candidate:"SKIP",candidate,
    reason:gap<=threshold?"weak_ev_gap":!stable?"deplete_disagreement":
     conflict?"opposite_deplete_direction":"quality_pass",
    gap,threshold,quality,mc_gap_se:noise,posterior_spread:posterior,
    eor_sensitivity:sensitivity,deplete_gap_deviation:deviation,
    deplete_agreement_limit:tolerance,deplete_consistent:stable&&!conflict,
    validated_hit_rate:false,verified_edge:false};
  }
 }

 class PhysicalPointV4{
  constructor(config=CONFIG){this.config=config;this.filter=new QualityFilterV4(config);}
  analyze(player,banker){
   if(!valid(player)||!valid(banker))throw Error("invalid_total_points");
   const pair={player,banker},result=outcome(player,banker);
   // 每次重新建立與八副牌先驗一致的粒子後驗；上一局不寫入本次組成。
   const pf=new quant.BlindPhysicsEstimator({particles:this.config.particles,
    exactSingleScore:true,seed:0xBBB256});
   pf.observe(result,pair);
   const raw=physics.summarizeHypotheses(pf,[result],
    {samplesPerParticle:this.config.simulationsPerParticle});
   if(raw.simulation_exhausted||raw.particle_support!==this.config.particles)
    throw Error("pf_posterior_unavailable");
   const early=physics.earlyShoeGuard(raw,pf,1,{independentFinalScore:true});
   const physical=early.physical;
   const core={version:VERSION,point_observation_independent:true,
    history_length:1,point_score_observation_count:1,
    observed_card_evidence:false,physical_ev:physical,
    ev_banker:physical.ev_banker,ev_player:physical.ev_player,
    quality:early.diagnostics};
   const eor=computeEOR(pf,pair,{particles:this.config.eorParticleCount,
    samples:this.config.eorSamples});
   const checked=deplete.attachShadow(core,pair);
   const report={...core,eor,deplete_mc:checked.deplete_mc,
    model_posterior:{sampled_compositions:pf.size,model_based:true,
     actual_card_ranks_known:false},
    simulated_probability:physical.simulated_probability,
    baseline_probability:physics.BASE,
    original_pf_ev:{B:raw.ev_banker,P:raw.ev_player},
    markov_used_for_direction:false,linucb_used_for_direction:false,
    context256_used_for_direction:false,road_used_for_direction:false,
    deplete_blended_into_primary:false};
   return{...report,v4:this.filter.decide(report)};
  }
 }
 return Object.freeze({VERSION,CONFIG,computeEOR,QualityFilterV4,PhysicalPointV4});
});
