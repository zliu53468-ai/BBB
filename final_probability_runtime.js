(() => {
"use strict";

const CORE=(typeof window!=="undefined")?window.__BGS256_CONTINUATION_TEST__:null;
const PARTICLE500=(typeof window!=="undefined")?window.__BGS_PARTICLE500__:null;
const LOCAL_CALIBRATION_API=(typeof window!=="undefined")?window.__BBB_BROWSER_CALIBRATION__:null;
const LOCAL_CALIBRATION=LOCAL_CALIBRATION_API?.createClient(typeof localStorage!=="undefined"?localStorage:null)||null;
let browserCalibrationTag="unready";
const localCalibrationConfig=()=>cardcountProfile?.local_calibration||{};
const MACRO=(typeof window!=="undefined")?window.__BGS_MACRO_EMA__:null;
if(!CORE||typeof CORE.hazardChoose!=="function")return;

const EV_CONSISTENCY_V1=false; // Shadow gate failed; set true only after holdout acceptance.
const EARLY_TIE_PENALTY=.15;
const VERSION=EV_CONSISTENCY_V1
  ?"PHYSICS_57D_V14_8_FORECAST_FIRST_EVC1"
  :"PHYSICS_57D_V14_8_FIRST_LAYER_EV1";
const PHYSICS_URL="physics_multitask_model.json";
let cardcountProfile=null,displayGapOverride=null;
const FIRST_SMOOTHING_KEY="bgs_first_layer_gap_smoothing_v1";
let firstSmoothingModelTag="unready",firstSmoothingStorageError="";

// Post-draw settlement: B/P/T are unconditional probabilities for the completed hand.
// The display/Core/XGB probabilities remain conditional B/P; reattach tie mass once.
function settlementPolicy(){
  const config=cardcountProfile?.settlement||{schema_version:1,mode:"post_draw_unconditional_v1",banker_commission:.05,player_commission:0,tie_settlement:"push"};
  if(config.mode==="legacy_conditional")return null;
  if(config.mode!=="post_draw_unconditional_v1"||config.schema_version!==1||config.tie_settlement!=="push")throw new Error("unsupported settlement policy");
  for(const key of ["banker_commission","player_commission"])
    if(typeof config[key]!=="number"||!Number.isFinite(config[key])||config[key]<0||config[key]>1)throw new Error("invalid "+key);
  return {mode:config.mode,banker_commission:config.banker_commission,player_commission:config.player_commission,tie_settlement:"push",stake:1};
}
function settledOutcomeEv(probabilities,policy=settlementPolicy()||{banker_commission:.05,player_commission:0,tie_settlement:"push"}){
  const values=["B","P","T"].map(k=>probabilities?.[k]);
  if(values.some(v=>typeof v!=="number"||!Number.isFinite(v)||v<0||v>1))throw new Error("invalid post-draw probabilities");
  const sum=values.reduce((a,b)=>a+b,0);
  if(Math.abs(sum-1)>1e-6)throw new Error("post-draw probabilities must sum to one");
  if(policy.tie_settlement!=="push")throw new Error("only tie push is supported");
  for(const key of ["banker_commission","player_commission"])
    if(typeof policy[key]!=="number"||!Number.isFinite(policy[key])||policy[key]<0||policy[key]>1)throw new Error("invalid "+key);
  const [B,P,T]=values.map(v=>v/sum);
  return {probabilities:{B,P,T},banker:(1-policy.banker_commission)*B-P,player:(1-policy.player_commission)*P-B,
    probability_basis:"post_draw_unconditional",stake:1,tie_net_return:0,banker_commission:policy.banker_commission,player_commission:policy.player_commission};
}
function conditionalSettlement(probabilityB,pTie){
  if(typeof probabilityB!=="number"||!Number.isFinite(probabilityB)||probabilityB<0||probabilityB>1||typeof pTie!=="number"||!Number.isFinite(pTie)||pTie<0||pTie>1)throw new Error("invalid conditional settlement");
  return settledOutcomeEv({B:(1-pTie)*probabilityB,P:(1-pTie)*(1-probabilityB),T:pTie});
}
function drawCountDistribution(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)return null;
  const values=physics.slice(0,3),sum=values.reduce((a,b)=>a+b,0);
  if(values.some(v=>!Number.isFinite(v)||v<0)||!Number.isFinite(sum)||sum<=0)return {available:false,reason:"invalid_draw_distribution"};
  const p=values.map(v=>v/sum);
  return {available:true,probabilities:{"4_cards":p[0],"5_cards":p[1],"6_cards":p[2]},expected_cards:4*p[0]+5*p[1]+6*p[2],input_sum:sum,normalized:Math.abs(sum-1)<=1e-6};
}
function postDrawAudit(direct,posterior,mixed,primary,finalPB){
  if(!settlementPolicy()||!primary)return null;
  const evaluate=physics=>Array.isArray(physics)?settledOutcomeEv({B:physics[23],P:physics[24],T:physics[25]}):null;
  return {schema_version:1,policy:settlementPolicy(),probability_basis:"post_draw_unconditional",pf_stage_applications:posterior?1:0,
    direct_physics:evaluate(direct),pf_posterior:evaluate(posterior),mixed:evaluate(mixed),
    first_layer:primary.firstLayerEv||null,candidate:conditionalSettlement(primary.directionalPB,primary.pT),final:conditionalSettlement(finalPB,primary.pT),
    model_ev_head_basis:"nominal_5pct_commission_training_features_unchanged"};
}


function directionSelectionPolicy(){
  const config=cardcountProfile?.direction_selection||{schema_version:1,mode:"post_draw_ev_argmax_v1",ev_tie_tolerance:1e-12};
  if(config.schema_version!==1||!["post_draw_ev_argmax_v1","physics_candidate_v1"].includes(config.mode))throw new Error("invalid direction selection policy");
  const epsilon=config.ev_tie_tol