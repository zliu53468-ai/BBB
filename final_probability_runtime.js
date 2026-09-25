(() => {
"use strict";

const CORE=(typeof window!=="undefined")?window.__BGS256_CONTINUATION_TEST__:null;
if(!CORE||typeof CORE.hazardChoose!=="function")return;

const VERSION="PHYSICS_57D_FINAL_PROBABILITY_V4";
const PHYSICS_URL="physics_multitask_model.json";
const FINAL56_URL="final_probability_model.json";
const ORIGINAL7_NAMES=["core_p_b","round_index","estimated_total_hands","remaining_ratio","sx_markov_p_same","stage","depth"];
const PHYSICS_NAMES=[
"cards_p4","cards_p5","cards_p6",
...Array.from({length:10},(_,i)=>"player_point_p"+i),
...Array.from({length:10},(_,i)=>"banker_point_p"+i),
"winner_p_b","winner_p_p","winner_p_t",
..."A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(x=>"next_rank_expected_"+x),
"next_suit_ratio_spades","next_suit_ratio_hearts","next_suit_ratio_diamonds","next_suit_ratio_clubs",
"shoe_consumed_cards","remaining_low_rank_density","remaining_high_rank_density",
"expected_point_diff_norm","expected_abs_point_diff_norm"];
const PHYSICS_INDEX=Object.fromEntries(PHYSICS_NAMES.map((name,index)=>[name,index]));
const EXTENDED_NAMES=["core_p_b_external","shoe_progress_weight",...ORIGINAL7_NAMES.slice(1).map(x=>"original7_"+x),...PHYSICS_NAMES,"physics_noise_score"];
const HISTORY_WINDOW=64,HISTORY_INPUT_DIM=213,PHYSICS_DIM=48,FEATURE_DIM=57;
const DEFAULT_BOUNDS=[.40,.60],EARLY_BOUNDS=[.45,.55],LATE_CLEAN_BOUNDS=[.35,.65],PHYSICS_NOISE_LOW_THRESHOLD=2e-4;
const SNAPSHOT_SCHEMA_VERSION=6;
const TRAINING_KEY="bgs_xgb_final_training_v5",PENDING_KEY="bgs_xgb_final_pending_v5";
const SHOE_KEY="bgs_xgb_final_shoe_id_v5",CUT_KEY="bgs_xgb_estimated_total_hands_v1";
const STORAGE_KEY="bgs256d_short_x_dynamic_v23",MAX_TRAINING_ROWS=15000;
const clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));
const bp=seq=>seq.filter(x=>x==="B"||x==="P");
const directionalRoundCount=seq=>bp(seq).length;
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));

let physicsBundle=null,final56Bundle=null;
let status={physics:false,final56:false,errors:[]};

function transitionSequence(seq){const a=bp(seq),out=[];for(let i=1;i<a.length;i++)out.push(a[i]===a[i-1]?"S":"X");return out;}
function sxMarkovPSame(seq,window=24,prior=1){
  const t=transitionSequence(seq);if(!t.length)return .5;
  const current=t.at(-1),start=Math.max(0,t.length-1-Math.max(2,window));let same=0,sw=0;
  for(let i=start;i<t.length-1;i++){if(t[i]!==current)continue;if(t[i+1]==="S")same++;else if(t[i+1]==="X")sw++;}
  return clip((same+prior)/(same+sw+2*prior));
}
function currentStage(seq){const a=bp(seq);if(!a.length)return 0;const side=a.at(-1);let n=1;for(let i=a.length-2;i>=0&&a[i]===side;i--)n++;return n;}
function currentDepth(seq){const t=transitionSequence(seq);if(!t.length)return 0;const token=t.at(-1);let n=1;for(let i=t.length-2;i>=0&&t[i]===token;i--)n++;return n;}
function getEstimatedTotalHands(){
  try{const c=+(window.__BGS_FINAL_CONFIG__?.estimatedTotalHands||0);if(c>=40&&c<=90)return c;const s=+localStorage.getItem(CUT_KEY);if(s>=40&&s<=90)return s;}catch(_){}
  return 60;
}
function setEstimatedTotalHands(v){const n=Math.round(+v||0);if(n<40||n>90)throw new Error("estimatedTotalHands must be 40..90");try{localStorage.setItem(CUT_KEY,String(n));}catch(_){}return n;}
function buildOriginal7(seq,corePrediction){
  const corePB=clip(+corePrediction?.probabilities?.B||.5),roundIndex=Math.max(1,Math.min(70,seq.length+1));
  const estimatedTotalHands=getEstimatedTotalHands(),remainingRatio=clip((estimatedTotalHands-(roundIndex-1))/Math.max(1,estimatedTotalHands));
  const signal=corePrediction?.singleHazard||null;
  return {core_p_b:corePB,round_index:roundIndex,estimated_total_hands:estimatedTotalHands,remaining_ratio:remainingRatio,
    sx_markov_p_same:sxMarkovPSame(seq),stage:Number.isFinite(+signal?.state?.length)?+signal.state.length:currentStage(seq),
    depth:Number.isFinite(+signal?.depth?.depth)?+signal.depth.depth:currentDepth(seq)};
}
function original7Vector(f){return ORIGINAL7_NAMES.map(n=>Number.isFinite(+f[n])?+f[n]:0);}

function entropy3(p){let s=0;for(const v of p)if(v>0)s-=v*Math.log(v);return s/Math.log(3);}
function ratios(seq,n){const b=n?seq.slice(-n):seq;if(!b.length)return[0,0,0];return[b.filter(x=>x==="B").length/b.length,b.filter(x=>x==="P").length/b.length,b.filter(x=>x==="T").length/b.length];}
function historyVector(seq){
  const one=Array(HISTORY_WINDOW*3).fill(0),tail=seq.slice(-HISTORY_WINDOW),offset=HISTORY_WINDOW-tail.length,map={B:0,P:1,T:2};
  tail.forEach((token,j)=>{one[(offset+j)*3+map[token]]=1;});
  const bps=bp(seq);let turns=0;for(let i=1;i<bps.length;i++)if(bps[i]!==bps[i-1])turns++;
  const r8=ratios(seq,8),r16=ratios(seq,16),r32=ratios(seq,32),rall=ratios(seq,Math.max(1,seq.length));
  const summary=[Math.min(seq.length,90)/90,Math.min(bps.length,90)/90,Math.min(currentStage(seq),12)/12,turns/Math.max(1,bps.length-1),
    ...r8,...r16,...r32,...rall,entropy3(r8),entropy3(r16),entropy3(r32),bps.length&&bps.at(-1)==="B"?1:0,bps.length&&bps.at(-1)==="P"?1:0];
  const out=[...one,...summary];if(out.length!==HISTORY_INPUT_DIM)throw new Error("history vector "+out.length);return out;
}
function dense(input,w,b,relu){
  const out=Array(b.length).fill(0);
  for(let j=0;j<b.length;j++){let s=+b[j]||0;for(let i=0;i<input.length;i++)s+=(+input[i]||0)*(+w[i][j]||0);out[j]=relu?Math.max(0,s):s;}
  return out;
}
function normalise(block,fallback){const a=block.map(v=>Math.max(0,Number.isFinite(+v)?+v:0)),s=a.reduce((x,y)=>x+y,0);return s>1e-12?a.map(v=>v/s):fallback.slice();}
function sanitizePhysics(raw){
  if(raw.length!==PHYSICS_DIM)throw new Error("physics dim mismatch");
  const out=Array(PHYSICS_DIM).fill(0);let src=0,dst=0;
  const copyNorm=(n,fb)=>{const a=normalise(raw.slice(src,src+n),fb);src+=n;for(const v of a)out[dst++]=v;};
  copyNorm(3,[.58,.34,.08]);copyNorm(10,Array(10).fill(.1));copyNorm(10,Array(10).fill(.1));copyNorm(3,[.4586,.4462,.0952]);
  for(let i=0;i<13;i++)out[dst++]=clip(raw[src++],0,6);
  copyNorm(4,Array(4).fill(.25));
  out[dst++]=clip(raw[src++],0,416);out[dst++]=clip(raw[src++],0,1);out[dst++]=clip(raw[src++],0,1);
  out[dst++]=clip(raw[src++],-1,1);out[dst++]=clip(raw[src++],0,1);
  return out;
}
function predictPhysics(seq){
  if(!physicsBundle?.trained)return null;
  let x=historyVector(seq),mean=physicsBundle.scaler?.mean||[],scale=physicsBundle.scaler?.scale||[];
  x=x.map((v,i)=>(v-(+mean[i]||0))/Math.max(1e-12,+scale[i]||1));
  const coefs=physicsBundle.coefs||[],intercepts=physicsBundle.intercepts||[];
  if(!coefs.length||coefs.length!==intercepts.length)throw new Error("invalid physics bundle");
  let h=x;for(let layer=0;layer<coefs.length;layer++)h=dense(h,coefs[layer],intercepts[layer],layer<coefs.length-1);
  const tMean=physicsBundle.target_scaler?.mean||[],tScale=physicsBundle.target_scaler?.scale||[];
  h=h.map((v,i)=>(+v||0)*Math.max(1e-12,+tScale[i]||1)+(+tMean[i]||0));
  return sanitizePhysics(h);
}

function findChild(node,id){return(node?.children||[]).find(c=>+c.nodeid===+id)||null;}
function splitIndex(split,names){const t=String(split??"");if(/^f\d+$/.test(t))return+t.slice(1);return names.indexOf(t);}
function evaluateTree(tree,vector,names){
  let node=tree,guard=0;while(node&&guard++<256){if(Object.prototype.hasOwnProperty.call(node,"leaf"))return+node.leaf||0;
    const idx=splitIndex(node.split,names),value=idx>=0?Math.fround(vector[idx]):NaN,threshold=Math.fround(+node.split_condition);
    node=findChild(node,!Number.isFinite(value)?node.missing:(value<threshold?node.yes:node.no));}
  return 0;
}
function predictFinalProbability(bundle,vector,names){
  if(!bundle?.trained||!Array.isArray(bundle.trees))return .5;
  let margin=+bundle.base_margin||0;
  for(const tree of bundle.trees)margin+=evaluateTree(tree,vector,names);
  return clip(sigmoid(margin));
}
function physicsNoiseScore(physics){const r=physicsIntegrity(physics);return r?Math.abs(r.cardCountProbabilitySum-1)+Math.abs(r.suitRatioSum-1):1;}
function buildExtended(corePB,o7,physics){
  const original=original7Vector(o7),progress=(Number(original[1])/70)**2,noise=physicsNoiseScore(physics);
  const out=[corePB,progress,...original.slice(1),...physics,noise];if(out.length!==FEATURE_DIM)throw new Error("extended dim "+out.length);return out;
}
function unpackPhysicsForecast(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)throw new Error("physics forecast dim mismatch");
  const value=name=>{const v=+physics[PHYSICS_INDEX[name]];return Number.isFinite(v)?v:0;};
  const cardCountProbabilities={"4_cards":value("cards_p4"),"5_cards":value("cards_p5"),"6_cards":value("cards_p6")};
  const expectedNextCardCount=4*cardCountProbabilities["4_cards"]+5*cardCountProbabilities["5_cards"]+6*cardCountProbabilities["6_cards"];
  const rankExpectedConsumption=Object.fromEntries("A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(rank=>[rank,value("next_rank_expected_"+rank)]));
  const suitConsumptionRatios=Object.fromEntries(["spades","hearts","diamonds","clubs"].map(suit=>[suit,value("next_suit_ratio_"+suit)]));
  const suitExpectedConsumption=Object.fromEntries(Object.entries(suitConsumptionRatios).map(([suit,ratio])=>[suit,expectedNextCardCount*ratio]));
  return {nextCardCountProbabilities:cardCountProbabilities,expectedNextCardCount,rankExpectedConsumption,suitConsumptionRatios,suitExpectedConsumption};
}
function physicsIntegrity(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)return null;
  const forecast=unpackPhysicsForecast(physics),sum=values=>values.reduce((total,value)=>total+(+value||0),0);
  const cardCountProbabilitySum=sum(Object.values(forecast.nextCardCountProbabilities));
  const suitRatioSum=sum(Object.values(forecast.suitConsumptionRatios));
  const rankExpectedConsumptionTotal=sum(Object.values(forecast.rankExpectedConsumption));
  const rankExpectedTotalGap=Math.abs(rankExpectedConsumptionTotal-forecast.expectedNextCardCount);
  const rankExpectedTotalTolerance=Math.max(.50,forecast.expectedNextCardCount*.10);
  const checks={cardCountDistribution:Math.abs(cardCountProbabilitySum-1)<=1e-4,suitRatioDistribution:Math.abs(suitRatioSum-1)<=1e-4,rankConsumptionTotal:rankExpectedTotalGap<=rankExpectedTotalTolerance};
  return {valid:Object.values(checks).every(Boolean),checks,cardCountProbabilitySum,expectedNextCardCount:forecast.expectedNextCardCount,rankExpectedConsumptionTotal,rankExpectedTotalGap,rankExpectedTotalTolerance,suitRatioSum};
}
function dataQuality(seq){const directionalRounds=directionalRoundCount(seq);return {directionalRounds,stage:directionalRounds<12?"cold":directionalRounds<20?"warm":"ready",entryEligible:directionalRounds>=12,preferredEntry:directionalRounds>=20};}
function applyProbabilityBounds(rawPB,roundIndex,noiseScore){
  const pair=roundIndex<=40?EARLY_BOUNDS:(roundIndex>50&&noiseScore<=PHYSICS_NOISE_LOW_THRESHOLD?LATE_CLEAN_BOUNDS:DEFAULT_BOUNDS);
  return {value:clip(rawPB,pair[0],pair[1]),low:pair[0],high:pair[1]};
}

function applyFinalPrediction(seq,corePrediction){
  const original7=buildOriginal7(seq,corePrediction),corePB=original7.core_p_b;
  let physics=null,physicsForecast=null,physicsIntegrityReport=null,extended=null,rawPB=corePB,finalPB=corePB,bounds=null,error="",mode="core",evDecision=null;
  try{
    if(physicsBundle?.trained){
      physics=predictPhysics(seq);
      physicsForecast=unpackPhysicsForecast(physics);
      physicsIntegrityReport=physicsIntegrity(physics);
    }
    if(physics&&final56Bundle?.trained){
      extended=buildExtended(corePB,original7,physics);
      rawPB=predictFinalProbability(final56Bundle,extended,EXTENDED_NAMES);
      bounds=applyProbabilityBounds(rawPB,original7.round_index,extended.at(-1));
      finalPB=bounds.value;
      const pTie=clip(+physics[PHYSICS_INDEX["winner_p_t"]]),pPlayer=1-finalPB;
      const evBanker=finalPB*.95-pPlayer,evPlayer=pPlayer-finalPB;
      const direction=evBanker>0&&evBanker>evPlayer?"B":evPlayer>0&&evPlayer>evBanker?"P":"Skip";
      evDecision={pTie,pPlayer,evBanker,evPlayer,direction,finalDirection:direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip",confidence:direction==="Skip"?0:Math.max(evBanker,evPlayer,0)};
      mode="final56";
    }
  }catch(e){error=String(e?.message||e||"runtime_error");rawPB=corePB;finalPB=corePB;mode="core";}
  const direction=evDecision?.direction||(finalPB>.5?"B":"P"),finalPP=1-finalPB;
  const confidence=evDecision?.confidence??(direction==="B"?finalPB:finalPP);
  return {...corePrediction,direction,final_direction:evDecision?.finalDirection||(direction==="B"?"莊 B":"閒 P"),confidence,ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,probabilities:{B:finalPB,P:finalPP},
    regime:mode==="final56"?(direction==="Skip"?"EV 觀望":direction!==corePrediction.direction?"Final XGB換邊":"Final XGB裁決"):corePrediction.regime,
    finalProbability:{version:VERSION,active:mode==="final56",mode,corePB,rawPB,finalPB,bounds,
      p_tie:evDecision?.pTie??null,p_player:evDecision?.pPlayer??null,ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,
      coreDirection:corePrediction.direction,finalDirection:evDecision?.finalDirection||(direction==="B"?"莊 B":"閒 P"),flipped:direction!==corePrediction.direction,
      original7,physics,physicsForecast,physicsIntegrity:physicsIntegrityReport,dataQuality:dataQuality(seq),extended,error}};
}

function readHistory(){
  if(typeof localStorage==="undefined")return[];
  for(const key of["bgs256d_frozen_6x15_forward_v18","bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16"]){
    try{const r=JSON.parse(localStorage.getItem(key)||"null");if(r&&Array.isArray(r.history))return r.history.filter(x=>["B","P","T"].includes(x)).slice(-500);}catch(_){}
  }return[];
}
function getShoeId(){try{let id=localStorage.getItem(SHOE_KEY)||"";if(!id){id="shoe_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,8);localStorage.setItem(SHOE_KEY,id);}return id;}catch(_){return"browser_shoe";}}
function rotateShoeId(){try{localStorage.removeItem(SHOE_KEY);localStorage.removeItem(PENDING_KEY);}catch(_){}}
function readRows(){try{const r=JSON.parse(localStorage.getItem(TRAINING_KEY)||"[]");return Array.isArray(r)?r:[];}catch(_){return[];}}
function writeRows(rows){try{localStorage.setItem(TRAINING_KEY,JSON.stringify(rows.slice(-MAX_TRAINING_ROWS)));}catch(_){}}
function cloneFiniteVector(values,dimension){return Array.isArray(values)&&values.length===dimension&&values.every(value=>Number.isFinite(+value))?values.map(value=>+value):null;}
function registerPrediction(seq,prediction){
  const r=prediction.finalProbability||{},f=r.original7||buildOriginal7(seq,prediction);
  const createdAt=Date.now(),shoeId=getShoeId(),quality=r.dataQuality||dataQuality(seq);
  const pending={schema_version:SNAPSHOT_SCHEMA_VERSION,prediction_id:`${shoeId}:${createdAt}:${seq.join("")}`,shoe_id:shoeId,created_at:createdAt,
    history_fingerprint:seq.join(""),history_event_count:seq.length,directional_round_count:quality.directionalRounds,data_stage:quality.stage,
    entry_eligible:quality.entryEligible,preferred_entry:quality.preferredEntry,mode:r.mode||"core",model_versions:{final_probability:final56Bundle?.schema_version??null,physics:physicsBundle?.schema_version??null},
    core_p_b:f.core_p_b,round_index:f.round_index,estimated_total_hands:f.estimated_total_hands,remaining_ratio:f.remaining_ratio,
    sx_markov_p_same:f.sx_markov_p_same,stage:f.stage,depth:f.depth,original_7d:original7Vector(f),physics_48d:cloneFiniteVector(r.physics,PHYSICS_DIM),
    features_57d:cloneFiniteVector(r.extended,FEATURE_DIM),shoe_progress_weight:Number.isFinite(+r.extended?.[1])?+r.extended[1]:null,physics_noise_score:Number.isFinite(+r.extended?.at(-1))?+r.extended.at(-1):null,raw_p_b:Number.isFinite(+r.rawPB)?+r.rawPB:null,final_p_b:Number.isFinite(+r.finalPB)?+r.finalPB:null,
    probability_bounds:r.bounds?[r.bounds.low,r.bounds.high]:null,predicted_direction:r.finalDirection||prediction.direction||"",physics_integrity:r.physicsIntegrity||null};
  try{localStorage.setItem(PENDING_KEY,JSON.stringify(pending));}catch(_){}
}
function settlePending(actualOutcome){
  const actual=String(actualOutcome||"").toUpperCase();
  if(actual!=="B"&&actual!=="P"&&actual!=="T")return;
  let p=null;try{p=JSON.parse(localStorage.getItem(PENDING_KEY)||"null");}catch(_){}if(!p)return;
  const row={...p,settled_at:Date.now(),actual_outcome:actual,actual_b:actual==="B"?1:actual==="P"?0:null,is_directional_label:actual!=="T"},rows=readRows();
  if(!rows.length||rows.at(-1)?.shoe_id!==row.shoe_id||rows.at(-1)?.history_fingerprint!==row.history_fingerprint)rows.push(row);
  writeRows(rows);try{localStorage.removeItem(PENDING_KEY);}catch(_){}
}
function exportTrainingData(){return JSON.stringify({schema_version:SNAPSHOT_SCHEMA_VERSION,target:"actual_b_binary",feature_names:EXTENDED_NAMES,original_7d_feature_names:ORIGINAL7_NAMES,physics_48d_feature_names:PHYSICS_NAMES,rows:readRows()},null,2);}
function downloadTrainingData(){const blob=new Blob([exportTrainingData()],{type:"application/json;charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="bgs_final57_training_"+Date.now()+".json";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);}

async function fetchBundle(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw new Error(url+":HTTP"+r.status);return await r.json();}
async function loadModels(){
  status={physics:false,final56:false,errors:[]};
  const rs=await Promise.allSettled([fetchBundle(PHYSICS_URL),fetchBundle(FINAL56_URL)]);
  if(rs[0].status==="fulfilled"&&rs[0].value?.model_type==="baccarat_physics_multitask_mlp"){physicsBundle=rs[0].value;status.physics=!!physicsBundle.trained;}else status.errors.push("physics_model");
  if(rs[1].status==="fulfilled"&&rs[1].value?.model_type==="xgb_final_probability_classifier"&&rs[1].value?.feature_names?.length===FEATURE_DIM){final56Bundle=rs[1].value;status.final56=!!final56Bundle.trained;}else{final56Bundle=null;status.errors.push("final57_model");}
  return status;
}
function saveSelection(direction){try{const old=JSON.parse(localStorage.getItem(STORAGE_KEY)||"null")||{},streak=old.last_selected===direction?Math.max(1,(+old.selection_streak||0)+1):1;localStorage.setItem(STORAGE_KEY,JSON.stringify({last_selected:direction,selection_streak:streak}));}catch(_){}}
function renderPrediction(p,n){
  const el=id=>document.getElementById(id),orb=el("directionOrb");if(!orb)return;const isSkip=p.direction==="Skip",isB=p.direction==="B";
  el("directionText").textContent=isSkip?"觀望":isB?"莊":"閒";el("directionCode").textContent=isSkip?"SKIP":isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";
  el("regime").textContent=p.regime;el("strength").textContent=isSkip?"觀望":p.strength>=.68?"穩定":p.strength>=.52?"中等":"保守";orb.className="direction-orb "+(isSkip?"":isB?"banker":"player");
  if(el("modePill"))el("modePill").textContent=p.finalProbability?.mode==="final56"?"Final 57D 完成":"Core 完成";
  if(el("roundCount"))el("roundCount").textContent=n;if(el("message"))el("message").textContent="第 "+(n+1)+" 局分析完成";
}
function installUI(){
  if(typeof document==="undefined")return;const old=document.getElementById("btnStart");if(!old)return;const btn=old.cloneNode(true);old.replaceWith(btn);
  btn.addEventListener("click",()=>{const history=readHistory();if(!history.length){const m=document.getElementById("message");if(m)m.textContent="請先輸入牌局紀錄";return;}
    const core=CORE.hazardChoose(history),p=applyFinalPrediction(history,core);saveSelection(p.direction);registerPrediction(history,p);renderPrediction(p,history.length);});
  const b=document.getElementById("btnB"),p=document.getElementById("btnP"),t=document.getElementById("btnT");
  if(b)b.addEventListener("click",()=>settlePending("B"));if(p)p.addEventListener("click",()=>settlePending("P"));if(t)t.addEventListener("click",()=>settlePending("T"));
  const end=document.getElementById("btnEnd");if(end)end.addEventListener("click",rotateShoeId);
}
if(typeof window!=="undefined")window.__BGS_FINAL56__={version:VERSION,applyFinalPrediction,predictPhysics,unpackPhysicsForecast,physicsIntegrity,dataQuality,buildOriginal7,historyVector,loadModels,setEstimatedTotalHands,getEstimatedTotalHands,
  registerPrediction,settlePending,exportTrainingData,downloadTrainingData,getTrainingRows:()=>readRows(),getTrainingCount:()=>readRows().length,getModelStatus:()=>({...status,mode:status.physics&&status.final56?"final56":"core"})};
loadModels();installUI();
})();
