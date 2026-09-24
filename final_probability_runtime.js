
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
      const pTie=clip(+physics[PHYSICS_INDEX["winner_p_t"]]),pPlayer=clip(1-finalPB-pTie);
      const evBanker=finalPB*.95-pPlayer,evPlayer=pPlayer-finalPB;
      const direction=evBanker>0&&evBanker>evPlayer?"B":evPlayer>0&&evPlayer>evBanker?"P":"Skip";
      evDecision={pTie,pPlayer,evBanker,evPlayer,direction,finalDirection:direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip",confidence:direction==="Skip"?0:Math.max(evBanker,evPlayer,0)};
      mode="final56";
    }
  }catch(e){error=String(e?.message||e||"runtime_error");rawPB=corePB;finalPB=corePB;mode="core";}
  const direction=evDecision?.direction||(finalPB>.5?"B":"P"),finalPP=1-finalPB;
  const confidence=evDecision?.confidence??(direction==="B"?finalPB:finalPP);
  return {...corePrediction,direction,confidence,probabilities:{B:finalPB,P:finalPP},
    regime:mode==="final56"?(direction==="Skip"?"EV 觀望":direction!==corePrediction.direction?"Final XGB換邊":"Final XGB裁決"):corePrediction.regime,
    finalProbability:{version:VERSION,active:mode==="final56",mode,corePB,rawPB,finalPB,bounds,
      pTie:evDecision?.pTie??null,pPlayer:evDecision?.pPlayer??null,evBanker:evDecision?.evBanker??null,evPlayer:evDecision?.evPlayer??null,
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
  const el=id=>document.getElementById(id),orb=el("directionOrb");if(!orb)return;const isB=p.direction==="B";
  el("directionText").textContent=isB?"莊":"閒";el("directionCode").textContent=isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";
  el("regime").textContent=p.regime;el("strength").textContent=p.strength>=.68?"穩定":p.strength>=.52?"中等":"保守";orb.className="direction-orb "+(isB?"banker":"player");
  if(el("modePill"))el("modePill").textContent=p.finalProbability?.mode==="final56"?"Final 57D 完成":"Core 完成";
  if(el("roundCount"))el("roundCount").textContent=n;if(el("message"))el("message").textContent="第 "+(n+1)+" 局分析完成";
}
function installUI(){
  if(typeof document==="undefined")return;const old=document.getElementById("btnStart");if(!old)return;const btn=old.cloneNode(true);old.replaceWith(btn);
  btn.addEventListener("click",()=>{const history=readHistory();if(!history.length){const m=document.getElementById("message");if(m)m.textContent="請先輸入牌局紀錄";return;}
    const core=CORE.hazardChoose(history),p=applyFinalPrediction(history,core);saveSelection(p.direction);registerPrediction(history,p);renderPrediction(p,history.length);});
  const b=document.getElementById("btnB"),p=document.getElementById("btnP"),t=document.getElementById("btnT");