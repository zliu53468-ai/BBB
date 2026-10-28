/* 純算牌優先：獨立、可回放的機率校準／選擇性閘門。無 Core/XGB/路單方向投票。 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_PHYSICS_PRIORITY__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const KEY='bgs_physics_priority_v1',VERSION='PHYSICS_PRIORITY_SELECTIVE_V1';
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const DEFAULTS={schema_version:1,version:VERSION,enabled:true,route_weight:0,
 stages:{early:{end:20,probability_band:.060,noise_max:.66,uncertainty_max:.66,ess_min:.60,quality_min:.62,ev_min:.015,alpha:.38},middle:{end:50,probability_band:.018,noise_max:.70,uncertainty_max:.70,ess_min:.60,quality_min:.62,ev_min:.010,alpha:.52},late:{probability_band:.015,noise_max:.72,uncertainty_max:.72,ess_min:.60,quality_min:.62,ev_min:.006,alpha:.62}},
 alpha:{signal_gain:.15,noise_penalty:.10,min:.22,max:.80},calibration:{min_samples:30,prior_strength:120,window:60,decay:.98,max_delta:.004},
 point_conditioned:{enabled:true,weight:.65,minimum_weight:.60,maximum_weight:.70},
 protection:{loss_raise_at:3,loss_multiplier:1.20,cooldown_at:5,cooldown_hands:9,win_relief_at:3,win_relief:.03},
 volume:{enabled:true,window:16,min_history:8,target:{early:.05,middle:.30,late:.35},band_multiplier:1.10},estimated_burn_min:2,estimated_burn_max:11};
const CONSERVATIVE_DEFAULTS={schema_version:1,version:'EARLY_MID_CONSERVATIVE_V2',enabled:false,min_calibration_samples:30,curves:{
 probability_band:[[1,.060],[20,.060],[30,.0275],[31,.0270],[35,.0225],[40,.019],[45,.019],[50,.015],[70,.015]],
 noise_max:[[1,.55],[30,.55],[40,.62],[45,.62],[50,.72],[70,.72]],uncertainty_max:[[1,.55],[30,.55],[40,.62],[45,.62],[50,.72],[70,.72]],
 ess_min:[[1,.75],[30,.75],[40,.68],[45,.68],[50,.60],[70,.60]],quality_min:[[1,.70],[30,.70],[40,.65],[45,.65],[50,.62],[70,.62]],
 ev_min:[[1,.015],[30,.015],[45,.010],[50,.006],[70,.006]],alpha:[[1,.28],[30,.28],[40,.50],[45,.50],[50,.62],[70,.62]],alpha_max:[[1,.40],[30,.40],[40,.65],[50,.80],[70,.80]],
 pf_max:[[1,.08],[30,.08],[35,.10],[45,.14],[50,.16],[70,.16]],residual_max_delta:[[1,.015],[30,.015],[45,.025],[50,.05],[70,.05]],
 calibration_gain:[[1,0],[30,0],[40,1],[70,1]],point_weight_max:[[1,0],[30,0],[40,.65],[70,.65]]}};
function interpolate(n,anchors){if(n<=anchors[0][0])return anchors[0][1];for(let i=1;i<anchors.length;i++)if(n<=anchors[i][0]){const [x,a]=anchors[i-1],[y,b]=anchors[i];return a+(n-x)/(y-x)*(b-a);}return anchors.at(-1)[1];}
function limits(n,custom={}){const v=custom.early_mid_conservative_v2;if(v?.enabled!==true)return null;const cfg={...CONSERVATIVE_DEFAULTS,...v},curves={...CONSERVATIVE_DEFAULTS.curves,...v.curves},out={version:cfg.version,min_calibration_samples:Math.max(30,cfg.min_calibration_samples),stage:n<=30?'early':n<=45?'middle':'late'};for(const [k,a]of Object.entries(curves))out[k]=interpolate(n,a);if(n<=30){out.probability_band=Math.max(.0275,out.probability_band);out.pf_max=Math.min(.08,out.pf_max);out.alpha_max=Math.min(.40,out.alpha_max);out.residual_max_delta=Math.min(.015,out.residual_max_delta);out.calibration_gain=0;out.point_weight_max=0;}return out;}
function config(c={}){const out={...DEFAULTS,...c};for(const k of ['alpha','calibration','point_conditioned','protection','volume'])out[k]={...DEFAULTS[k],...c[k]};out.stages={};for(const k of ['early','middle','late'])out.stages[k]={...DEFAULTS.stages[k],...c.stages?.[k]};return out;}
function stage(n,c){return n<=c.stages.early.end?'early':n<=c.stages.middle.end?'middle':'late';}
function state(records,c){let losses=0,wins=0,cooldown_until=0;for(const r of records){if(!['B','P'].includes(r.direction)||r.actual==='T')continue;if(r.direction===r.actual){wins++;losses=0;}else{losses++;wins=0;if(losses>=c.protection.cooldown_at)cooldown_until=r.round+c.protection.cooldown_hands;}}return {losses,wins,cooldown_until};}
function infer(input,records=[],custom={}){
 const c=config(custom),n=input.history.length+1,tight=limits(n,c),s=tight?.stage||stage(n,c),p={...c.stages[s],...tight},protection=state(records,c),reasons=[];
 const source=input.probabilities||{},mass=source.B+source.P;
 if(!finite(source.B)||source.B<0||!finite(source.P)||source.P<0||!finite(mass)||mass<=0||!finite(source.T)||source.T<0||Math.abs(mass+source.T-1)>1e-5)return {version:VERSION,direction:'Skip',confidence_level:'低',reasons:['physics_unavailable'],stage:s,round:n,eligible:false};
 const raw=source.B/mass,noise=input.noise,uncertainty=input.uncertainty,ess=input.ess_ratio,quality=input.particle_quality;
 let numerator=0,weight=0,count=0;const recent=records.filter(r=>r.actual!=='T'&&finite(r.raw_q)).slice(-c.calibration.window);
 for(let i=0;i<recent.length;i++){const r=recent[i],w=c.calibration.decay**(recent.length-1-i);numerator+=w*((r.actual==='B'?1:0)-r.raw_q);weight+=w;count++;}
 const minimum=tight?Math.max(c.calibration.min_samples,tight.min_calibration_samples):c.calibration.min_samples;
 const delta=count>=minimum?clamp(numerator/(c.calibration.prior_strength+weight),-c.calibration.max_delta,c.calibration.max_delta)*(tight?.calibration_gain??1):0;
 // Calibration cannot cross the current Physics sign; no history-based flip.
 let calibrated=clamp(raw+delta);if((calibrated-.5)*(raw-.5)<0)calibrated=.5;
 if(input.clip_limits?.every(finite))calibrated=clamp(calibrated,input.clip_limits[0],input.clip_limits[1]);
 const strength=clamp(Math.abs(calibrated-.5)/.06),alpha=clamp(p.alpha+c.alpha.signal_gain*strength-c.alpha.noise_penalty*(finite(noise)?noise:1),c.alpha.min,Math.min(c.alpha.max,tight?.alpha_max??c.alpha.max));
 const previous=records.at(-1)?.smoothed_q;
 let q=finite(previous)?alpha*calibrated+(1-alpha)*previous:calibrated;
 const sign_reset=(q-.5)*(calibrated-.5)<0;if(sign_reset)q=calibrated;
 q=input.clip_limits?.every(finite)?clamp(q,input.clip_limits[0],input.clip_limits[1]):clamp(q);const direction=q>.5?'B':q<.5?'P':'Skip';
 let band=p.probability_band;if(protection.losses>=c.protection.loss_raise_at)band*=c.protection.loss_multiplier;
 if(!tight&&protection.wins>=c.protection.win_relief_at)band*=1-clamp(c.protection.win_relief,0,.05);
 const v=c.volume,last=records.slice(-v.window),actionRate=last.length?last.filter(r=>['B','P'].includes(r.direction)).length/last.length:0;
 const volume_tightened=v.enabled&&last.length>=v.min_history&&actionRate>(v.target[s]??.35);if(volume_tightened)band*=Math.max(1,v.band_multiplier);
 const evB=.95*mass*q-mass*(1-q),evP=mass*(1-q)-mass*q;
 const evFloor=Math.max(p.ev_min,finite(input.legacy_ev_floor)?input.legacy_ev_floor:0),bestEV=direction==='B'?evB:direction==='P'?evP:-Infinity;
 if(!finite(noise)||noise>p.noise_max)reasons.push('noise_high');
 if(!finite(uncertainty)||uncertainty>p.uncertainty_max)reasons.push('uncertainty_high');
 if(!finite(ess)||ess<p.ess_min)reasons.push('ess_low_or_missing');
 if(!finite(quality)||quality<p.quality_min)reasons.push('particle_quality_low_or_missing');
 if(Math.abs(q-.5)<=band)reasons.push('probability_band');
 if(bestEV<=evFloor)reasons.push('nonpositive_or_insufficient_ev');
 if(n<=protection.cooldown_until)reasons.push('loss_cooldown');
 if(input.point_error)reasons.push('point_posterior_unavailable');
 const uncertain=reasons.some(r=>['noise_high','uncertainty_high','ess_low_or_missing','particle_quality_low_or_missing'].includes(r));
 const eligible=!reasons.length,level=eligible?(Math.abs(q-.5)>=band*1.5&&noise<.55?'高':'中'):'低';
 return {version:tight?.version||VERSION,direction:eligible?direction:'Skip',candidate_direction:direction,eligible,confidence_level:level,uncertainty_skip:uncertain,reasons,stage:s,round:n,
  raw_q:raw,calibrated_q:calibrated,smoothed_q:q,alpha,sign_reset,calibration:{samples:count,delta,...(tight?{minimum,gain:tight.calibration_gain}:{})},...(tight?{conservative_limits:tight,residual:input.residual||null}:{}),probabilities:{B:mass*q,P:mass*(1-q),T:source.T},directional:{B:q,P:1-q},ev_banker:evB,ev_player:evP,
  probability_band:band,ev_floor:evFloor,noise,uncertainty,ess_ratio:ess,particle_quality:quality,protection,volume:{action_rate:actionRate,tightened:volume_tightened,relief_applied:false},
  route_weight:0,xgb_influence:0,point_weight:input.point_weight||0,point_diagnostics:input.point_diagnostics||null,
  remaining_cards_interval:[Math.max(0,416-6*input.history.length-c.estimated_burn_max),Math.max(0,416-4*input.history.length-c.estimated_burn_min)],remaining_note:'依已輸入局數與4–6張/局及未知燒牌推估；不是已知牌階'};
}
function createClient(storage){
 let memory={shoe_id:null,records:[],pending:null},storageError=false;
 function save(){try{if(storage)storage.setItem(KEY,JSON.stringify(memory));}catch(_){storageError=true;}}
 function reconcile(history,shoe){
  if(memory.shoe_id!==shoe){let old=null;try{old=JSON.parse(storage?.getItem(KEY)||'null');}catch(_){storageError=true;}memory=old?.shoe_id===shoe&&Array.isArray(old.records)?old:{shoe_id:shoe,records:[],pending:null};}
  const matches=r=>Number.isInteger(r.round)&&r.round>=1&&r.fingerprint===history.slice(0,r.round-1).join('');
  memory.records=memory.records.filter(r=>r.round<=history.length&&matches(r)&&r.actual===history[r.round-1]);
  const pending=memory.pending;
  if(pending&&matches(pending)&&history.length>=pending.round){if(!memory.records.some(r=>r.round===pending.round))memory.records.push({...pending,actual:history[pending.round-1]});memory.pending=null;}
  else if(pending&&(!matches(pending)||pending.round!==history.length+1))memory.pending=null;
  memory.records.sort((a,b)=>a.round-b.round);save();return memory.records;
 }
 function predict(input,custom){let records=reconcile(input.history,input.shoe_id);const tag=limits(input.history.length+1,custom)?.version||VERSION;if((memory.policy_version&&memory.policy_version!==tag)||(!memory.policy_version&&tag!==VERSION)){memory={shoe_id:input.shoe_id,records:[],pending:null,policy_version:tag};records=[];}memory.policy_version=tag;const out=infer(input,records,custom);if(storageError){out.direction='Skip';out.eligible=false;out.reasons.push('storage_error');}
  memory.pending={...out,fingerprint:input.history.join('')};save();if(storageError){out.direction='Skip';out.eligible=false;out.reasons.push('storage_error');}return out;
 }
 return {predict,reconcile,export:()=>JSON.parse(JSON.stringify(memory)),clear:()=>{memory={shoe_id:null,records:[],pending:null};storageError=false;try{storage?.removeItem(KEY);}catch(_){storageError=true;}}};
}
return {VERSION,KEY,DEFAULTS,CONSERVATIVE_DEFAULTS,limits,interpolate,config,infer,state,createClient};
});
