/* B↔P advisory only. Historical change evidence is not a guaranteed next result. */
(function(root,f){const a=f();if(typeof module==='object'&&module.exports)module.exports=a;if(root)root.__BBB_RELATIVE_BASELINE__=a;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function apply(f,c={},history=[]){
 if(c.enabled!==true||!f?.available)return f;
 const p=f.directional?.B,q=f.directional?.P;
 if(!Number.isFinite(p)||!Number.isFinite(q)||Math.abs(p+q-1)>1e-9||p<0||q<0)return f;
 const integer=(key,fallback,lo,hi)=>Number.isFinite(c[key])?Math.round(clamp(c[key],lo,hi)):fallback;
 const recentWindow=integer('recent_window',8,4,12),referenceWindow=integer('reference_window',16,8,24),minRecent=integer('min_recent',6,4,recentWindow),minReference=integer('min_reference',8,4,referenceWindow);
 const seq=history.filter(x=>x==='B'||x==='P'),recent=seq.slice(-recentWindow),reference=seq.slice(-recentWindow-referenceWindow,-recentWindow);
 const prior=Number.isFinite(c.prior_strength)?clamp(c.prior_strength,2,20):6;
 const mean=a=>(a.filter(x=>x==='B').length+prior*.5)/(a.length+prior);
 const current=mean(recent),previous=mean(reference),shift=current-previous;
 const minimum=Number.isFinite(c.min_shift)?clamp(c.min_shift,.05,.3):.12;
 const max=Number.isFinite(c.max_raw_gap)?clamp(c.max_raw_gap,0,.03):.02;
 const enough=recent.length>=minRecent&&reference.length>=minReference;
 const evidence=enough&&Math.abs(shift)>=minimum&&Math.abs(current-.5)>=.05;
 const candidate=shift>0?'B':'P',aligned=candidate==='B'?current>.5:current<.5;
 const eligible=evidence&&aligned&&Math.abs(p-q)<=max+1e-12;
 const direction=eligible?candidate:f.direction;
 return {...f,direction,relative_baseline:{enabled:true,version:c.version||'BIDIRECTIONAL_ADVISORY_V2',mode:'bidirectional_switch',applied:eligible,display_only:true,probabilities_changed:false,formal_decision_changed:false,raw_direction:f.direction,advisory_direction:direction,switch:eligible&&direction!==f.direction?f.direction+'2'+direction:null,recent_samples:recent.length,reference_samples:reference.length,recent_pb:current,reference_pb:previous,shift,min_shift:minimum,max_raw_gap:max,reason:!enough?'insufficient_context':!evidence||!aligned?'insufficient_change_evidence':Math.abs(p-q)>max?'strong_model_preserved':'symmetric_change_advisory',validated:false}};
}
return {apply};});
