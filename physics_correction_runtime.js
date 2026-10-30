/* Stateless Physics correction before adaptive XGB. No outcome/tie override. */
(function(root,f){const api=f();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_PHYSICS_CORRECTION__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';const finite=Number.isFinite,clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function apply(base,physics,c={}){
 if(!finite(base)||base<0||base>1)throw Error('invalid_base_probability');
 const numeric=(k,f,a,b)=>clamp(finite(c[k])?c[k]:f,a,b),max=numeric('max_weight',.20,0,.20),min=numeric('min_weight',.10,0,max),maxDelta=numeric('max_delta',.005,0,.005),low=numeric('uncertainty_low',.35,0,.8),high=numeric('uncertainty_high',.65,low+.01,1),scale=numeric('signal_scale',.05,.005,.25);
 const valid=Array.isArray(physics)&&physics.length===48&&[23,24,25,42].every(i=>finite(physics[i]))&&[23,24,25].every(i=>physics[i]>=0)&&Math.abs(physics[23]+physics[24]+physics[25]-1)<1e-5&&physics[23]+physics[24]>0&&physics[42]>=0&&physics[42]<=1;
 const q=valid?physics[23]/(physics[23]+physics[24]):null,u=valid?physics[42]:null,gate=valid?clamp((high-u)/(high-low),0,1):0,signal=valid?clamp(Math.abs(q-.5)/scale,0,1):0,nominal=min+(max-min)*signal,proposed=c.enabled===true&&valid?nominal*gate:0,delta=valid?clamp(proposed*(q-base),-maxDelta,maxDelta):0,value=clamp(base+delta,0,1),effective=valid&&Math.abs(q-base)>1e-12?clamp((value-base)/(q-base),0,max):0;
 return {value,audit:{enabled:c.enabled===true,version:c.version||'PHYSICS_BOUNDED_CORRECTION_V1',base_pb:base,physics_pb:q,corrected_pb:value,physics_uncertainty:u,uncertainty_gate:gate,signal_strength:signal,nominal_weight:nominal,proposed_weight:proposed,weight:effective||0,max_weight:max,delta:value-base,max_delta:maxDelta,reason:c.enabled!==true?'disabled':!valid?'invalid_physics':gate===0?'high_uncertainty':Math.abs(delta)<1e-12?'zero_gap':'bounded_physics_correction',tie_rule_changed:false,validated:false}};
}
return {apply};});
