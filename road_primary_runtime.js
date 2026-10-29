/* Road/Markov primary candidate; Physics+PF is a bounded secondary correction. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_ROAD_PRIMARY__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';const tokens=['B','P','T'],clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),side=x=>Math.abs(x)<1e-12?'Balanced':x>0?'B':'P';
function estimate(seq,c={}){
 if(!Array.isArray(seq)||!seq.every(x=>tokens.includes(x)))throw Error('road_invalid_history');
 const prior=c.prior||{B:.45,P:.45,T:.10},values=tokens.map(t=>prior[t]);if(!values.every(v=>Number.isFinite(v)&&v>=0)||Math.abs(values.reduce((a,b)=>a+b,0)-1)>1e-9)throw Error('road_invalid_prior');
 // No hand-number or stage gate: selection uses observed context support only.
 const b=c.adaptive_bayesian_v1,active=b?.enabled===true;
 const finite=(v,f,lo,hi)=>Number.isFinite(v)?clamp(v,lo,hi):f;
 const history=seq.slice(-Math.max(4,Math.min(500,c.window??200))),minimum=Math.max(1,Math.floor(c.min_observations??12)),strength=Math.max(1,c.prior_strength??24);
 const backoff=[];let counts={B:0,P:0,T:0},n=0,order=Math.max(1,Math.min(c.five_level_v1?.enabled===true?5:3,Math.floor(c.order??2)));
 for(;order>=1;order--){counts={B:0,P:0,T:0};n=0;const context=history.slice(-order).join('');for(let i=order;i<history.length;i++)if(history.slice(i-order,i).join('')===context){counts[history[i]]++;n++;}backoff.push({order,observations:n,eligible:n>=minimum});if(n>=minimum)break;}
 if(n<minimum)return {available:false,reason:'insufficient_road_context',history_rounds:seq.length,observations:n,minimum,backoff,order:Math.max(1,order),counts};
 const supportScale=finite(b?.support_scale,12,1,100);
 const extra=finite(b?.extra_prior_strength,24,0,100);
 const effectiveStrength=strength+(active?extra*supportScale/(n+supportScale):0);
 const diagnostics=active?{bayesian:{enabled:true,version:b.version||'CONTEXT_ADAPTIVE_BAYES_V1',selected_order:order,observations:n,base_prior_strength:strength,effective_prior_strength:effectiveStrength,data_weight:n/(n+effectiveStrength),prior_weight:effectiveStrength/(n+effectiveStrength)}}:{};
 const probabilities=Object.fromEntries(tokens.map(t=>[t,(counts[t]+effectiveStrength*prior[t])/(n+effectiveStrength)])),mass=probabilities.B+probabilities.P,q=probabilities.B/mass;
 return {available:true,reason:'road_candidate_ready',history_rounds:seq.length,observations:n,minimum,backoff,order,counts,prior:{...prior},prior_strength:effectiveStrength,...diagnostics,probabilities,directional_pb:q,direction:side(2*q-1)};
}
function reliability(physical,diagnostics,noise,c={}){
 const b=c.adaptive_pipeline_v1;if(b?.enabled!==true)return null;
 const unit=v=>Number.isFinite(v)?clamp(v):0;
 const uncertainty=Number.isFinite(physical?.[42])?clamp(physical[42]):1;
 const quality=unit(diagnostics?.particle_quality),ess=unit(diagnostics?.recent_ess_ratio);
 const noiseScore=Number.isFinite(noise)?clamp(noise):1;
 const rawValue=Math.min(1-uncertainty,1-noiseScore,quality,ess);
 const configured=c.five_level_v1?.reliability_anchors;
 const valid=Array.isArray(configured)&&configured.length===5&&configured.every((v,i,a)=>Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)&&v[0]===i/4&&v[1]>=0&&v[1]<=v[0]&&(!i||v[1]>=a[i-1][1]));
 const points=valid?configured:[[0,0],[.25,.15],[.5,.35],[.75,.60],[1,1]];
 const five=c.five_level_v1?.enabled===true,segment=Math.min(3,Math.floor(rawValue*4)),fraction=(rawValue-points[segment][0])/.25;
 const value=five?points[segment][1]+fraction*(points[segment+1][1]-points[segment][1]):rawValue;
 const levels=five?{five_level:{enabled:true,raw_reliability:rawValue,lower_level:segment+1,upper_level:segment+2,fraction,nearest_level:Math.min(5,1+Math.round(rawValue*4)),mapped_reliability:value,anchors:points}}:{};
 return {enabled:true,version:b.version||'ADAPTIVE_PIPELINE_V1',reliability:value,...levels,uncertainty,noise:noiseScore,particle_quality:quality,ess_ratio:ess,missing_diagnostics:!Number.isFinite(diagnostics?.particle_quality)||!Number.isFinite(diagnostics?.recent_ess_ratio),probability_shrink_count:1};
}
function combine(road,physical,c={},diagnostics=null,noise=null){
 if(!Array.isArray(physical)||physical.length!==48||!physical.every(Number.isFinite))throw Error('road_invalid_physics');
 const pm=physical[23]+physical[24],total=pm+physical[25];if(!(pm>0&&total>0))throw Error('road_invalid_mass');
 const base={B:physical[23]/total,P:physical[24]/total,T:physical[25]/total},pq=base.B/(base.B+base.P);
 const adaptive=reliability(physical,diagnostics,noise,c);
 if(!road?.available&&adaptive){const q=.5+adaptive.reliability*(pq-.5),out=physical.slice(),dm=base.B+base.P;out[23]=dm*q;out[24]=dm*(1-q);out[25]=base.T;out[39]=.95*out[23]-out[24];out[40]=out[24]-out[23];out[41]=out[39]-out[40];return {physics:out,audit:{enabled:true,version:c.version,available:false,fallback:'physics_pf',reason:road?.reason||'road_unavailable',road:road||null,physical_probabilities:base,physical_directional_pb:pq,physics_weight_effective:adaptive.reliability,road_weight_effective:0,delta:q-pq,adaptive:{...adaptive,scope:'fallback_physics_signal',raw_probabilities:base,adjusted_probabilities:{B:out[23],P:out[24],T:out[25]}}}};}
 if(!road?.available)return {physics:physical.slice(),audit:{enabled:true,version:'ROAD_PRIMARY_PHYSICS_ADJUSTMENT_V1',available:false,fallback:'physics_pf',reason:road?.reason||'road_unavailable',road:road||null,physical_probabilities:base,physical_directional_pb:pq,physics_weight_effective:1,road_weight_effective:0,delta:0}};
 const q=road.directional_pb,uncertainty=clamp(physical[42]),trusted=uncertainty<=clamp(c.max_physics_uncertainty??.70),requested=trusted?clamp(c.physics_weight??.10,0,Math.min(.15,c.max_physics_weight??.15)):0;
 const limit=clamp(c.max_delta??.004,0,.005);let delta=clamp(requested*(adaptive?adaptive.reliability:1)*(pq-q),-limit,limit);if(road.direction==='Balanced'||side(2*(q+delta)-1)!==road.direction)delta=0;
 const dm=base.B+base.P,out=physical.slice();out[23]=dm*(q+delta);out[24]=dm*(1-q-delta);out[25]=base.T;out[39]=.95*out[23]-out[24];out[40]=out[24]-out[23];out[41]=out[39]-out[40];
 const effective=Math.abs(pq-q)>1e-12?Math.abs(delta/(pq-q)):0;
 return {physics:out,audit:{enabled:true,version:'ROAD_PRIMARY_PHYSICS_ADJUSTMENT_V1',available:true,fallback:null,...(adaptive?{adaptive:{...adaptive,scope:'physical_correction_only',raw_probabilities:base,adjusted_probabilities:{B:out[23],P:out[24],T:out[25]}}}:{}),reason:'road_primary_physics_secondary',road,road_direction:road.direction,road_directional_pb:q,physical_probabilities:base,physical_directional_pb:pq,physics_trusted:trusted,physics_weight_requested:requested,physics_weight_effective:effective,road_weight_effective:1-effective,delta,max_delta:limit,mixed_probabilities:{B:out[23],P:out[24],T:out[25]},tie_mass_source:'physics_pf',ev_banker:out[39],ev_player:out[40]}};
}
return {estimate,combine,reliability};
});
