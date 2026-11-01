/* PF-first candidate. Pure combination helper; no sampling/model training here. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_PF_PRIMARY__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),side=q=>q>.5?'B':q<.5?'P':'Balanced';
function seed(){const v=Array(48).fill(0);for(let i=0;i<3;i++)v[i]=1/3;for(let i=3;i<23;i++)v[i]=.1;v[23]=v[24]=.45;v[25]=.1;for(let i=26;i<39;i++)v[i]=5/13;v[42]=.5;v[44]=v[45]=5/13;return v;}
function combine(posterior,direct,diagnostics,noise,custom={}){
 if(!Array.isArray(posterior)||posterior.length!==48||!posterior.every(Number.isFinite)||!Array.isArray(direct)||direct.length!==48||!direct.every(Number.isFinite))throw Error('pf_primary_invalid_vectors');
 const mass=posterior[23]+posterior[24]+posterior[25],dm=direct[23]+direct[24];if(mass<=0||dm<=0||posterior.slice(23,26).some(v=>v<0))throw Error('pf_primary_invalid_probabilities');
 const raw={B:posterior[23]/mass,P:posterior[24]/mass,T:posterior[25]/mass},directional=raw.B+raw.P;if(directional<=0)throw Error('pf_primary_no_directional_mass');
 const q=raw.B/directional,physical=direct[23]/dm,a=side(q),b=side(physical),trusted=Number.isFinite(noise)&&noise<=clamp(custom.max_physics_noise??.70)&&direct[42]<=clamp(custom.max_physics_uncertainty??.70);
 const conflict=a!=='Balanced'&&b!=='Balanced'&&a!==b,forcedSkip=trusted&&conflict&&Math.abs(2*physical-1)>=Math.max(.02,custom.conflict_gap??.04);
 const requested=trusted&&a===b&&a!=='Balanced'?clamp(custom.physics_weight??.08,0,Math.min(.12,custom.max_physics_weight??.12)):0;
 const limit=clamp(custom.max_delta??.004,0,.006);let final=q+clamp(requested*(physical-q),-limit,limit);if(a==='Balanced'||side(final)!==a)final=q;
 const out=posterior.slice();out[23]=directional*final;out[24]=directional*(1-final);out[25]=raw.T;out[39]=.95*out[23]-out[24];out[40]=out[24]-out[23];out[41]=out[39]-out[40];out[42]=Math.max(clamp(diagnostics?.posterior_uncertainty??1),clamp(direct[42]));
 const effective=Math.abs(physical-q)>1e-12?(final-q)/(physical-q):0;
 return {physics:out,audit:{enabled:true,version:'PF_PRIMARY_PHYSICS_SECONDARY_V1',raw_pf_probabilities:raw,raw_pf_direction:a,raw_pf_directional_pb:q,physics_direction:b,physics_directional_pb:physical,physics_trusted:trusted,physics_conflict:conflict,forced_skip_by_conflict:forcedSkip,physics_weight_requested:requested,physics_weight_effective:effective,pf_weight_effective:1-effective,delta:final-q,max_delta:limit,calibrated_probabilities:{B:out[23],P:out[24],T:out[25]},ev_banker:out[39],ev_player:out[40]}};
}
return {seed,combine};
});
