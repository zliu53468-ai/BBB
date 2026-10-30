/* Same-shoe prequential display correction; never a formal wager input. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.__BBB_RELATIVE_BASELINE__=api;})(typeof window!=='undefined'?window:null,function(){
'use strict';
const finite=Number.isFinite,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function settings(c={}){return {...c,min_samples:Math.round(clamp(finite(c.min_samples)?c.min_samples:25,25,200)),prior_strength:clamp(finite(c.prior_strength)?c.prior_strength:100,50,1000),max_delta:clamp(finite(c.max_delta)?c.max_delta:.0025,0,.0025),noise_start:clamp(finite(c.noise_start)?c.noise_start:.55,0,.8),noise_stop:clamp(finite(c.noise_stop)?c.noise_stop:.85,.81,1)};}
function valid(f){const p=f?.directional?.B,q=f?.directional?.P,b=f?.probabilities;return f?.available&&finite(p)&&finite(q)&&p>=0&&q>=0&&Math.abs(p+q-1)<1e-9&&b&&['B','P','T'].every(k=>finite(b[k])&&b[k]>=0)&&Math.abs(b.B+b.P+b.T-1)<1e-9&&b.B+b.P>0;}
function createClient(){
 let activeShoe=null;const journals=new Map();
 function journal(history,shoe,model,c){if(activeShoe!==shoe){journals.clear();activeShoe=shoe;}const key=JSON.stringify([model,settings(c)]),events=(journals.get(key)||[]).filter(e=>e.index<history.length&&e.prefix===history.slice(0,e.index).join('')&&history[e.index]===e.actual);journals.set(key,events);return events;}
 function prepare(history,f,shoe,model,c={}){if(c.enabled!==true||!valid(f)||!history.every(x=>['B','P','T'].includes(x)))return null;return {shoe,model,config:JSON.stringify(settings(c)),index:history.length,prefix:history.join(''),p:f.relative_baseline?.base_pb??f.directional.B};}
 function commit(ticket,actual,history,c={}){if(c.enabled!==true||!ticket||ticket.config!==JSON.stringify(settings(c))||history.length!==ticket.index+1||history.at(-1)!==actual||history.slice(0,-1).join('')!==ticket.prefix||!['B','P','T'].includes(actual)||!finite(ticket.p)||ticket.p<0||ticket.p>1)return false;const events=journal(history,ticket.shoe,ticket.model,c);if(events.some(e=>e.index===ticket.index))return false;events.push({...ticket,actual});return true;}
 function apply(f,c={},history=[],shoe='',model=''){
  if(c.enabled!==true||!valid(f))return f;
  const cfg=settings(c),events=journal(history,shoe,model,c).filter(e=>e.actual!=='T'),n=events.length,residual=events.reduce((sum,e)=>sum+(e.actual==='B'?1:0)-e.p,0),base=f.directional.B;
  const gate=finite(f.uncertainty)?clamp((cfg.noise_stop-f.uncertainty)/(cfg.noise_stop-cfg.noise_start),0,1):0;
  const ready=n>=cfg.min_samples,delta=ready?clamp(residual/(cfg.prior_strength+n),-cfg.max_delta,cfg.max_delta)*gate:0,p=clamp(base+delta,0,1),mass=f.probabilities.B+f.probabilities.P;
  const direction=p===.5?'Balanced':p>.5?'B':'P',applied=Math.abs(p-base)>1e-12;
  const audit={enabled:true,version:c.version,mode:'same_shoe_residual',display_only:true,formal_decision_changed:false,probabilities_changed:applied,applied,ready,directional_samples:n,min_samples:cfg.min_samples,base_pb:base,corrected_pb:p,delta:p-base,max_delta:cfg.max_delta,noise_gate:gate,raw_direction:f.direction,advisory_direction:direction,reason:!ready?'insufficient_settled_predictions':gate===0?'noise_gate_closed':applied?'bounded_bidirectional_residual':'zero_residual',validated:false};
  return {...f,...(applied?{direction,directional:{B:p,P:1-p},probabilities:{B:mass*p,P:mass*(1-p),T:f.probabilities.T},weak_signal:Math.abs(p-.5)<.005}:{}),relative_baseline:audit};
 }
 return {apply,prepare,commit};
}
const client=createClient();return {...client,createClient};});
