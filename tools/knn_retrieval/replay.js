/* Offline-only adapter. Frozen PR72 runtime, strong-only + additional irreversible
 * per-shoe two-loss stop. No UI, model source, thresholds or core calculations edited. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const [input,output,signalFile,group='baseline']=process.argv.slice(2);
const dir=path.resolve(__dirname,'../..'),payload=JSON.parse(fs.readFileSync(input,'utf8'));
const config=signalFile?JSON.parse(fs.readFileSync(signalFile,'utf8')):null;
if(!['baseline','ungated','gated'].includes(group))throw Error('invalid group');
const names=['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'];
const hashes=Object.fromEntries([...names,'cardcount_profile.json','physics_multitask_model.json','final_probability_model.json'].map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,f))).digest('hex')]));
const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'pr72_blob_manifest.json'),'utf8'));
for(const [file,expected]of Object.entries(manifest.git_blob_sha1)){
 const bytes=fs.readFileSync(path.join(dir,file)),actual=crypto.createHash('sha1').update(Buffer.from(`blob ${bytes.length}\0`)).update(bytes).digest('hex');
 if(actual!==expected)throw Error('Frozen PR72 source differs: '+file);
}
function seeded(seed){let x=seed>>>0;return()=>{x+=0x6D2B79F5;let t=x;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
const logit=p=>Math.log(Math.max(1e-6,Math.min(1-1e-6,p))/(1-Math.max(1e-6,Math.min(1-1e-6,p))));
async function context(seed){
 const store=new Map(),math=Object.create(Math);math.random=seeded(seed);
 const ctx={console,Math:math,localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
 document:{getElementById:()=>null},fetch:async url=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(dir,String(url).split('?')[0]),'utf8'))})};
 ctx.window=ctx;vm.createContext(ctx);for(const f of names)vm.runInContext(fs.readFileSync(path.join(dir,f),'utf8'),ctx,{filename:f});
 const api=ctx.__BGS_FINAL56__;await api.loadModels();if(!api.getModelStatus().physics||!api.getModelStatus().final56)throw Error('models unavailable');
 return {api,M:ctx.__BGS_MACRO_EMA__,store};
}
(async()=>{
 const byShoe=new Map();for(const row of payload.rows){const a=byShoe.get(row.shoe_id)||[];a.push(row);byShoe.set(row.shoe_id,a);}
 const rows=[];let index=0;
 for(const [shoe,events]of byShoe){
  // Fresh VM per shoe, same seed across groups: no hidden PF/EMA carry-over.
  const {api,M,store}=await context((payload.seed||20261029)+index++*104729);
  store.set('bgs_xgb_final_shoe_id_v5',shoe);const history=[];let losses=0,halted=false;
  for(const row of events){
   if(row.round_index!==history.length+1)throw Error('nonconsecutive round');
   let prediction=null;
   if(history.length){
    prediction=api.applyFinalPrediction(history);const r=prediction.finalProbability;
    if(r.error||r.mode!=='physics_primary'||!Number.isFinite(r.finalPB))throw Error('runtime failed');
    const raw=prediction.direction,tier=r.entry_tier;
    const standard=tier==='strong'&&r.stake_multiplier===1&&['B','P'].includes(raw);
    let action=standard&&!halted?raw:'Skip',fused=r.finalPB,gateUsed=false,signal=null;
    if(group!=='baseline'){
     signal=config.signals[`${shoe}:${row.round_index}`];if(!signal)throw Error('missing prefix retrieval');
     const fit=config.models[group];
     gateUsed=group==='ungated'||signal.mean_distance<=fit.max_distance&&signal.entropy<=fit.max_entropy;
     if(gateUsed){const [b,w0,w1]=fit.weights;const z=b+w0*logit(r.finalPB)+w1*logit(signal.p_b);fused=1/(1+Math.exp(-z));}
     if(action!=='Skip'&&(action==='B'?fused<.5:fused>.5))action='Skip';
    }
    const executed={...prediction,direction:action,finalProbability:{...r,finalDirection:action}};
    api.registerPrediction(history,executed);
    rows.push({data_kind:payload.data_kind,shoe_id:shoe,round_index:row.round_index,outcome:row.outcome,
     raw_direction:raw,entry_tier:tier,stake_multiplier:r.stake_multiplier,standard_eligible:standard,
     direction:action,p_b:r.finalPB,fused_p_b:fused,retrieval_used:gateUsed,halted_before:halted,
     signal:signal?{...signal,
      w_count:raw==='Skip'?null:raw==='B'?signal.b_count:signal.p_count,
      l_count:raw==='Skip'?null:raw==='B'?signal.p_count:signal.b_count,
      w_probability:raw==='Skip'||signal.b_count+signal.p_count===0?null:(raw==='B'?signal.b_count:signal.p_count)/(signal.b_count+signal.p_count),
      l_probability:raw==='Skip'||signal.b_count+signal.p_count===0?null:(raw==='B'?signal.p_count:signal.b_count)/(signal.b_count+signal.p_count)}:null});
    if(action!=='Skip'&&row.outcome!=='T'){
     losses=action===row.outcome?0:losses+1;if(losses>=2)halted=true;
    }
   }
   const observed=M.normalizeHand(row);
   if(prediction)api.settlePending(row.outcome,observed,history);
   history.push(row.outcome);api.recordCompletedHand(history,observed);
  }
 }
 fs.writeFileSync(output,JSON.stringify({data_kind:payload.data_kind,group,runtime_sha256:hashes,rows})+'\n');
 console.log(JSON.stringify({data_kind:payload.data_kind,group,rows:rows.length}));
})().catch(e=>{console.error(e);process.exit(1);});
