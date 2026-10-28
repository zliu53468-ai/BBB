'use strict';
// Headless adapter, using the actual browser functions. Frozen sources untouched.
const fs=require('fs'),path=require('path'),cp=require('child_process'),ROOT=path.resolve(__dirname,'..');
const BASE='c6f4a0afbf9fc9de3806ae662319471daadfb405';
const source=(file,legacy,revision)=>((legacy||revision)&&['cardcount_profile.json','final_probability_runtime.js','physics_priority_runtime.js'].includes(file))?cp.execFileSync('git',['show',(revision||BASE)+':'+file],{cwd:ROOT,encoding:'utf8'}):fs.readFileSync(path.join(ROOT,file),'utf8');
async function runtime({legacy=false,memoized=true,fastIO=true,revision=null}={}){
 const data=new Map(),win={},doc={getElementById:()=>null},storage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,String(v)),removeItem:k=>data.delete(k)};
 const fetch=async f=>({ok:true,json:async()=>JSON.parse(source(String(f).split('?')[0],legacy,revision))});
 let rowShoe=null,rows=[],canonical=new WeakSet();
 function rowReset(){const shoe=data.get('bgs_xgb_final_shoe_id_v5');if(shoe!==rowShoe){rowShoe=shoe;rows=[];canonical=new WeakSet();}}
 win.__labReadRows=()=>{rowReset();return rows.slice();};
 win.__labWriteRows=input=>{rowReset();rows=input.map(r=>{if(canonical.has(r))return r;const copy=JSON.parse(JSON.stringify(r));canonical.add(copy);return copy;});};
 win.__labPosteriorCache=new Map();
 for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js',...(!legacy?['point_conditioned_pf.js','physics_priority_runtime.js']:[]),'final_probability_runtime.js']){
  let code=source(f,legacy,revision);
  if(f==='particle_filter_runtime.js'&&memoized){
   const start=code.indexOf('  const rng=rng32((seedFromHistory(seq)^(Number.isInteger(config.seed_offset)?config.seed_offset:0))>>>0);'),end=code.indexOf('  const winner=normalizeBlock(posterior.slice(23,26));',start);if(start<0||end<0)throw Error('PF memo anchor mismatch');
   const original=code.slice(start,end);
   code=code.slice(0,start)+`const memoKey=(config.seed_offset||0)+'|'+seq.join('')+'|'+Math.round(clip(config.forecast_draws??FORECAST_DRAWS,1,5));
let forecast,posterior,lastEssRatio,resamplingCount,forecastDraws;
const cached=window.__labPosterior?.key===memoKey?window.__labPosterior:seq.length<=8?window.__labPosteriorCache.get(memoKey):null;
if(cached){({forecast,posterior,lastEssRatio,resamplingCount,forecastDraws}=cached);window.__labPosterior={key:memoKey,...cached};}
else{const pack=(()=>{${original}\nreturn{forecast,posterior,lastEssRatio,resamplingCount,forecastDraws};})();({forecast,posterior,lastEssRatio,resamplingCount,forecastDraws}=pack);window.__labPosterior={key:memoKey,...pack};if(seq.length<=8)window.__labPosteriorCache.set(memoKey,pack);}
`+code.slice(end);
  }
  if(f==='final_probability_runtime.js'&&fastIO){
   const read='function readRows(){try{const r=JSON.parse(localStorage.getItem(TRAINING_KEY)||"[]");return Array.isArray(r)?r:[];}catch(_){return[];}}';
   const write='function writeRows(rows){try{localStorage.setItem(TRAINING_KEY,JSON.stringify(rows.slice(-MAX_TRAINING_ROWS)));}catch(_){}}';
   if(!code.includes(read)||!code.includes(write))throw Error('Pinned journal adapter mismatch');
   code=code.replace(read,'function readRows(){return window.__labReadRows();}').replace(write,'function writeRows(rows){window.__labWriteRows(rows.slice(-MAX_TRAINING_ROWS));}');
  }
  new Function('window','document','localStorage','fetch',code)(win,doc,storage,fetch);
 }
 await win.__BGS_FINAL56__.loadModels();return {win,data,api:win.__BGS_FINAL56__,storage,doc};
}
function settle(r,history,p,outcome,observation){
 const ticket=r.api.prepareDisplayCalibrationTickets(history,p.base_raw_next_hand_forecast||p.raw_next_hand_forecast);
 r.api.registerPrediction(history,p);r.api.settlePending(outcome,observation,history);
 const after=[...history,outcome];r.data.set('bgs256d_frozen_6x15_forward_v18',JSON.stringify({history:after}));
 r.api.commitDisplayCalibrationTickets(ticket,outcome,after);r.api.recordCompletedHand(after,observation);
}
module.exports={runtime,settle,BASE,ROOT};
