"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),cp=require("node:child_process"),crypto=require("node:crypto");
const profile=JSON.parse(fs.readFileSync("cardcount_profile.json","utf8"));
async function make(mode="post_draw_unconditional_v1",banker=.05,player=0){
 const store=new Map([["bgs_xgb_final_shoe_id_v5","ev-test"]]);
 const config={...profile,settlement:{schema_version:1,mode,banker_commission:banker,player_commission:player,tie_settlement:"push"}};
 const ctx={console,document:{getElementById:()=>null},localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},fetch:async f=>({ok:true,json:async()=>f==="cardcount_profile.json"?config:JSON.parse(fs.readFileSync(f,"utf8"))})};
 ctx.window=ctx;vm.createContext(ctx);
 for(const f of ["app256forward.js","app256continuation.js","particle_filter_runtime.js","macro_ema.js","browser_calibration.js","final_probability_runtime.js"])vm.runInContext(fs.readFileSync(f,"utf8"),ctx,{filename:f});
 const api=ctx.__BGS_FINAL56__;await api.loadModels();return {api,ctx,store};
}
const close=(a,b)=>assert(Math.abs(a-b)<1e-11,a+" != "+b);
(async()=>{
 const {api,store}=await make(),legacy=(await make("legacy_conditional")).api;
 const cases=[];
 for(const T of [0,.1,.6,1])for(const q of [.1,.49,.51,.8]){
  const p={B:(1-T)*q,P:(1-T)*(1-q),T},ev=api.settledOutcomeEv(p);
  close(ev.banker,.95*p.B-p.P);close(ev.player,p.P-p.B);close(ev.probabilities.B+ev.probabilities.P+ev.probabilities.T,1);
  const fromQ=api.conditionalSettlement(q,T);close(fromQ.banker,ev.banker);close(fromQ.player,ev.player);
  cases.push({p,banker:ev.banker,player:ev.player});
 }
 const allTie=api.settledOutcomeEv({B:0,P:0,T:1});close(allTie.banker,0);close(allTie.player,0);
 assert.throws(()=>api.settledOutcomeEv({B:.51,P:.49,T:.1}),/sum/);
 assert.throws(()=>api.settledOutcomeEv({B:NaN,P:.4,T:.1}),/invalid/);
 assert.throws(()=>api.settledOutcomeEv({B:.5,P:.4,T:.1},{banker_commission:-.05,player_commission:0,tie_settlement:"push"}),/invalid/);
 const custom=(await make("post_draw_unconditional_v1",.02,.01)).api;
 close(custom.settledOutcomeEv({B:.5,P:.4,T:.1}).banker,.98*.5-.4);
 close(custom.settledOutcomeEv({B:.5,P:.4,T:.1}).player,.99*.4-.5);
 const customRun=custom.applyFinalPrediction("BPPBTBBPPTBBPPBTBP".split("")),customFinal=customRun.settlement_audit.final;
 close(customRun.ev_banker,.98*customFinal.probabilities.B-customFinal.probabilities.P);
 close(customRun.ev_player,.99*customFinal.probabilities.P-customFinal.probabilities.B);
 const python="import json,sys;from tools.post_draw_report import settled_ev;rows=json.load(sys.stdin);json.dump([settled_ev(r['p']) for r in rows],sys.stdout)";
 const mirrored=JSON.parse(cp.execFileSync("python",["-c",python],{input:JSON.stringify(cases),encoding:"utf8"}));
 mirrored.forEach((x,i)=>{close(x.banker,cases[i].banker);close(x.player,cases[i].player);});
 store.set("bgs_xgb_final_training_v5",JSON.stringify(Array.from({length:8},(_,i)=>({shoe_id:"ev-test",round_index:i+1,final_p_b:.51,predicted_direction:"Skip"}))));
 assert.equal(api.volumeGuardState(60,.55,api.decisionPolicy(60,.3)).active,false,"missing tie probabilities were treated as zero");
 store.delete("bgs_xgb_final_training_v5");
 let comparisons=0;
 for(const h of ["B","T","BPPBTBBPPTBBPPBTBP","BPT".repeat(18)]){
  const a=api.applyFinalPrediction(h.split("")),b=legacy.applyFinalPrediction(h.split("")),r=a.finalProbability,audit=r.settlement_audit;
  assert.equal(audit.probability_basis,"post_draw_unconditional");assert.equal(audit.pf_stage_applications,1);
  assert.equal(JSON.stringify(r.extended),JSON.stringify(b.finalProbability.extended),"57D definitions changed");
  assert.equal(JSON.stringify(r.physics),JSON.stringify(b.finalProbability.physics),"Physics/PF output changed");
  assert.equal(r.skip_diagnostics.early_tie_penalty_applied,false,"old failed tie penalty was enabled");
  close(audit.mixed.probabilities.B,r.physics[23]);close(audit.mixed.probabilities.T,r.physics[25]);
  close(audit.final.probabilities.B,(1-r.physics[25])*r.finalPB);
  close(r.p_player,audit.final.probabilities.P);
  close(a.ev_banker,audit.final.banker);close(a.ev_player,audit.final.player);
  if(a.direction!=="Skip")assert((a.direction==="B"?a.ev_banker:a.ev_player)>0);
  for(const s of ["direct_physics","pf_posterior","mixed"]){
   const d=r.draw_report[s];assert(d.available&&d.normalized);close(Object.values(d.probabilities).reduce((x,y)=>x+y),1);
   assert(d.expected_cards>=4&&d.expected_cards<=6);
  }
  comparisons++;
 }
 // Original PF rules tested via read-only instrumentation, without editing its file.
 const ctx={};ctx.window=ctx;vm.createContext(ctx);
 const pf=fs.readFileSync("particle_filter_runtime.js","utf8").replace('if(typeof window!=="undefined")window.__BGS_PARTICLE500__=','window.__PF_RULE_TEST__={deal,bankerDraws};\nif(typeof window!=="undefined")window.__BGS_PARTICLE500__=');
 vm.runInContext(pf,ctx);
 const rule=(bt,p3)=>p3===null?bt<=5:bt<=2?true:bt===3?p3!==8:bt===4?p3>=2&&p3<=7:bt===5?p3>=4&&p3<=7:bt===6?p3===6||p3===7:false;
 for(let b=0;b<=9;b++)for(const p of [null,0,1,2,3,4,5,6,7,8,9])assert.equal(ctx.__PF_RULE_TEST__.bankerDraws(b,p),rule(b,p));
 function planned(ranks){
  const own=Array(13).fill(4),base=own.slice();let i=0;
  const rng=()=>{const rank=ranks[i++];assert(rank!==undefined,"unexpected extra draw");const total=own.reduce((a,b)=>a+b),before=own.slice(0,rank).reduce((a,b)=>a+b,0);own[rank]--;return (before+.5)/total;};
  const hand=ctx.__PF_RULE_TEST__.deal(base,rng);assert.deepEqual(base,Array(13).fill(4));return hand;
 }
 assert.equal(planned([3,2,3,3]).cardCount,4,"natural must end after four cards");
 assert.equal(planned([1,3,2,4]).cardCount,4,"banker natural 9 prevents player third card");
 assert.equal(planned([2,1,2,2,0]).cardCount,5,"player stands; banker draws");
 assert.equal(planned([1,0,2,1,7]).cardCount,5,"banker 3 stands versus player third 8");
 const six=planned([1,0,2,1,6,8]);assert.equal(six.cardCount,6);assert.equal(six.outcome,"T");
 const locked=require("./tools/top_parameters/production_manifest.json").sha256;
 for(const f of ["app256forward.js","app256continuation.js","physics_multitask_model.json","final_probability_model.json"])assert.equal(crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex"),locked[f],"locked model/rules changed");
 console.log(JSON.stringify({passed:true,probability_cases:cases.length,runtime_comparisons:comparisons,banker_rule_cases:110,completed_hand_cases:5,old_extra_tie_penalty:false,feature_dim:57}));
})().catch(e=>{console.error(e);process.exitCode=1;});
