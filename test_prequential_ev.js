"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),M=require('./browser_calibration.js');
const profile=JSON.parse(fs.readFileSync('cardcount_profile.json','utf8'));
delete profile.winrate_priority_v1; // Exercise the original version-specific regression fixture.
const config={...profile.local_calibration,stage_anchors:{}};
const memory=()=>{const m=new Map();return {getItem:k=>m.get(k)??null,setItem:(k,v)=>m.set(k,String(v)),removeItem:k=>m.delete(k)}};
const forecast=()=>({available:true,directional:{B:.55,P:.45},probabilities:{B:.495,P:.405,T:.1},uncertainty:.2,calibration_bounds:{low:.4,high:.6}});
function learn(candidateAction='B',missingTrial=false){const store=memory(),client=M.createClient(store),h=[];
 for(let i=0;i<55;i++){const f=forecast(),p=client.apply(f,h,'s','m',config).proposed_forecast.directional.B;
 if(!missingTrial)f.validation_trial={proposal_p:p,base_action:'B',candidate_action:candidateAction,banker_commission:.05,player_commission:0};
 const ticket=client.prepare(h,f,'s','m');assert(!client.commit(ticket,'B',h,config),'future outcome must be saved first');h.push('B');assert(client.commit(ticket,'B',h,config));assert(!client.commit(ticket,'B',h,config),'duplicate');}
 return {store,client,h};}
const good=learn(),learned=good.client.apply(forecast(),good.h,'s','m',config);assert(learned.status.ready&&learned.status.applied&&learned.status.bet_gate);assert(learned.status.delta>0&&learned.status.delta<=.005+1e-12);
const bad=learn('P'),blocked=bad.client.apply(forecast(),bad.h,'s','m',config);assert(blocked.status.brier_gain>0);assert(!blocked.status.bet_gate&&!blocked.status.applied,'Brier gain must not override worse bet hit/EV');
const missing=learn('B',true);assert(!missing.client.apply(forecast(),missing.h,'s','m',config).status.ready,'missing formal comparison must fail closed');
const before=JSON.stringify(good.client.inspect(good.h,'s','m',config));good.client.apply(forecast(),good.h,'s','m',config);assert.equal(JSON.stringify(good.client.inspect(good.h,'s','m',config)),before,'inference cannot learn');
const tieForecast=forecast(),tieProposal=good.client.apply(tieForecast,good.h,'s','m',config).proposed_forecast.directional.B;
tieForecast.validation_trial={proposal_p:tieProposal,base_action:'B',candidate_action:'B',banker_commission:.05,player_commission:0};const tieTicket=good.client.prepare(good.h,tieForecast,'s','m');good.h.push('T');assert(good.client.commit(tieTicket,'T',good.h,config));assert.equal(good.client.inspect(good.h,'s','m',config).directional_samples,55,'tie must not become a B/P training label');
const reloaded=M.createClient(good.store);assert.equal(reloaded.apply(forecast(),good.h,'s','m',config).forecast.directional.B,learned.forecast.directional.B);assert(!reloaded.apply(forecast(),good.h,'different','m',config).status.applied);assert(!reloaded.apply(forecast(),good.h,'s','different',config).status.applied);
const noisy=forecast();noisy.uncertainty=.99;assert(!reloaded.apply(noisy,good.h,'s','m',config).status.applied);assert(!reloaded.apply(forecast(),good.h.slice(0,5),'s','m',config).status.applied);
async function runtime(enabled){const store=memory(),settings=structuredClone(profile);settings.local_calibration.formal_ev_enabled=enabled;store.setItem('bgs_xgb_final_shoe_id_v5','s');const ctx={console,localStorage:store,document:{getElementById:()=>null},fetch:async f=>({ok:true,json:async()=>f==='cardcount_profile.json'?settings:JSON.parse(fs.readFileSync(f,'utf8'))})};ctx.window=ctx;vm.createContext(ctx);for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','final_probability_runtime.js'])vm.runInContext(fs.readFileSync(f,'utf8'),ctx,{filename:f});await ctx.__BGS_FINAL56__.loadModels();return {api:ctx.__BGS_FINAL56__,store};}
(async()=>{const a=await runtime(true),b=await runtime(false);const empty=a.api.applyFinalPrediction([]),ticket=a.api.prepareCalibrationTicket([],empty),tag=ticket.model;
 const seed=M.createClient(a.store),h=[];
 for(let i=0;i<55;i++){const f=forecast(),p=seed.apply(f,h,'s',tag,config).proposed_forecast.directional.B;f.validation_trial={proposal_p:p,base_action:'B',candidate_action:'B',banker_commission:.05,player_commission:0};const t=seed.prepare(h,f,'s',tag);h.push('B');assert(seed.commit(t,'B',h,config));}
 const p=a.api.applyFinalPrediction(h),base=b.api.applyFinalPrediction(h),r=p.finalProbability;
 assert(p.browser_calibration.ready&&p.browser_calibration.applied,'formal calibration must actually participate');
 assert.notEqual(r.finalPB,base.finalProbability.finalPB);assert(Math.abs(r.finalPB-base.finalProbability.finalPB)<=.005+1e-12);
 assert.equal(JSON.stringify(r.extended),JSON.stringify(base.finalProbability.extended));assert.equal(JSON.stringify(r.physics),JSON.stringify(base.finalProbability.physics));assert.equal(r.smoothedPB,base.finalProbability.smoothedPB,'PR59 EMA changed');
 const expected=a.api.conditionalSettlement(r.finalPB,p.next_hand_forecast.probabilities.T);assert(Math.abs(p.ev_banker-expected.banker)<1e-12);assert(Math.abs(p.ev_player-expected.player)<1e-12);assert.equal(p.forecast_direction,p.ev_direction.direction);assert.equal(p.next_hand_forecast.directional.B,r.finalPB);
 const next=a.api.prepareCalibrationTicket(h,p);assert.equal(next.p,base.finalProbability.finalPB,'do not learn from already calibrated probability');assert(next.trial&&next.trial.proposal_p);assert(!a.api.commitCalibrationTicket(next,'P',h));h.push('P');assert(a.api.commitCalibrationTicket(next,'P',h));assert(!a.api.commitCalibrationTicket(next,'P',h));
 console.log(JSON.stringify({passed:true,formal_delta:p.browser_calibration.delta,brier_and_bet_gate:true,ev_display_same_probability:true,no_leakage:true,no_feature_or_ema_change:true}));
})().catch(e=>{console.error(e);process.exitCode=1});
