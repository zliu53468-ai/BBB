/* 新版獨立預測轉接；不改Core ABI，不沿用跨靴記憶。 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.__BBB_QUANT_DYNAMIC__=api;
})(typeof window!=='undefined'?window:null,function(){
 'use strict';
 function predict(Engine,history,config={}){
  if(typeof Engine!=='function'||!Array.isArray(history)||!history.every(x=>['B','P','T'].includes(x)))throw Error('quant_engine_or_history_invalid');
  const engine=Engine.fromHistory(history,config),quant=engine.predictNext(),q=quant.probabilities,mass=q.B+q.P,pb=q.B/mass;
  const forecast={available:true,source:'quant_dynamic_v2',direction:quant.direction,probabilities:q,directional:{B:pb,P:1-pb},uncertainty:null,weak_signal:quant.direction==='Skip',quant_dynamic:true};
  return {direction:quant.direction,bet_direction:quant.direction,forecast_direction:quant.direction,confidence:0,probabilities:forecast.directional,ev_banker:quant.ev_banker,ev_player:quant.ev_player,bet_fraction:quant.bet_fraction,stake_multiplier:quant.bet_fraction,entry_tier:quant.direction==='Skip'?'skip':'experimental',quant_dynamic:quant,raw_next_hand_forecast:forecast,next_hand_forecast:forecast,browser_calibration:null,finalProbability:{version:quant.version,active:true,mode:'quant_dynamic',primary_source:forecast.source,primary_candidate:quant.direction,quant_dynamic:quant,finalPB:pb,physics_raw_p_b:null,physics_smoothed_p_b:null,physics:null,extended:null,original7:null,p_tie:q.T,ev_banker:quant.ev_banker,ev_player:quant.ev_player,finalDirection:quant.direction,execution_order:['expected_depletion_replay','dynamic_count_weight','ev_005_and_count_2_gate','quarter_kelly_clip'],xgb_influence:0,core_influence:0,new_128d_injected:false}};
 }
 function render(document,quant,testing){
  const set=(id,text)=>{const e=document.getElementById(id);if(e)e.textContent=text;};
  set('modePill','動態算牌實驗版');
  set('localCalibrationStatus','B/P/T期望扣牌估計；非真實殘牌、非完整重訓');
  set('regime',testing?'測試方向，不列正式下注':quant.direction==='Skip'?'觀望：TC或EV未達門檻':('實驗建議：'+(quant.direction==='B'?'莊':'閒')+'；資金比例 '+(quant.bet_fraction*100).toFixed(2)+'%'));
  set('message','下一局實驗估計｜TC '+quant.estimated_shoe.trueCount.toFixed(2)+'｜算牌權重 '+(quant.w_count*100).toFixed(1)+'%｜莊EV '+(quant.ev_banker*100).toFixed(2)+'%／閒EV '+(quant.ev_player*100).toFixed(2)+'%；須 |TC|≥2 且 EV>0.5%。未驗證勝率；5%下限可能高於原¼ Kelly。');
  for(const [key,id] of [['banker_pair','sideBankerPair'],['player_pair','sidePlayerPair'],['tie','sideTie'],['lucky_six','sideLuckySix']]){
   const s=quant.side_bets[key];set(id,s.bet_allowed?'建議（正EV估計）':'不建議・觀望');
  }
  set('sideBetNote','逐局指標：對子 '+quant.estimated_shoe.pairCount.toFixed(2)+'／幸運六 '+quant.estimated_shoe.l6Index.toFixed(2)+'。B/P/T不能辨識對子；無校準邊注機率及賠率時維持觀望。');
 }
 return {predict,render};
});
