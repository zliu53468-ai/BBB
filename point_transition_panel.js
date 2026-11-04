/** 總點數介面：先持久化成功才顯示新方向，模型不受重繪或重新整理重複訓練。 */
(function(root){
 'use strict';
 const $=id=>document.getElementById(id),KEY='bgs_point_transition_history_v1';
 const api=root.BBBPointTransitions;
 if(!api)throw Error('點數模型載入失敗');
 const {PointTransitionPredictor,VERSION}=api;
 let model=null,busy=false,failed=false,epoch=0;
 const text=(id,value)=>{if($(id))$(id).textContent=value;};
 const pct=n=>Number.isFinite(n)?(n*100).toFixed(2)+'%':'—';
 function observeOnly(message){text('directionText','觀望');text('directionCode','SKIP');
  text('confidence','—');text('strength','—');text('regime','未產生方向');text('message',message);
  if($('directionOrb'))$('directionOrb').className='direction-orb idle';}
 function render(r){
  if(!r){observeOnly('請輸入閒、莊最終總點數（各 0～9），歷史會持續累積。');return;}
  const skip=r.direction==='SKIP';
  text('directionText',skip?'觀望':r.direction==='B'?'莊':'閒');
  text('directionCode',skip?'SKIP':r.relaxed?'ADAPTIVE SIGNAL':'TRANSITION SIGNAL');
  text('confidence','莊閒機率差 '+pct(r.gap));
  text('strength',r.sampleSize<api.DEFAULTS.minSamples?'樣本不足，先累積紀錄':
   '莊 '+pct(r.probability.B)+'｜閒 '+pct(r.probability.P)+'｜和 '+pct(r.probability.T));
  text('regime',r.relaxed?'門檻放寬 25%（探索訊號）':'標準統計門檻');
  const reason=r.reason==='insufficient_samples'?'此點數組合後續樣本不足':r.reason==='gap_below_threshold'?'機率差未超過動態門檻':'點數轉移方向通過門檻';
  text('message','目前狀態：閒 '+r.state.player+'／莊 '+r.state.banker+
   '｜同狀態後續樣本 '+r.sampleSize+'（非和局 '+r.decisiveSamples+'）'+
   '｜掃描 '+(r.scanDepth===0?'全部累積紀錄':r.scanDepth+' 局')+
   '｜動態門檻 '+pct(r.threshold)+'｜'+reason+
   (r.relaxed?'｜已擴大窗口並降低門檻；此為探索，不代表顯著性或勝率保證':'')+
   '｜歷史相關不代表下一局必然重複。');
  if($('directionOrb'))$('directionOrb').className='direction-orb '+(skip?'idle':r.direction==='B'?'banker':'player');
  text('localCalibrationStatus','已累積 '+model.totalRounds+' 局；精確狀態樣本至少 '+api.DEFAULTS.minSamples+
   '、非和局至少 '+api.DEFAULTS.minDecisive+'；近 6 局訊號 '+r.recentSignals+' 次，目標 1～2 次，不強制出手。');
 }
 async function persist(candidate){
  root.localStorage.setItem(KEY,JSON.stringify(candidate.snapshot()));
  // bgs_ 前綴進入既有會員隔離 IndexedDB；等待非同步寫入落地。
  const storage=root.__BBB_LICENSE_STORAGE__;
  if(storage){
   const started=Date.now();
   for(;;){const s=storage.status();if(s.error)throw Error(s.error);if(!s.active)throw Error('會員儲存空間未就緒');
    if(!s.pending_writes)break;if(Date.now()-started>5000)throw Error('歷史儲存逾時');
    await new Promise(resolve=>setTimeout(resolve,20));}
  }
 }
 async function change(fn){
  if(busy||failed)return;busy=true;const id=epoch;
  if($('btnPointAnalyze'))$('btnPointAnalyze').disabled=true;
  observeOnly('正在更新與儲存總點數歷史…');
  try{
   // 交易副本：非法輸入或儲存失敗，不修改目前已確認模型。
   const candidate=new PointTransitionPredictor(model.snapshot());const result=fn(candidate);
   await persist(candidate);if(id!==epoch)return;
   model=candidate;render(result);
   text('pointInputStatus','歷史已儲存；輸入下一局總點數可持續更新轉移矩陣。');
  }catch(e){observeOnly('本次未完成，觀望：'+String(e.message||e));text('pointInputStatus','未顯示新方向，請檢查輸入或儲存空間。');}
  finally{if(id===epoch){busy=false;if($('btnPointAnalyze'))$('btnPointAnalyze').disabled=false;}}
 }
 function total(id){const s=String($(id)?.value??'').trim();if(!/^[0-9]$/.test(s))throw Error('只接受單一 0～9 整數總點；不輸入逐張牌');return Number(s);}
 async function submit(){
  if(busy||failed)return;
  let p,b;try{p=total('pointPlayerFinal');b=total('pointBankerFinal');}
  catch(e){observeOnly(e.message);return;}
  await change(m=>m.observe(p,b));
 }
 function bind(id,fn){$(id)?.addEventListener('click',e=>{e.preventDefault();e.stopImmediatePropagation();fn();},true);}
 function download(){try{
  const value=root.localStorage.getItem(KEY);if(!value)throw Error('尚無總點數歷史');
  const blob=new Blob([value],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='BBB-point-transitions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }catch(e){text('pointInputStatus','匯出失敗：'+e.message);}}
 for(const id of ['btnB','btnP','btnT','btnStart','btnNoSkipTest'])if($(id))$(id).hidden=true;
 document.querySelectorAll('.bbb-results-block').forEach(e=>e.hidden=true);
 if($('forecastGap'))$('forecastGap').hidden=true;
 const label=document.querySelector('label[for="forecastGap"]');if(label)label.hidden=true;
 if($('drawLengthDiagnostics'))$('drawLengthDiagnostics').hidden=true;
 bind('btnPointAnalyze',submit);
 bind('btnBack',()=>change(m=>m.undo()));
 bind('btnNewSequence',()=>change(m=>{m.startNewSequence();return null;}));
 for(const id of ['btnExportTraining','btnExportObservations'])bind(id,download);
 // 保留授權系統明示的清除學習／結束分析事件，不阻擋其遠端同步。
 $('btnEnd')?.addEventListener('click',()=>{epoch++;busy=false;observeOnly('正在清除本機學習並結束分析…');},true);
 for(const id of ['pointPlayerFinal','pointBankerFinal'])$(id)?.addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();if(!root.matchMedia?.('(pointer: coarse)').matches)submit();}
 });
 text('modePill','總點數轉移矩陣 · 自適應觀望');
 try{
  const raw=root.localStorage.getItem(KEY);model=new PointTransitionPredictor(raw?JSON.parse(raw):null);
  render(model.nextIsBreak?null:model.latest());
  text('pointInputStatus',raw?'已恢復累積歷史；不會因重新整理而重設。':'僅輸入閒、莊最終總點，各 0～9。');
 }catch(e){failed=true;observeOnly('歷史載入異常，觀望：'+e.message);text('pointInputStatus','不會自動覆寫舊歷史；可先匯出檢查，或明示清除學習。');}
 root.__BBB_POINT_TRANSITION_UI__={submit,get snapshot(){return model?.snapshot()||null;},version:VERSION};
})(window);
