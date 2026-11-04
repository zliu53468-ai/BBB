/** Licensed panel for the local-card pipeline. No old prediction scripts load. */
(function(root){
 'use strict';
 const $=id=>document.getElementById(id),KEY='bbb_pure_card_audit_v1';
 let worker=null,timer=null,request=0,pending=false,last=null,rows=[];
 try{const v=JSON.parse(root.localStorage.getItem(KEY)||'[]');if(Array.isArray(v))rows=v.slice(-500);}catch(_){}
 const text=(id,value)=>{const e=$(id);if(e)e.textContent=value;};
 const pct=v=>Number.isFinite(v)?(100*v).toFixed(4)+'%':'—';
 function skip(message){text('directionText','觀望');text('directionCode','SKIP');
  text('regime','未確認正 EV');text('message',message);text('confidence','—');text('strength','—');
  const orb=$('directionOrb');if(orb)orb.className='direction-orb idle';}
 function finish(){pending=false;clearTimeout(timer);if($('btnPointAnalyze'))$('btnPointAnalyze').disabled=false;}
 function stop(){request++;if(worker)worker.terminate();worker=null;finish();}
 function fail(reason){stop();last=null;skip('本次計算未完成，觀望：'+reason);text('pointInputStatus','請確認牌面後重新分析。');}
 function save(next){root.localStorage.setItem(KEY,JSON.stringify(next.slice(-500)));rows=next.slice(-500);}
 function parse(s){const a=String(s).trim().toUpperCase().split(/[+＋,，\s]+/);
  if(a.length<2||a.length>3)throw Error('每邊請輸入 2～3 張牌，如 J+5+2，不是最終總點');
  return a.map(v=>{if(/^(0|10|J|Q|K)$/.test(v))return 0;if(v==='A')return 1;if(/^[1-9]$/.test(v))return +v;throw Error('牌面格式不正確');});}
 const reasons={direction_disagreement:'三方方向不一致',ev_uplift_below_threshold:'EV 提升未達門檻',mc_uncertainty_above_limit:'MC 不確定性超過門檻'};
 function render(r){const d=r.decision,side=d.direction;
  text('directionText',side==='SKIP'?'觀望':side==='B'?'莊':'閒');text('directionCode',side==='SKIP'?'SKIP':'EV COMPARISON');
  const orb=$('directionOrb');if(orb)orb.className='direction-orb '+(side==='SKIP'?'idle':side==='B'?'banker':'player');
  text('confidence','相對基準 EV 提升 '+pct(d.uplift));
  text('strength','莊 '+pct(r.theory.ev.B)+'｜閒 '+pct(r.theory.ev.P));
  text('regime','單局獨立｜未驗證下注優勢');
  const label=s=>s==='B'?'莊':s==='P'?'閒':'無方向';
  text('message','EOR '+label(r.eor.direction)+' → PF '+r.pf.count+' → 理論 '+label(d.theoretical_direction)+
   ' → Deplete MC '+label(d.mc_direction)+' → '+(d.reasons.map(x=>reasons[x]||x).join('、')||'方向條件通過')+
   '｜EV 門檻 '+pct(d.threshold)+'｜MC 標準誤 '+pct(d.mc_se)+'（上限 0.1500%）'+
   (r.diagnostic.opposed?'｜20 局旁路對沖，門檻已提高':'')+
   '｜自訂 EOR；假想局部牌靴，並非真實整靴；未證明勝率提升。');
  text('localCalibrationStatus','牌組假設：八副牌基準減本局已知牌＋高斯壓力情境；歷史不進入 PF。');
 }
 function start(){
  if(pending)return;
  stop();skip('計算中，先觀望。');last=null;
  let player,banker;try{player=parse($('pointPlayerFinal').value);banker=parse($('pointBankerFinal').value);}catch(e){fail(e.message);return;}
  if(typeof root.Worker!=='function'){fail('瀏覽器不支援背景計算');return;}
  let settled=rows.filter(r=>r.actual).slice(-20),previous=null;
  const continuous=$('pureContinuous')?.checked===true;
  if(continuous&&rows.length&&!rows[rows.length-1].actual)previous=rows[rows.length-1];
  const sum=a=>a.reduce((x,y)=>x+y,0)%10,pt=sum(player),bt=sum(banker),actual=bt>pt?'B':pt>bt?'P':'T';
  if(previous)settled=[...settled,{eor_direction:previous.eor_direction,actual}].slice(-20);
  pending=true;$('btnPointAnalyze').disabled=true;const id=++request;
  text('pointInputStatus','背景計算中：5,000 粒子＋10,000 次模擬，請稍候。');
  try{
   worker=new root.Worker('pure_card_worker.js?v=20261105-local-card-v1');
   worker.onerror=()=>fail('背景程式無法載入或執行');worker.onmessage=event=>{
    if(event.data.id!==request)return;
    if(event.data.error){fail(event.data.error);return;}
    const r=event.data.report;
    if(!r||r.version!=='BBB_LOCAL_CARD_EOR_PF5000_MC10000_V1'){fail('模型版本不一致');return;}
    try{
     const next=rows.map(x=>({...x}));
     if(previous)next[next.length-1].actual=actual;
     // Unconfirmed gaps do not become fabricated settlement labels.
     next.push({id:Date.now()+'-'+id,player,banker,eor_direction:r.eor.direction,
      direction:r.decision.direction,features:r.observation.features,
      version:r.version,eor_profile:r.eor.profile,actual:null,report:r});
     save(next);last=r;render(r);finish();worker.terminate();worker=null;
     if($('pureContinuous'))$('pureContinuous').checked=false;
     text('pointInputStatus','已完成；下一筆仍須輸入逐張牌。只有確認連續下一局時才勾選結算。');
    }catch(e){fail('本機紀錄儲存失敗');}
   };
   timer=setTimeout(()=>fail('計算逾時'),120000);
   worker.postMessage({id,player,banker,settledRows:settled});
  }catch(e){fail('背景計算啟動失敗');}
 }
 function intercept(id,fn){const e=$(id);if(e)e.addEventListener('click',event=>{event.preventDefault();event.stopImmediatePropagation();fn();},true);}
 function download(){try{const blob=new Blob([JSON.stringify({version:1,rows},null,2)],{type:'application/json'}),a=document.createElement('a'),url=URL.createObjectURL(blob);
  a.href=url;a.download='BBB-local-card-audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){text('pointInputStatus','匯出失敗');}}
 for(const id of ['btnB','btnP','btnT','btnStart','btnNoSkipTest'])if($(id))$(id).hidden=true;
 document.querySelectorAll('.bbb-results-block').forEach(e=>e.hidden=true);
 const gap=$('forecastGap');if(gap){gap.hidden=true;const label=document.querySelector('label[for="forecastGap"]');if(label)label.hidden=true;}
 if($('drawLengthDiagnostics'))$('drawLengthDiagnostics').hidden=true;
 const hint=document.createElement('label');hint.style.cssText='display:block;line-height:1.7;margin-top:10px';
 const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.id='pureContinuous';hint.appendChild(checkbox);
 hint.appendChild(document.createTextNode(' 本筆是上一筆預測的連續下一局（勾選才結算旁路；漏局／換桌勿勾選）'));
 $('bbbPointEntry')?.appendChild(hint);
 intercept('btnPointAnalyze',start);
 intercept('btnBack',()=>{stop();last=null;try{save([]);skip('連續診斷與本機新模型紀錄已重設。');}catch(e){fail('無法清除紀錄');}});
 text('btnBack','重設連續診斷');
 // Preserve the license client's authenticated end/reset handler.
 $('btnEnd')?.addEventListener('click',()=>{stop();last=null;skip('結束分析，觀望。');},true);
 for(const id of ['btnExportTraining','btnExportObservations'])intercept(id,download);
 for(const id of ['pointPlayerFinal','pointBankerFinal'])$(id)?.addEventListener('keydown',e=>{
  const mobile=root.matchMedia?.('(pointer: coarse)').matches;
  if(e.key==='Enter'){e.preventDefault();if(!mobile)start();}
 });
 text('modePill','EOR → PF5000 → 組成引擎 → MC10000 → V4');
 skip('請輸入當局每張牌面（各方 2～3 張），不是最終總點。');
 text('localCalibrationStatus','每局獨立；Markov 僅收緊門檻；LinUCB／上下文不參與。');
 root.__BBB_PURE_CARD_UI__={start,reset:()=>{stop();last=null;},get latestReport(){return last;}};
})(window);
