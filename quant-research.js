/* Independent educational probability calculator. No calls into BBB Soft Shoe,
 * licensed prediction, wager gates, or external game/advertising services.
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBQuantMath=api;
  if(typeof document==="undefined")return;
  const $=id=>document.getElementById(id);
  const input=$("quantHistory"),status=$("quantInputStatus");
  if(!input||!$("quantAddB")||!$("quantAddP")||!$("quantAddT"))return;

  const pct=v=>(v*100).toFixed(2)+"%";
  const pp=v=>(v>=0?"+":"")+(v*100).toFixed(2)+" pp";
  const set=(id,value)=>{const el=$(id);if(el)el.textContent=value;};
  let sequence=[];
  function redraw(){
    const data=api.summarize(sequence);
    input.value=sequence.join(" ");
    set("quantTotal",String(data.total));
    ["B","P","T"].forEach(k=>{
      const result=data.items[k];
      set("quantCount"+k,String(result.count));
      set("quantFreq"+k,data.total?pct(result.frequency):"—");
      set("quantDeviation"+k,data.total?pp(result.deviation):"—");
      const bar=$("quantBar"+k);if(bar)bar.style.width=data.total?pct(result.frequency):"0%";
    });
    set("quantEvB",pct(data.ev.B));
    set("quantEvP",pct(data.ev.P));
    set("quantEvT",pct(data.ev.T));
    set("quantKelly",pct(data.kelly.B));
    set("quantKellyP",pct(data.kelly.P));
    set("quantKellyT",pct(data.kelly.T));
    set("quantRecent",sequence.length?sequence.slice(-18).join(" · "):"尚無紀錄");
    set("quantRiskSignal","0.00%（標準規則的理論 EV 為負）");
  }
  const msg=(s,invalid=false)=>{
    if(!status)return;
    status.textContent=s;
    status.style.color=invalid?"#FDA4AF":"#94A3B8";
  };
  function commit(raw){
    try{
      sequence=api.parseHistory(raw);
      redraw();
      msg("已更新 "+sequence.length+" 局歷史紀錄。歷史頻率不會改變下局理論機率。");
    }catch(e){msg(e.message,true);}
  }
  for(const k of ["B","P","T"]){
    $("quantAdd"+k).addEventListener("click",()=>{
      if(sequence.length>=2000){msg("為避免頁面負荷過高，最多允許 2,000 局。",true);return;}
      sequence.push(k);redraw();msg("已加入 "+k+"，共 "+sequence.length+" 局。");
    });
  }
  $("quantApply")?.addEventListener("click",()=>commit(input.value));
  $("quantUndo")?.addEventListener("click",()=>{
    sequence.pop();redraw();msg("已撤銷最後一局。");
  });
  $("quantClear")?.addEventListener("click",()=>{
    sequence=[];redraw();msg("歷史紀錄已清空；本工具不會傳送紀錄至伺服器。");
  });
  input.addEventListener("keydown",e=>{
    if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){e.preventDefault();commit(input.value);}
  });
  redraw();
})(typeof window!=="undefined"?window:null,function(){
  "use strict";
  /* Eight-deck standard baccarat, approximation, before any table-specific
   * commission changes; tie pushes B/P bets, and T is paid 8 to 1.
   * Sources/assumptions must be revalidated for any different ruleset.
   */
  const BASE=Object.freeze({B:0.458597,P:0.446247,T:0.095156});
  const BASE_KEYS=["B","P","T"];
  function parseHistory(raw){
    if(typeof raw!=="string")throw new TypeError("歷史紀錄必須是文字。");
    const clean=raw.replace(/[\s,，、;；|/／·.-]+/g,"").toUpperCase();
    if(clean.length>2000)throw new RangeError("最多可記錄 2,000 局。");
    if(/[^BPT莊閒和]/u.test(clean))throw new Error("只接受 B、P、T 或 莊、閒、和；請移除其他字元。");
    const map={莊:"B",閒:"P",和:"T"};
    return Array.from(clean,ch=>map[ch]||ch);
  }
  function theoreticalEV(){
    return {B:0.95*BASE.B-BASE.P,P:BASE.P-BASE.B,T:8*BASE.T-(BASE.B+BASE.P)};
  }
  function kellyFraction(pWin,pLose,payout){
    if(![pWin,pLose,payout].every(Number.isFinite)||pWin<0||pLose<0||payout<=0)return 0;
    const denom=payout*(pWin+pLose);
    if(denom<=0)return 0;
    return Math.min(1,Math.max(0,(payout*pWin-pLose)/denom));
  }
  function summarize(sequence){
    if(!Array.isArray(sequence)||!sequence.every(v=>BASE_KEYS.includes(v)))throw new TypeError("序列只能包含 B/P/T。");
    const total=sequence.length,counts={B:0,P:0,T:0};
    for(const v of sequence)counts[v]++;
    const items={};
    for(const k of BASE_KEYS){
      const frequency=total?counts[k]/total:0;
      items[k]={count:counts[k],frequency,deviation:total?frequency-BASE[k]:0,baseline:BASE[k]};
    }
    return {
      total,items,ev:theoreticalEV(),
      kelly:{
        B:kellyFraction(BASE.B,BASE.P,0.95),
        P:kellyFraction(BASE.P,BASE.B,1),
        T:kellyFraction(BASE.T,BASE.B+BASE.P,8)
      }
    };
  }
  return Object.freeze({BASE,parseHistory,theoreticalEV,kellyFraction,summarize});
});
