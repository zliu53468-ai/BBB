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

  // Accessible segmented tabs: all panel contents already exist in index.html.
  const tabButtons=Array.from(document.querySelectorAll("[data-quant-tab]"));
  const tabIds=new Set(["quantTabProbability","quantTabKelly","quantTabSimulation"]);
  function selectTab(id,focus=false){
    if(!tabIds.has(id))return;
    tabButtons.forEach(btn=>{
      const active=btn.dataset.quantTab===id;
      btn.classList.toggle("is-active",active);
      btn.setAttribute("aria-selected",String(active));
      btn.tabIndex=active?0:-1;
      if(focus&&active)btn.focus();
    });
    tabIds.forEach(panelId=>{
      const panel=$(panelId);if(!panel)return;
      const active=panelId===id;
      panel.classList.toggle("is-active",active);
      panel.setAttribute("aria-hidden",String(!active));
    });
  }
  tabButtons.forEach((button,i)=>{
    button.addEventListener("click",()=>selectTab(button.dataset.quantTab));
    button.addEventListener("keydown",event=>{
      if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
      event.preventDefault();
      const next=event.key==="Home"?0:event.key==="End"?tabButtons.length-1
        :(i+(event.key==="ArrowRight"?1:tabButtons.length-1))%tabButtons.length;
      selectTab(tabButtons[next].dataset.quantTab,true);
    });
  });
  document.querySelectorAll("[data-quant-target]").forEach(link=>{
    link.addEventListener("click",()=>{
      const id=link.dataset.quantTarget;
      if(tabIds.has(id))selectTab(id);
    });
  });
  if(location.hash==="#quantTabKelly")selectTab("quantTabKelly");
  if(location.hash==="#quantTabSimulation")selectTab("quantTabSimulation");
  window.addEventListener("hashchange",()=>{
    const chosen=location.hash.slice(1);
    if(tabIds.has(chosen))selectTab(chosen);
  });

  // Static KPI metrics are real, visible HTML values before JS starts.
  // Count-up is optional animation; reduced-motion users retain static figures.
  const prefersReducedMotion=window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  if(!prefersReducedMotion&&typeof requestAnimationFrame==="function"){
    document.querySelectorAll("[data-count-up]").forEach(el=>{
      const target=Number(el.dataset.countUp),suffix=el.dataset.suffix||"";
      if(!Number.isFinite(target)||target<0)return;
      const format=n=>Math.round(n).toLocaleString("en-US")+suffix;
      const start=performance.now();
      function frame(t){
        const progress=Math.min(1,(t-start)/850);
        const eased=1-Math.pow(1-progress,3);
        el.textContent=format(target*eased);
        if(progress<1)requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  // Hypothetical Kelly worksheet only: no model probabilities are imported.
  const kellyInputs=["quantBankroll","quantHypWin","quantHypPayout","quantKellyScale","quantRiskCap"].map($);
  function recalcKelly(){
    const [bankroll,winRate,payout,scale,cap]=kellyInputs.map(el=>Number(el?.value));
    const within=(v,min,max)=>Number.isFinite(v)&&v>=min&&v<=max;
    if(!within(bankroll,1,100000000)||!within(winRate,0,100)||
      !within(payout,.01,100)||!within(scale,0,100)||!within(cap,0,100)){
      set("quantHypEV","輸入有誤");set("quantHypFullKelly","—");set("quantHypStake","—");
      set("quantKellyStatus","請確認各欄位的有效範圍。");return;
    }
    const p=winRate/100,ev=p*payout-(1-p);
    const full=api.kellyFraction(p,1-p,payout);
    const allocated=Math.min(full*scale/100,cap/100);
    set("quantHypEV",pct(ev));set("quantHypFullKelly",pct(full));
    set("quantHypStake",pct(allocated)+" / "+Math.round(bankroll*allocated).toLocaleString("zh-TW")+" 元");
    set("quantKellyStatus","此試算完全以你自行假設的勝率計算，不是下一局預測或實際下注建議。");
  }
  kellyInputs.forEach(el=>el?.addEventListener("input",recalcKelly));
  recalcKelly();

  // Seeded Monte Carlo example: standardized random outcomes, not a backtest.
  function recalcSimulation(){
    const seed=Number($("quantSimSeed")?.value),stake=Number($("quantSimStake")?.value);
    if(!Number.isInteger(seed)||seed<0||seed>4294967295||
      !Number.isFinite(stake)||stake<1||stake>10000){
      set("quantSimStatus","隨機種子須為 0 至 4294967295 的整數，固定金額為 1 至 10000 元。");return;
    }
    const sim=api.simulateFlatBanker(seed,10000,stake,10000);
    set("quantSimEnd",Math.round(sim.endBalance).toLocaleString("zh-TW")+" 元");
    set("quantSimMDD",pct(sim.maxDrawdownPct));
    set("quantSimBets",sim.betsPlaced.toLocaleString("en-US")+" / 10,000");
    const line=$("quantSimLine"),baseline=$("quantSimBaseline");
    if(line&&baseline){
      const all=sim.samples;
      const low=Math.min(...all.map(x=>x.balance),10000);
      const high=Math.max(...all.map(x=>x.balance),10000);
      const range=Math.max(1,high-low);
      const y=v=>267-232*(v-low)/range;
      const pts=all.map((row,i)=>{
        const x=22+(938*i/(all.length-1));
        return x.toFixed(1)+","+y(row.balance).toFixed(1);
      });
      line.setAttribute("points",pts.join(" "));
      const baseY=y(10000).toFixed(1);
      baseline.setAttribute("y1",baseY);baseline.setAttribute("y2",baseY);
    }
    set("quantSimStatus","固定隨機種子 "+seed+"，採獨立標準機率抽樣，起始資金 10,000 元；這只是一條模擬路徑，不能代表策略績效。");
  }
  $("quantRunSim")?.addEventListener("click",recalcSimulation);
  recalcSimulation();

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

  function simulateFlatBanker(seed,initialBankroll=10000,stake=10,rounds=10000){
    if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw new RangeError("seed 需為合法的 32 位元整數。");
    if(!Number.isFinite(initialBankroll)||initialBankroll<=0||!Number.isFinite(stake)||stake<=0)
      throw new RangeError("模擬資金與固定金額均必須大於 0。");
    if(!Number.isInteger(rounds)||rounds<1||rounds>10000)throw new RangeError("模擬局數須介於 1 至 10,000。");
    let state=seed>>>0,balance=initialBankroll,peak=initialBankroll,maxDrawdownPct=0,betsPlaced=0;
    const samples=[{round:0,balance}];
    for(let round=1;round<=rounds;round++){
      state=(Math.imul(1664525,state)+1013904223)>>>0;
      const r=state/4294967296;
      if(balance>=stake){
        betsPlaced++;
        if(r<BASE.B)balance+=stake*.95;
        else if(r<BASE.B+BASE.P)balance-=stake;
        // On tie, banker wager pushes and balance is unchanged.
        peak=Math.max(peak,balance);
        maxDrawdownPct=Math.max(maxDrawdownPct,(peak-balance)/peak);
      }
      if(round%50===0||round===rounds)samples.push({round,balance});
    }
    return {seed,rounds,initialBankroll,stake,betsPlaced,endBalance:balance,maxDrawdownPct,samples};
  }
  return Object.freeze({BASE,parseHistory,theoreticalEV,kellyFraction,summarize,simulateFlatBanker});
});
