(() => {
  "use strict";
  const SHOE_DIM = 128, ROAD_DIM = 128, DIM = 256;
  const STORAGE_KEY = "bgs256d_stability_v12_user_panel_v1";
  const LEGACY_KEYS = [
    "bgs256d_128plus128_frozen_direct_tech_panel_v1",
    "bgs128d_64plus64_frozen_direct_tech_panel_v1",
    "bgs64d_32plus32_frozen_direct_tech_panel_v2",
    "bgs32d_frozen_direct_tech_panel_v1"
  ];
  const DECKS = 8, TOTAL_CARDS = 52 * DECKS, AVG_CARDS_PER_HAND = 4.9;
  const el = id => document.getElementById(id);
  const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
  const bp = seq => seq.filter(x => x === "B" || x === "P");
  let state = { history: [], active: false, lastPrediction: null };

  function runs(seq){
    const values = bp(seq);
    if(!values.length) return [];
    const out=[]; let side=values[0], length=1;
    for(let i=1;i<values.length;i++){
      if(values[i]===side) length++;
      else { out.push([side,length]); side=values[i]; length=1; }
    }
    out.push([side,length]); return out;
  }
  function bankerRatio(seq,w){ const a=bp(seq).slice(-Math.max(1,w)); return a.length ? a.filter(x=>x==="B").length/a.length : .5; }
  function turnRate(seq,w){ const a=bp(seq).slice(-Math.max(2,w)); if(a.length<2) return .5; let t=0; for(let i=1;i<a.length;i++) if(a[i]!==a[i-1]) t++; return t/(a.length-1); }
  function tieRatio(seq,w=0){ const a=w?seq.slice(-w):seq; return a.length ? a.filter(x=>x==="T").length/a.length : 0; }
  function binaryEntropy(seq,w=12){ const a=bp(seq).slice(-w); if(!a.length) return 1; const p=a.filter(x=>x==="B").length/a.length,q=1-p; let e=0; if(p)e-=p*Math.log2(p); if(q)e-=q*Math.log2(q); return clip(e); }
  function outcomeEntropy(seq,w=12){ const a=seq.slice(-w); if(!a.length)return 1; let e=0; for(const o of["B","P","T"]){ const p=a.filter(x=>x===o).length/a.length; if(p)e-=p*Math.log2(p); } return clip(e/Math.log2(3)); }
  function balance(seq,w){ return clip(1-Math.abs(bankerRatio(seq,w)-.5)*2); }
  function sameTail(seq,w){ const a=bp(seq).slice(-w); if(a.length<w)return .5; return a.every(x=>x===a[0])?1:0; }
  function alternatingTail(seq,w){ const a=bp(seq).slice(-w); if(a.length<w)return .5; for(let i=1;i<a.length;i++) if(a[i]===a[i-1]) return 0; return 1; }
  function runVolatility(seq,w=6){ const h=runs(seq).slice(-w).map(x=>x[1]); if(h.length<2)return .25; let d=0; for(let i=1;i<h.length;i++) d+=Math.abs(h[i]-h[i-1]); return clip(d/(h.length-1)/3); }
  function runTrend(seq,w=5){ const h=runs(seq).slice(-w).map(x=>x[1]); if(h.length<2)return .5; return clip(.5+((h.at(-1)-h[0])/(h.length-1))/6); }
  function runStats(seq,w){ const h=runs(seq).slice(-w).map(x=>x[1]); if(!h.length)return{avg:0,max:0,std:0}; const m=h.reduce((a,b)=>a+b,0)/h.length; const v=h.reduce((s,x)=>s+(x-m)**2,0)/h.length; return{avg:clip(m/8),max:clip(Math.max(...h)/12),std:clip(Math.sqrt(v)/6)}; }

  function hazard(seq){
    const rs=runs(seq); if(!rs.length)return .5;
    const current=rs.at(-1)[1], completed=rs.slice(0,-1).map(x=>x[1]);
    if(!completed.length) return clip(.38+current*.045,.35,.68);
    let reached=0,turned=0;
    for(const length of completed){ if(length>=current){ reached++; if(length===current)turned++; } }
    const empirical=(turned+2.5)/(reached+5);
    const globalAvg=completed.reduce((a,b)=>a+b,0)/completed.length;
    const shape=clip(.5+(current-globalAvg)/8,.25,.75);
    return clip(.72*empirical+.28*shape,.18,.82);
  }

  function hsmmStable(seq){
    const tr=turnRate(seq,10), rs=runs(seq), current=rs.length?rs.at(-1)[1]:0;
    const runNorm=clip(current/6), ent=binaryEntropy(seq,12), vol=runVolatility(seq,6);
    const trendLike=Math.exp(-((((tr-.24)/.25)**2)+(((runNorm-.66)/.34)**2)+(((vol-.24)/.28)**2)));
    const jumpLike=Math.exp(-((((tr-.84)/.20)**2)+(((runNorm-.18)/.24)**2)+(((vol-.30)/.30)**2)));
    const noisy=Math.exp(-((((tr-.53)/.22)**2)+(((ent-.98)/.18)**2)+(((vol-.62)/.28)**2)));
    const stable=trendLike+jumpLike;
    return clip(stable/(stable+noisy+.35));
  }

  function derivedMark(heights,column,row,newColumn,offset){
    if(newColumn){ if(column<offset+1)return ""; return heights[column-1]===heights[column-1-offset]?"R":"U"; }
    if(column<offset)return "";
    const reference=heights[column-offset];
    return (reference>=row)===(reference>=row-1)?"R":"U";
  }
  function buildDerivedRoads(seq){
    const values=bp(seq), sides=[], heights=[], out={big_eye:[],small_road:[],cockroach_road:[]};
    const offsets={big_eye:1,small_road:2,cockroach_road:3};
    for(const side of values){
      const newColumn=!sides.length||side!==sides.at(-1);
      if(newColumn){ sides.push(side); heights.push(1); } else heights[heights.length-1]++;
      const column=heights.length-1,row=heights[column];
      for(const [name,offset] of Object.entries(offsets)){
        const mark=derivedMark(heights,column,row,newColumn,offset);
        if(mark) out[name].push(mark);
      }
    }
    return out;
  }
  function regularity(values,w=8){ const a=values.slice(-w).filter(x=>x==="R"||x==="U"); return a.length?[a.filter(x=>x==="R").length/a.length,a.length]:[.5,0]; }
  function derivedInfo(seq,w=8){
    const roads=buildDerivedRoads(seq);
    const [be,bn]=regularity(roads.big_eye,w),[sm,sn]=regularity(roads.small_road,w),[cr,cn]=regularity(roads.cockroach_road,w);
    const mean=(be+sm+cr)/3;
    const consensus=clip(1-(Math.abs(be-mean)+Math.abs(sm-mean)+Math.abs(cr-mean))/1.5);
    return {be,sm,cr,bn,sn,cn,consensus,support:clip((bn+sn+cn)/(w*3)),roads};
  }

  function buildShoeVector(seq){
    const used=Math.min(TOTAL_CARDS,seq.length*AVG_CARDS_PER_HAND), remaining=Math.max(0,TOTAL_CARDS-used);
    const rr=clip(remaining/TOTAL_CARDS), pen=clip(1-rr), maturity=clip(seq.length/70), handsNorm=clip(seq.length/(TOTAL_CARDS/AVG_CARDS_PER_HAND));
    const v=[
      rr,pen,maturity,handsNorm,tieRatio(seq),balance(seq,Math.max(1,bp(seq).length)),binaryEntropy(seq,12),outcomeEntropy(seq,12),
      clip(1-Math.abs(pen-.125)/.125),clip(1-Math.abs(pen-.375)/.125),clip(1-Math.abs(pen-.625)/.125),clip(1-Math.abs(pen-.875)/.125),
      clip(seq.length/8),clip(seq.length/16),clip(seq.length/32),clip(seq.length/64)
    ];
    for(const w of [2,4,6,8,12,16,24,32,48,64]) v.push(tieRatio(seq,w),binaryEntropy(seq,w),outcomeEntropy(seq,w),balance(seq,w));
    for(const w of [3,5,7,10,14,20,28,40,56]) v.push(tieRatio(seq,w),binaryEntropy(seq,w),outcomeEntropy(seq,w),balance(seq,w));
    v.push(
      pen**2,pen**3,Math.sqrt(pen),Math.sqrt(Math.sqrt(pen)),rr**2,rr**3,Math.sqrt(rr),Math.sqrt(Math.sqrt(rr)),
      maturity**2,Math.sqrt(maturity),clip(Math.log1p(seq.length)/Math.log1p(128)),1
    );
    v.push(
      clip(.5+(tieRatio(seq,4)-tieRatio(seq,16))/2),clip(.5+(tieRatio(seq,8)-tieRatio(seq,32))/2),clip(.5+(tieRatio(seq,16)-tieRatio(seq,64))/2),
      clip(.5+(binaryEntropy(seq,4)-binaryEntropy(seq,16))/2),clip(.5+(binaryEntropy(seq,8)-binaryEntropy(seq,32))/2),clip(.5+(binaryEntropy(seq,16)-binaryEntropy(seq,64))/2),
      clip(.5+(outcomeEntropy(seq,4)-outcomeEntropy(seq,16))/2),clip(.5+(outcomeEntropy(seq,8)-outcomeEntropy(seq,32))/2),clip(.5+(outcomeEntropy(seq,16)-outcomeEntropy(seq,64))/2),
      clip(.5+(balance(seq,4)-balance(seq,16))/2),clip(.5+(balance(seq,8)-balance(seq,32))/2),clip(.5+(balance(seq,16)-balance(seq,64))/2)
    );
    v.push(
      clip(1-pen/.30),clip(1-Math.abs(pen-.45)/.30),clip((pen-.55)/.35),clip(1-pen/.16),
      clip(1-Math.abs(pen-.30)/.20),clip(1-Math.abs(pen-.62)/.24),clip((pen-.76)/.20),
      clip(seq.length/4),clip(seq.length/12),clip(seq.length/20),clip(seq.length/28),clip(seq.length/40)
    );
    if(v.length!==SHOE_DIM) throw new Error(`shoe vector mismatch: ${v.length}`);
    return v;
  }

  function buildRoadVector(seq){
    const rs=runs(seq), current=rs.length?rs.at(-1):["",0], prior=i=>rs.length>i?rs.at(-1-i):["",0];
    const d8=derivedInfo(seq,8), hz=hazard(seq), hs=hsmmStable(seq);
    const s4=runStats(seq,4),s6=runStats(seq,6),s8=runStats(seq,8),s12=runStats(seq,12);
    const runDelta=rs.length>1?clip(.5+(current[1]-prior(1)[1])/12):.5;
    const v=[
      current[0]==="B"?1:current[0]==="P"?0:.5,current[0]==="P"?1:current[0]==="B"?0:.5,clip(current[1]/8),
      ...[1,2,3,4,5,6].map(i=>clip(prior(i)[1]/8)),
      hz,clip(1-hz),hs,runVolatility(seq,6),runTrend(seq,5),
      d8.be,d8.sm,d8.cr,d8.consensus,d8.support,clip((d8.be+d8.sm+d8.cr)/3),
      sameTail(seq,2),sameTail(seq,3),alternatingTail(seq,4),alternatingTail(seq,6)
    ];
    for(const w of [2,4,6,8,10,12,16,20,24,32,48,64]) v.push(bankerRatio(seq,w));
    for(const w of [2,4,6,8,10,12,16,20,24,32,48,64]) v.push(turnRate(seq,w));
    for(const w of [4,6,8,12,16,24,32]){ const d=derivedInfo(seq,w); v.push(d.be,d.sm,d.cr); }
    v.push(s4.avg,s6.avg,s8.avg,s12.avg,s4.max,s8.max,s12.max,s4.std,s8.std,s12.std,runTrend(seq,8),runTrend(seq,12),runDelta);
    v.push(alternatingTail(seq,3),alternatingTail(seq,5),alternatingTail(seq,8),alternatingTail(seq,10),sameTail(seq,4),sameTail(seq,5),sameTail(seq,6),sameTail(seq,7),sameTail(seq,8),sameTail(seq,10));
    v.push(
      clip(.5+(bankerRatio(seq,4)-bankerRatio(seq,16))/2),clip(.5+(bankerRatio(seq,8)-bankerRatio(seq,32))/2),clip(.5+(bankerRatio(seq,16)-bankerRatio(seq,64))/2),
      clip(.5+(turnRate(seq,4)-turnRate(seq,16))/2),clip(.5+(turnRate(seq,8)-turnRate(seq,32))/2),clip(.5+(turnRate(seq,16)-turnRate(seq,64))/2)
    );
    v.push(hz**2,(1-hz)**2,hs**2,d8.consensus**2);
    for(const w of [3,5,7,9,11,14,18,22,28,36,40,56,72]) v.push(bankerRatio(seq,w));
    for(const w of [3,5,7,9,11,14,18,22,28,36,40,56,72]) v.push(turnRate(seq,w));
    if(v.length!==ROAD_DIM) throw new Error(`road vector mismatch: ${v.length}`);
    return v;
  }

  function context256(seq){ const shoe=buildShoeVector(seq),road=buildRoadVector(seq),vector=[...shoe,...road]; if(vector.length!==DIM)throw new Error(`context mismatch: ${vector.length}`); return vector; }

  function pairPattern(seq){
    const rs=runs(seq); if(rs.length<4)return{reliability:0,direction:0};
    const completed=rs.slice(0,-1).slice(-5).map(x=>x[1]);
    const support=completed.filter(x=>x===2).length/Math.max(1,completed.length);
    const current=rs.at(-1), side=current[0]==="B"?1:-1;
    return {reliability:clip((support-.35)/.55),direction:current[1]<=1?side:-side};
  }

  function deterministicDirection(seq){
    let hash=0; const token="BGS256_STABILITY|"+seq.join("");
    for(let i=0;i<token.length;i++) hash=(hash*31+token.charCodeAt(i))>>>0;
    return hash%2?"B":"P";
  }

  function analyzeStable(seq){
    const values=bp(seq),rs=runs(seq),current=rs.length?rs.at(-1):["",0],side=current[0]==="B"?1:current[0]==="P"?-1:0;
    const r4=bankerRatio(seq,4),r8=bankerRatio(seq,8),r16=bankerRatio(seq,16),r24=bankerRatio(seq,24),r32=bankerRatio(seq,32),r64=bankerRatio(seq,64);
    const t4=turnRate(seq,4),t8=turnRate(seq,8),t16=turnRate(seq,16),t32=turnRate(seq,32);
    const hz=hazard(seq),hs=hsmmStable(seq),vol=runVolatility(seq,6),derived=derivedInfo(seq,8),pair=pairPattern(seq);

    const shortBias=.45*((r4-.5)*2)+.55*((r8-.5)*2);
    const midBias=.55*((r16-.5)*2)+.45*((r24-.5)*2);
    const longBias=.55*((r32-.5)*2)+.45*((r64-.5)*2);
    const ratioDirectional=.44*shortBias+.34*midBias+.22*longBias;
    const ratioDivergence=.60*(shortBias-midBias)+.40*(midBias-longBias);

    const trendRel=clip(.20+.34*(1-t8)+.20*(1-hz)+.16*hs+.10*Math.max(0,sameTail(seq,3)-.5)*2);
    const altRel=clip(.08+.58*clip((t8-.52)/.40)+.18*alternatingTail(seq,4)+.10*alternatingTail(seq,6)-.20*Math.max(0,sameTail(seq,3)-.5)*2);
    const reversalRel=clip(.10+.34*hz+.24*Math.abs(t4-t16)+.18*vol+.14*(1-hs));

    const trendComponent=side*clip(.50*(1-t8)+.28*(1-hz)+.22*hs);
    const alternateComponent=-side*clip((t8-.50)/.42);
    const reversalComponent=clip(-side*clip((hz-.46)/.36)*.65+ratioDivergence*.70,-1,1);

    const maturity=clip(values.length/24);
    const conflict=clip(Math.abs(t4-t16)*.7+Math.abs(shortBias-longBias)*.55+vol*.45);
    const structureConfidence=clip(.26+.24*hs+.18*derived.consensus+.14*derived.support+.18*maturity-.22*conflict);

    let score=.30*ratioDirectional+.27*trendRel*trendComponent+.15*altRel*alternateComponent+.14*reversalRel*reversalComponent+.14*pair.reliability*pair.direction;
    score*=.58+.42*structureConfidence;
    score=clip(score,-1,1);

    const candidates=[["趨勢",trendRel],["跳動",altRel],["轉折",reversalRel],["雙跳",pair.reliability]].sort((a,b)=>b[1]-a[1]);
    let regime="混合";
    if(candidates[0][1]-candidates[1][1]>=.10&&candidates[0][1]>=.52) regime=candidates[0][0];

    let direction,reason;
    if(Math.abs(score)<.035){ direction=deterministicDirection(seq); reason="低差距固定歷史判定"; }
    else { direction=score>0?"B":"P"; reason="穩定化多尺度評分"; }

    const confidence=clip(.50+Math.min(.09,Math.abs(score)*.105+Math.max(0,structureConfidence-.5)*.025),.50,.59);
    return {direction,confidence,score,reason,regime,structureConfidence,context:context256(seq),diagnostics:{shortBias,midBias,longBias,ratioDivergence,trendRel,altRel,reversalRel,pairRel:pair.reliability,hazard:hz,hsmm:hs,turn8:t8,turn32:t32,derivedConsensus:derived.consensus,conflict}};
  }

  function save(){ try{ localStorage.setItem(STORAGE_KEY,JSON.stringify(state)); }catch(_){} }
  function load(){
    try{
      const own=localStorage.getItem(STORAGE_KEY);
      if(own){ const raw=JSON.parse(own); if(raw&&Array.isArray(raw.history)){ state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500); return; } }
      for(const key of LEGACY_KEYS){
        const text=localStorage.getItem(key); if(!text)continue;
        const raw=JSON.parse(text);
        if(raw&&Array.isArray(raw.history)){ state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500); save(); return; }
      }
    }catch(_){}
  }
  function clearAllStorage(){ try{ localStorage.removeItem(STORAGE_KEY); for(const key of LEGACY_KEYS)localStorage.removeItem(key); }catch(_){} }

  function setMessage(text,warning=false){ const box=el("message"); if(!box)return; box.textContent=text; box.classList.toggle("warning",warning); }
  function addOutcome(outcome){ if(!["B","P","T"].includes(outcome))return; state.history.push(outcome); state.active=false; state.lastPrediction=null; save(); setMessage(`已記錄第 ${state.history.length} 局`); render(); }
  function startAnalysis(){ if(!state.history.length){ setMessage("請先輸入牌局紀錄",true); return; } state.lastPrediction=analyzeStable(state.history); state.active=true; save(); setMessage(`第 ${state.history.length+1} 局分析完成`); render(); }
  function backOne(){ if(!state.history.length){ setMessage("目前沒有可返回的紀錄",true); return; } state.history.pop(); state.active=false; state.lastPrediction=null; save(); setMessage("已返回上一局"); render(); }
  function endAnalysis(){ state={history:[],active:false,lastPrediction:null}; clearAllStorage(); setMessage("本靴資料已清空"); render(); }

  function renderHistory(){
    const box=el("historyTrack");
    if(!state.history.length){ box.innerHTML='<div class="empty-history">尚未輸入牌局</div>'; return; }
    box.innerHTML=state.history.map((v,i)=>`<span class="history-chip ${v}" title="第 ${i+1} 局">${v}</span>`).join("");
    box.scrollLeft=box.scrollWidth;
  }
  function renderPrediction(){
    const p=state.active?state.lastPrediction:null,orb=el("directionOrb");
    if(!p){
      el("directionText").textContent="—"; el("directionCode").textContent="等待分析"; el("confidence").textContent="—"; el("regime").textContent="—"; el("strength").textContent="—"; orb.className="direction-orb idle"; return;
    }
    const isB=p.direction==="B";
    el("directionText").textContent=isB?"莊":"閒";
    el("directionCode").textContent=isB?"BANKER":"PLAYER";
    el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";
    el("regime").textContent=p.regime;
    el("strength").textContent=p.structureConfidence>=.68?"穩定":p.structureConfidence>=.52?"中等":"保守";
    orb.className="direction-orb "+(isB?"banker":"player");
  }
  function render(){
    el("roundPill").textContent=`${state.history.length} 局`;
    el("modePill").textContent=state.active?"分析完成":"準備中";
    el("roundCount").textContent=state.history.length;
    renderPrediction(); renderHistory();
  }

  el("btnB").addEventListener("click",()=>addOutcome("B"));
  el("btnP").addEventListener("click",()=>addOutcome("P"));
  el("btnT").addEventListener("click",()=>addOutcome("T"));
  el("btnStart").addEventListener("click",startAnalysis);
  el("btnBack").addEventListener("click",backOne);
  el("btnEnd").addEventListener("click",endAnalysis);

  load(); render();
})();