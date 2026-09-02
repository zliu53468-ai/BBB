(() => {
  "use strict";
  const SHOE_DIM=128,ROAD_DIM=128,DIM=256;
  const STORAGE_KEY="bgs256d_column_geometry_v13_user_panel_v1";
  const LEGACY_KEYS=["bgs256d_stability_v12_user_panel_v1","bgs256d_128plus128_frozen_direct_tech_panel_v1","bgs128d_64plus64_frozen_direct_tech_panel_v1","bgs64d_32plus32_frozen_direct_tech_panel_v2","bgs32d_frozen_direct_tech_panel_v1"];
  const TOTAL_CARDS=416,AVG_CARDS_PER_HAND=4.9;
  const el=id=>document.getElementById(id),clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));
  const bp=s=>s.filter(x=>x==="B"||x==="P");
  let state={history:[],active:false,lastPrediction:null};

  function runs(seq){const a=bp(seq);if(!a.length)return[];const o=[];let side=a[0],n=1;for(let i=1;i<a.length;i++){if(a[i]===side)n++;else{o.push([side,n]);side=a[i];n=1}}o.push([side,n]);return o}
  function bankerRatio(seq,w){const a=bp(seq).slice(-Math.max(1,w));return a.length?a.filter(x=>x==="B").length/a.length:.5}
  function turnRate(seq,w){const a=bp(seq).slice(-Math.max(2,w));if(a.length<2)return .5;let t=0;for(let i=1;i<a.length;i++)if(a[i]!==a[i-1])t++;return t/(a.length-1)}
  function tieRatio(seq,w=0){const a=w?seq.slice(-w):seq;return a.length?a.filter(x=>x==="T").length/a.length:0}
  function binaryEntropy(seq,w=12){const a=bp(seq).slice(-w);if(!a.length)return 1;const p=a.filter(x=>x==="B").length/a.length,q=1-p;let e=0;if(p)e-=p*Math.log2(p);if(q)e-=q*Math.log2(q);return clip(e)}
  function outcomeEntropy(seq,w=12){const a=seq.slice(-w);if(!a.length)return 1;let e=0;for(const o of["B","P","T"]){const p=a.filter(x=>x===o).length/a.length;if(p)e-=p*Math.log2(p)}return clip(e/Math.log2(3))}
  const balance=(seq,w)=>clip(1-Math.abs(bankerRatio(seq,w)-.5)*2);
  function sameTail(seq,w){const a=bp(seq).slice(-w);return a.length<w?.5:+a.every(x=>x===a[0])}
  function alternatingTail(seq,w){const a=bp(seq).slice(-w);if(a.length<w)return .5;for(let i=1;i<a.length;i++)if(a[i]===a[i-1])return 0;return 1}
  function runVolatility(seq,w=6){const h=runs(seq).slice(-w).map(x=>x[1]);if(h.length<2)return .25;let d=0;for(let i=1;i<h.length;i++)d+=Math.abs(h[i]-h[i-1]);return clip(d/(h.length-1)/3)}
  function runTrend(seq,w=5){const h=runs(seq).slice(-w).map(x=>x[1]);return h.length<2?.5:clip(.5+((h.at(-1)-h[0])/(h.length-1))/6)}
  function runStats(seq,w){const h=runs(seq).slice(-w).map(x=>x[1]);if(!h.length)return{avg:0,max:0,std:0};const m=h.reduce((a,b)=>a+b,0)/h.length,v=h.reduce((s,x)=>s+(x-m)**2,0)/h.length;return{avg:clip(m/8),max:clip(Math.max(...h)/12),std:clip(Math.sqrt(v)/6)}}
  function hazard(seq){const rs=runs(seq);if(!rs.length)return .5;const current=rs.at(-1)[1],done=rs.slice(0,-1).map(x=>x[1]);if(!done.length)return clip(.38+current*.045,.35,.68);let reached=0,turned=0;for(const n of done){if(n>=current){reached++;if(n===current)turned++}}const empirical=(turned+2.5)/(reached+5),avg=done.reduce((a,b)=>a+b,0)/done.length,shape=clip(.5+(current-avg)/8,.25,.75);return clip(.72*empirical+.28*shape,.18,.82)}
  function hsmmStable(seq){const tr=turnRate(seq,10),rs=runs(seq),cur=rs.length?rs.at(-1)[1]:0,r=clip(cur/6),ent=binaryEntropy(seq,12),vol=runVolatility(seq,6);const trend=Math.exp(-((((tr-.24)/.25)**2)+(((r-.66)/.34)**2)+(((vol-.24)/.28)**2))),jump=Math.exp(-((((tr-.84)/.20)**2)+(((r-.18)/.24)**2)+(((vol-.30)/.30)**2))),noise=Math.exp(-((((tr-.53)/.22)**2)+(((ent-.98)/.18)**2)+(((vol-.62)/.28)**2)));return clip((trend+jump)/(trend+jump+noise+.35))}

  function derivedMark(h,c,row,newCol,off){if(newCol){if(c<off+1)return"";return h[c-1]===h[c-1-off]?"R":"U"}if(c<off)return"";const ref=h[c-off];return(ref>=row)===(ref>=row-1)?"R":"U"}
  function buildDerivedRoads(seq){const a=bp(seq),sides=[],h=[],out={big_eye:[],small_road:[],cockroach_road:[]},offs={big_eye:1,small_road:2,cockroach_road:3};for(const side of a){const nc=!sides.length||side!==sides.at(-1);if(nc){sides.push(side);h.push(1)}else h[h.length-1]++;const c=h.length-1,row=h[c];for(const[k,o]of Object.entries(offs)){const m=derivedMark(h,c,row,nc,o);if(m)out[k].push(m)}}return out}
  function regularity(v,w=8){const a=v.slice(-w).filter(x=>x==="R"||x==="U");return a.length?[a.filter(x=>x==="R").length/a.length,a.length]:[.5,0]}
  function derivedInfo(seq,w=8){const roads=buildDerivedRoads(seq),[be,bn]=regularity(roads.big_eye,w),[sm,sn]=regularity(roads.small_road,w),[cr,cn]=regularity(roads.cockroach_road,w),m=(be+sm+cr)/3;return{be,sm,cr,bn,sn,cn,consensus:clip(1-(Math.abs(be-m)+Math.abs(sm-m)+Math.abs(cr-m))/1.5),support:clip((bn+sn+cn)/(w*3)),roads}}

  function columnGeometry(seq){
    const rs=runs(seq),heights=rs.map(x=>x[1]),cur=heights.at(-1)||0,completed=heights.slice(0,-1),last4=completed.slice(-4),last8=completed.slice(-8);
    const avg=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
    const mean4=avg(last4),mean8=avg(last8),std8=last8.length?Math.sqrt(last8.reduce((s,x)=>s+(x-mean8)**2,0)/last8.length):0;
    let rising=0,falling=0,equal=0;
    for(let i=completed.length-1;i>0;i--){const d=completed[i]-completed[i-1];if(d>0&&falling===0&&equal===0)rising++;else if(d<0&&rising===0&&equal===0)falling++;else if(d===0&&rising===0&&falling===0)equal++;else break}
    const last=completed.at(-1)||0,prev=completed.at(-2)||0,deltaLast=last-prev;
    const expected=last4.length?mean4:(last||cur||1);
    const currentVsMean=clip(.5+(cur-expected)/8);
    const repeatSupport=completed.length?completed.slice(-6).filter(x=>x===cur).length/Math.min(6,completed.length):0;
    const slope=last4.length>=2?(last4.at(-1)-last4[0])/(last4.length-1):0;
    return{heights,cur,last,prev,mean4,mean8,std8,rising,falling,equal,deltaLast,slope,currentVsMean,repeatSupport};
  }
  function geometryQuality(g){
    if(g.heights.length<3)return .5;
    const smooth=clip(1-g.std8/4),repeat=clip(g.repeatSupport),trend=clip(Math.abs(g.slope)/2),stairs=clip((g.rising+g.falling)/4);
    return clip(.28*smooth+.22*repeat+.25*trend+.25*stairs);
  }
  function candidateGeometryScore(baseSeq,candidate){
    const before=columnGeometry(baseSeq),afterSeq=[...baseSeq,candidate],after=columnGeometry(afterSeq),beforeDerived=derivedInfo(baseSeq,8),afterDerived=derivedInfo(afterSeq,8);
    const currentSide=(runs(baseSeq).at(-1)||["",0])[0],continuation=candidate===currentSide;
    const completed=before.heights.slice(0,-1),recent=completed.slice(-6),avg=recent.length?recent.reduce((s,x)=>s+x,0)/recent.length:Math.max(1,before.cur);
    const target=continuation?before.cur+1:1;
    const heightFit=clip(1-Math.abs(target-avg)/Math.max(3,avg+1));
    const repeatFit=recent.length?recent.filter(x=>x===target).length/recent.length:.5;
    const stairTarget=before.last?before.last+Math.sign(before.slope||before.deltaLast||0):target;
    const stairFit=clip(1-Math.abs(target-stairTarget)/4);
    const overflowPenalty=continuation?clip((target-(avg+Math.max(2,before.std8*1.5)))/4):0;
    const derivedGain=clip(.5+(afterDerived.consensus-beforeDerived.consensus)*.65+(afterDerived.support-beforeDerived.support)*.35);
    const geomAfter=geometryQuality(after);
    const regular=clip(.28*heightFit+.18*repeatFit+.18*stairFit+.20*geomAfter+.16*derivedGain-.20*overflowPenalty);
    return{candidate,continuation,regular,heightFit,repeatFit,stairFit,overflowPenalty,derivedGain,geomAfter,targetHeight:target,after};
  }
  function candidateComparison(seq){
    const rs=runs(seq),current=rs.at(-1)||["",0];if(!current[0])return{directional:0,reliability:0,continueScore:.5,reverseScore:.5,continueSide:"",reverseSide:"",detail:{}};
    const continueSide=current[0],reverseSide=current[0]==="B"?"P":"B",c=candidateGeometryScore(seq,continueSide),r=candidateGeometryScore(seq,reverseSide);
    const gap=c.regular-r.regular,reliability=clip(Math.abs(gap)*2.4);
    const directional=(continueSide==="B"?1:-1)*gap;
    return{directional:clip(directional,-1,1),reliability,continueScore:c.regular,reverseScore:r.regular,continueSide,reverseSide,detail:{continue:c,reverse:r}};
  }

  function buildShoeVector(seq){
    const used=Math.min(TOTAL_CARDS,seq.length*AVG_CARDS_PER_HAND),remaining=Math.max(0,TOTAL_CARDS-used),rr=clip(remaining/TOTAL_CARDS),pen=clip(1-rr),maturity=clip(seq.length/70),hands=clip(seq.length/(TOTAL_CARDS/AVG_CARDS_PER_HAND));
    const v=[rr,pen,maturity,hands,tieRatio(seq),balance(seq,Math.max(1,bp(seq).length)),binaryEntropy(seq,12),outcomeEntropy(seq,12),clip(1-Math.abs(pen-.125)/.125),clip(1-Math.abs(pen-.375)/.125),clip(1-Math.abs(pen-.625)/.125),clip(1-Math.abs(pen-.875)/.125),clip(seq.length/8),clip(seq.length/16),clip(seq.length/32),clip(seq.length/64)];
    for(const w of[2,4,6,8,12,16,24,32,48,64])v.push(tieRatio(seq,w),binaryEntropy(seq,w),outcomeEntropy(seq,w),balance(seq,w));
    for(const w of[3,5,7,10,14,20,28,40,56])v.push(tieRatio(seq,w),binaryEntropy(seq,w),outcomeEntropy(seq,w),balance(seq,w));
    v.push(pen**2,pen**3,Math.sqrt(pen),Math.sqrt(Math.sqrt(pen)),rr**2,rr**3,Math.sqrt(rr),Math.sqrt(Math.sqrt(rr)),maturity**2,Math.sqrt(maturity),clip(Math.log1p(seq.length)/Math.log1p(128)),1);
    v.push(clip(.5+(tieRatio(seq,4)-tieRatio(seq,16))/2),clip(.5+(tieRatio(seq,8)-tieRatio(seq,32))/2),clip(.5+(tieRatio(seq,16)-tieRatio(seq,64))/2),clip(.5+(binaryEntropy(seq,4)-binaryEntropy(seq,16))/2),clip(.5+(binaryEntropy(seq,8)-binaryEntropy(seq,32))/2),clip(.5+(binaryEntropy(seq,16)-binaryEntropy(seq,64))/2),clip(.5+(outcomeEntropy(seq,4)-outcomeEntropy(seq,16))/2),clip(.5+(outcomeEntropy(seq,8)-outcomeEntropy(seq,32))/2),clip(.5+(outcomeEntropy(seq,16)-outcomeEntropy(seq,64))/2),clip(.5+(balance(seq,4)-balance(seq,16))/2),clip(.5+(balance(seq,8)-balance(seq,32))/2),clip(.5+(balance(seq,16)-balance(seq,64))/2));
    v.push(clip(1-pen/.30),clip(1-Math.abs(pen-.45)/.30),clip((pen-.55)/.35),clip(1-pen/.16),clip(1-Math.abs(pen-.30)/.20),clip(1-Math.abs(pen-.62)/.24),clip((pen-.76)/.20),clip(seq.length/4),clip(seq.length/12),clip(seq.length/20),clip(seq.length/28),clip(seq.length/40));
    if(v.length!==SHOE_DIM)throw new Error(`shoe vector mismatch: ${v.length}`);return v;
  }
  function buildRoadVector(seq){
    const rs=runs(seq),current=rs.at(-1)||["",0],prior=i=>rs.length>i?rs.at(-1-i):["",0],d8=derivedInfo(seq,8),hz=hazard(seq),hs=hsmmStable(seq),s4=runStats(seq,4),s6=runStats(seq,6),s8=runStats(seq,8),s12=runStats(seq,12),g=columnGeometry(seq),cand=candidateComparison(seq);
    const runDelta=rs.length>1?clip(.5+(current[1]-prior(1)[1])/12):.5;
    const v=[current[0]==="B"?1:current[0]==="P"?0:.5,current[0]==="P"?1:current[0]==="B"?0:.5,clip(current[1]/8),...[1,2,3,4,5,6].map(i=>clip(prior(i)[1]/8)),hz,clip(1-hz),hs,runVolatility(seq,6),runTrend(seq,5),d8.be,d8.sm,d8.cr,d8.consensus,d8.support,clip((d8.be+d8.sm+d8.cr)/3),sameTail(seq,2),sameTail(seq,3),alternatingTail(seq,4),alternatingTail(seq,6)];
    for(const w of[2,4,6,8,10,12,16,20,24,32,48,64])v.push(bankerRatio(seq,w));for(const w of[2,4,6,8,10,12,16,20,24,32,48,64])v.push(turnRate(seq,w));
    for(const w of[4,6,8,12,16,24,32]){const d=derivedInfo(seq,w);v.push(d.be,d.sm,d.cr)}
    v.push(s4.avg,s6.avg,s8.avg,s12.avg,s4.max,s8.max,s12.max,s4.std,s8.std,s12.std,runTrend(seq,8),runTrend(seq,12),runDelta);
    v.push(alternatingTail(seq,3),alternatingTail(seq,5),alternatingTail(seq,8),alternatingTail(seq,10),sameTail(seq,4),sameTail(seq,5),sameTail(seq,6),sameTail(seq,7),sameTail(seq,8),sameTail(seq,10));
    v.push(clip(.5+(bankerRatio(seq,4)-bankerRatio(seq,16))/2),clip(.5+(bankerRatio(seq,8)-bankerRatio(seq,32))/2),clip(.5+(bankerRatio(seq,16)-bankerRatio(seq,64))/2),clip(.5+(turnRate(seq,4)-turnRate(seq,16))/2),clip(.5+(turnRate(seq,8)-turnRate(seq,32))/2),clip(.5+(turnRate(seq,16)-turnRate(seq,64))/2));
    v.push(hz**2,(1-hz)**2,hs**2,d8.consensus**2,clip(g.cur/8),clip(g.last/8),clip(g.mean4/8),clip(g.std8/6),clip(.5+g.slope/6),clip(g.repeatSupport),clip(g.rising/4),clip(g.falling/4),clip(cand.continueScore),clip(cand.reverseScore),clip(.5+cand.directional/2),clip(cand.reliability));
    for(const w of[3,5,7,9,11,14,18])v.push(bankerRatio(seq,w));for(const w of[3,5,7,9,11,14,18])v.push(turnRate(seq,w));
    if(v.length!==ROAD_DIM)throw new Error(`road vector mismatch: ${v.length}`);return v;
  }
  function context256(seq){const shoe=buildShoeVector(seq),road=buildRoadVector(seq),vector=[...shoe,...road];if(vector.length!==DIM)throw new Error(`context mismatch: ${vector.length}`);return vector}
  function pairPattern(seq){const rs=runs(seq);if(rs.length<4)return{reliability:0,direction:0};const done=rs.slice(0,-1).slice(-5).map(x=>x[1]),support=done.filter(x=>x===2).length/Math.max(1,done.length),current=rs.at(-1),side=current[0]==="B"?1:-1;return{reliability:clip((support-.35)/.55),direction:current[1]<=1?side:-side}}
  function deterministicDirection(seq){let h=0;const token="BGS256_COLUMN_V13|"+seq.join("");for(let i=0;i<token.length;i++)h=(h*31+token.charCodeAt(i))>>>0;return h%2?"B":"P"}

  function analyzeStable(seq){
    const values=bp(seq),rs=runs(seq),current=rs.at(-1)||["",0],side=current[0]==="B"?1:current[0]==="P"?-1:0;
    const r4=bankerRatio(seq,4),r8=bankerRatio(seq,8),r16=bankerRatio(seq,16),r24=bankerRatio(seq,24),r32=bankerRatio(seq,32),r64=bankerRatio(seq,64),t4=turnRate(seq,4),t8=turnRate(seq,8),t16=turnRate(seq,16),t32=turnRate(seq,32),hz=hazard(seq),hs=hsmmStable(seq),vol=runVolatility(seq,6),derived=derivedInfo(seq,8),pair=pairPattern(seq),geom=columnGeometry(seq),cand=candidateComparison(seq);
    const shortBias=.45*((r4-.5)*2)+.55*((r8-.5)*2),midBias=.55*((r16-.5)*2)+.45*((r24-.5)*2),longBias=.55*((r32-.5)*2)+.45*((r64-.5)*2),ratioDirectional=.44*shortBias+.34*midBias+.22*longBias,ratioDivergence=.60*(shortBias-midBias)+.40*(midBias-longBias);
    const trendRel=clip(.18+.30*(1-t8)+.18*(1-hz)+.14*hs+.10*Math.max(0,sameTail(seq,3)-.5)*2+.10*geometryQuality(geom));
    const altRel=clip(.07+.52*clip((t8-.52)/.40)+.16*alternatingTail(seq,4)+.09*alternatingTail(seq,6)-.18*Math.max(0,sameTail(seq,3)-.5)*2);
    const reversalRel=clip(.10+.27*hz+.19*Math.abs(t4-t16)+.14*vol+.10*(1-hs)+.20*cand.reliability);
    const trendComponent=side*clip(.46*(1-t8)+.24*(1-hz)+.18*hs+.12*geometryQuality(geom));
    const alternateComponent=-side*clip((t8-.50)/.42),reversalComponent=clip(-side*clip((hz-.46)/.36)*.50+ratioDivergence*.52+cand.directional*.65,-1,1);
    const maturity=clip(values.length/24),conflict=clip(Math.abs(t4-t16)*.55+Math.abs(shortBias-longBias)*.42+vol*.34+(1-cand.reliability)*.08);
    const structureConfidence=clip(.22+.18*hs+.14*derived.consensus+.10*derived.support+.12*maturity+.16*geometryQuality(geom)+.16*cand.reliability-.18*conflict);
    let score=.24*ratioDirectional+.22*trendRel*trendComponent+.11*altRel*alternateComponent+.13*reversalRel*reversalComponent+.10*pair.reliability*pair.direction+.20*cand.directional*(.45+.55*cand.reliability);
    score*=.58+.42*structureConfidence;score=clip(score,-1,1);
    const candidates=[["柱型",cand.reliability],["趨勢",trendRel],["跳動",altRel],["轉折",reversalRel],["雙跳",pair.reliability]].sort((a,b)=>b[1]-a[1]);let regime="混合";if(candidates[0][1]-candidates[1][1]>=.08&&candidates[0][1]>=.50)regime=candidates[0][0];
    let direction,reason;if(Math.abs(score)<.035){direction=deterministicDirection(seq);reason="低差距固定歷史判定"}else{direction=score>0?"B":"P";reason="256維柱高與反轉候選評分"}
    const confidence=clip(.50+Math.min(.09,Math.abs(score)*.10+Math.max(0,structureConfidence-.5)*.025+cand.reliability*.012),.50,.59);
    return{direction,confidence,score,reason,regime,structureConfidence,context:context256(seq),diagnostics:{shortBias,midBias,longBias,ratioDivergence,trendRel,altRel,reversalRel,pairRel:pair.reliability,hazard:hz,hsmm:hs,turn8:t8,turn32:t32,derivedConsensus:derived.consensus,conflict,columnHeights:geom.heights,currentColumnHeight:geom.cur,columnSlope:geom.slope,columnRepeatSupport:geom.repeatSupport,continueSide:cand.continueSide,reverseSide:cand.reverseSide,continueScore:cand.continueScore,reverseScore:cand.reverseScore,candidateReliability:cand.reliability,candidateDirectional:cand.directional}};
  }

  function save(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify(state))}catch(_){}}
  function load(){try{const own=localStorage.getItem(STORAGE_KEY);if(own){const raw=JSON.parse(own);if(raw&&Array.isArray(raw.history)){state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500);return}}for(const key of LEGACY_KEYS){const text=localStorage.getItem(key);if(!text)continue;const raw=JSON.parse(text);if(raw&&Array.isArray(raw.history)){state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500);save();return}}}catch(_){}}
  function clearAllStorage(){try{localStorage.removeItem(STORAGE_KEY);for(const key of LEGACY_KEYS)localStorage.removeItem(key)}catch(_){}}
  function setMessage(text,warning=false){const box=el("message");if(!box)return;box.textContent=text;box.classList.toggle("warning",warning)}
  function addOutcome(o){if(!["B","P","T"].includes(o))return;state.history.push(o);state.active=false;state.lastPrediction=null;save();setMessage(`已記錄第 ${state.history.length} 局`);render()}
  function startAnalysis(){if(!state.history.length){setMessage("請先輸入牌局紀錄",true);return}state.lastPrediction=analyzeStable(state.history);state.active=true;save();setMessage(`第 ${state.history.length+1} 局分析完成`);render()}
  function backOne(){if(!state.history.length){setMessage("目前沒有可返回的紀錄",true);return}state.history.pop();state.active=false;state.lastPrediction=null;save();setMessage("已返回上一局");render()}
  function endAnalysis(){state={history:[],active:false,lastPrediction:null};clearAllStorage();setMessage("本靴資料已清空");render()}
  function renderHistory(){const box=el("historyTrack");if(!state.history.length){box.innerHTML='<div class="empty-history">尚未輸入牌局</div>';return}box.innerHTML=state.history.map((v,i)=>`<span class="history-chip ${v}" title="第 ${i+1} 局">${v}</span>`).join("");box.scrollLeft=box.scrollWidth}
  function renderPrediction(){const p=state.active?state.lastPrediction:null,orb=el("directionOrb");if(!p){el("directionText").textContent="—";el("directionCode").textContent="等待分析";el("confidence").textContent="—";el("regime").textContent="—";el("strength").textContent="—";orb.className="direction-orb idle";return}const isB=p.direction==="B";el("directionText").textContent=isB?"莊":"閒";el("directionCode").textContent=isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";el("regime").textContent=p.regime;el("strength").textContent=p.structureConfidence>=.68?"穩定":p.structureConfidence>=.52?"中等":"保守";orb.className="direction-orb "+(isB?"banker":"player")}
  function render(){el("roundPill").textContent=`${state.history.length} 局`;el("modePill").textContent=state.active?"分析完成":"準備中";el("roundCount").textContent=state.history.length;renderPrediction();renderHistory()}
  el("btnB").addEventListener("click",()=>addOutcome("B"));el("btnP").addEventListener("click",()=>addOutcome("P"));el("btnT").addEventListener("click",()=>addOutcome("T"));el("btnStart").addEventListener("click",startAnalysis);el("btnBack").addEventListener("click",backOne);el("btnEnd").addEventListener("click",endAnalysis);
  load();render();
})();
