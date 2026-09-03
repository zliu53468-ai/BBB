(() => {
"use strict";

const SHOE_DIM = 128;
const ROAD_DIM = 128;
const DIM = 256;
const BIG_ROAD_ROWS = 6;
const BIG_ROAD_COLS = 15;
const ARMS = ["P", "B"];
const ALPHA = 0.5;
const RIDGE = 1.0;
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const SCORE_TEMP = 0.42;
const SCORE_TIE_EPS = 1e-9;
const TOTAL_CARDS = 416;
const AVG_CARDS_PER_HAND = 4.9;
const STORAGE_KEY = "bgs256d_frozen_6x15_sensitive_v17";
const LEGACY_KEYS = [
  "bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15",
  "bgs256d_side_aware_column_v14_user_panel_v1",
  "bgs256d_column_geometry_v13_user_panel_v1",
  "bgs256d_stability_v12_user_panel_v1",
  "bgs256d_128plus128_frozen_direct_tech_panel_v1",
  "bgs128d_64plus64_frozen_direct_tech_panel_v1",
  "bgs64d_32plus32_frozen_direct_tech_panel_v2",
  "bgs32d_frozen_direct_tech_panel_v1"
];

const el = id => (typeof document !== "undefined" ? document.getElementById(id) : null);
const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const bp = seq => seq.filter(x => x === "B" || x === "P");
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

let state = {
  history: [],
  active: false,
  lastPrediction: null,
  last_selected: "",
  selection_streak: 0
};

function runs(seq) {
  const a = bp(seq);
  if (!a.length) return [];
  const out = [];
  let side = a[0], n = 1;
  for (let i = 1; i < a.length; i++) {
    if (a[i] === side) n++;
    else { out.push([side, n]); side = a[i]; n = 1; }
  }
  out.push([side, n]);
  return out;
}

function bankerRatio(seq, w) {
  const a = bp(seq).slice(-Math.max(1, w));
  return a.length ? a.filter(x => x === "B").length / a.length : 0.5;
}

function turnRate(seq, w) {
  const a = bp(seq).slice(-Math.max(2, w));
  if (a.length < 2) return 0.5;
  let t = 0;
  for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) t++;
  return t / (a.length - 1);
}

function tieRatio(seq, w = 0) {
  const a = w ? seq.slice(-w) : seq;
  return a.length ? a.filter(x => x === "T").length / a.length : 0;
}

function binaryEntropy(seq, w = 12) {
  const a = bp(seq).slice(-w);
  if (!a.length) return 1;
  const p = a.filter(x => x === "B").length / a.length, q = 1 - p;
  let e = 0;
  if (p) e -= p * Math.log2(p);
  if (q) e -= q * Math.log2(q);
  return clip(e);
}

function outcomeEntropy(seq, w = 12) {
  const a = seq.slice(-w);
  if (!a.length) return 1;
  let e = 0;
  for (const o of ["B", "P", "T"]) {
    const p = a.filter(x => x === o).length / a.length;
    if (p) e -= p * Math.log2(p);
  }
  return clip(e / Math.log2(3));
}

const balance = (seq, w) => clip(1 - Math.abs(bankerRatio(seq, w) - 0.5) * 2);

function sameTail(seq, w) {
  const a = bp(seq).slice(-w);
  return a.length < w ? 0.5 : +a.every(x => x === a[0]);
}

function alternatingTail(seq, w) {
  const a = bp(seq).slice(-w);
  if (a.length < w) return 0.5;
  for (let i = 1; i < a.length; i++) if (a[i] === a[i - 1]) return 0;
  return 1;
}

function runVolatility(seq, w = 6) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  if (h.length < 2) return 0.25;
  let d = 0;
  for (let i = 1; i < h.length; i++) d += Math.abs(h[i] - h[i - 1]);
  return clip(d / (h.length - 1) / 3);
}

function runTrend(seq, w = 5) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  return h.length < 2 ? 0.5 : clip(0.5 + ((h.at(-1) - h[0]) / (h.length - 1)) / 6);
}

function runStats(seq, w) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  if (!h.length) return { avg: 0, max: 0, std: 0 };
  const mean = h.reduce((a, b) => a + b, 0) / h.length;
  const variance = h.reduce((s, x) => s + (x - mean) ** 2, 0) / h.length;
  return { avg: clip(mean / 8), max: clip(Math.max(...h) / 12), std: clip(Math.sqrt(variance) / 6) };
}

function cellKey(row, col) { return `${row}:${col}`; }

function buildBigRoad(seq) {
  const occupied = new Map(), cells = [], streaks = [];
  let currentSide = "", currentRow = 0, currentCol = -1, currentOriginCol = -1, streakIndex = -1, pendingTies = 0;

  const place = (side, row, col, originCol, move) => {
    const cell = { side, row, col, originCol, streakIndex, move, ties: 0, index: cells.length };
    occupied.set(cellKey(row, col), cell);
    cells.push(cell);
    streaks[streakIndex].cells.push(cell);
    return cell;
  };

  for (const outcome of seq) {
    if (outcome === "T") {
      if (cells.length) cells.at(-1).ties++;
      else pendingTies++;
      continue;
    }
    if (outcome !== "B" && outcome !== "P") continue;

    if (!currentSide) {
      currentSide = outcome; currentRow = 0; currentCol = 0; currentOriginCol = 0; streakIndex = 0;
      streaks.push({ side: outcome, originCol: 0, cells: [] });
      const first = place(outcome, 0, 0, 0, "start");
      first.ties += pendingTies; pendingTies = 0;
      continue;
    }

    if (outcome !== currentSide) {
      const previousOrigin = currentOriginCol;
      currentSide = outcome; currentRow = 0; currentCol = previousOrigin + 1;
      while (occupied.has(cellKey(0, currentCol))) currentCol++;
      currentOriginCol = currentCol; streakIndex++;
      streaks.push({ side: outcome, originCol: currentOriginCol, cells: [] });
      place(outcome, currentRow, currentCol, currentOriginCol, "new-column");
      continue;
    }

    const belowRow = currentRow + 1;
    if (belowRow < BIG_ROAD_ROWS && !occupied.has(cellKey(belowRow, currentCol))) {
      currentRow = belowRow;
      place(outcome, currentRow, currentCol, currentOriginCol, "down");
    } else {
      let nextCol = currentCol + 1;
      while (occupied.has(cellKey(currentRow, nextCol))) nextCol++;
      currentCol = nextCol;
      place(outcome, currentRow, currentCol, currentOriginCol, belowRow >= BIG_ROAD_ROWS ? "tail-bottom" : "tail-collision");
    }
  }

  for (const streak of streaks) {
    const rows = streak.cells.map(c => c.row);
    streak.logicalLength = streak.cells.length;
    streak.verticalHeight = rows.length ? Math.max(...rows) + 1 : 0;
    streak.tailLength = streak.cells.filter(c => c.col > streak.originCol).length;
    streak.endRow = streak.cells.length ? streak.cells.at(-1).row : 0;
    streak.endCol = streak.cells.length ? streak.cells.at(-1).col : streak.originCol;
    streak.hasCollisionTail = streak.cells.some(c => c.move === "tail-collision");
    streak.hasBottomTail = streak.cells.some(c => c.move === "tail-bottom");
  }

  const maxCol = cells.length ? Math.max(...cells.map(c => c.col)) : 0;
  const viewStartCol = Math.max(0, maxCol - BIG_ROAD_COLS + 1), viewEndCol = viewStartCol + BIG_ROAD_COLS - 1;
  const visibleCells = cells.filter(c => c.col >= viewStartCol && c.col <= viewEndCol);
  const visibleStreaks = streaks.map(streak => {
    const vcells = streak.cells.filter(c => c.col >= viewStartCol && c.col <= viewEndCol);
    if (!vcells.length) return null;
    const rows = vcells.map(c => c.row);
    return {
      side: streak.side,
      originCol: streak.originCol,
      originVisible: streak.originCol >= viewStartCol && streak.originCol <= viewEndCol,
      cells: vcells,
      visibleLength: vcells.length,
      verticalHeight: Math.max(...rows) + 1,
      tailLength: vcells.filter(c => c.col > streak.originCol).length,
      logicalLength: streak.logicalLength,
      fullVerticalHeight: streak.verticalHeight,
      fullTailLength: streak.tailLength,
      endRow: vcells.at(-1).row,
      endCol: vcells.at(-1).col
    };
  }).filter(Boolean);

  const grid = Array.from({ length: BIG_ROAD_ROWS }, () => Array(BIG_ROAD_COLS).fill(null));
  for (const c of visibleCells) {
    const vc = c.col - viewStartCol;
    if (vc >= 0 && vc < BIG_ROAD_COLS) grid[c.row][vc] = c.side;
  }

  return { rows: BIG_ROAD_ROWS, cols: BIG_ROAD_COLS, cells, streaks, occupied, maxCol, viewStartCol, viewEndCol, visibleCells, visibleStreaks, grid, currentCell: cells.at(-1) || null, currentStreak: streaks.at(-1) || null };
}

function visibleCompletedStreaks(road) {
  const current = road.currentStreak;
  return road.visibleStreaks.filter(s => s.originVisible).filter(s => !current || s.originCol !== current.originCol);
}

function median(values) {
  if (!values.length) return 0;
  const a = [...values].sort((x, y) => x - y), m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function modeValue(values) {
  if (!values.length) return { value: 0, support: 0 };
  const counts = new Map();
  for (const x of values) counts.set(x, (counts.get(x) || 0) + 1);
  let best = values[0], bestCount = 0;
  for (const [value, count] of counts) {
    if (count > bestCount || (count === bestCount && Math.abs(value - median(values)) < Math.abs(best - median(values)))) { best = value; bestCount = count; }
  }
  return { value: best, support: bestCount / values.length };
}

function summarize(values, scaleStd = 6) {
  if (!values.length) return { count: 0, mean: 0, med: 0, mode: 0, recent: 0, std: 0, target: 0, reliability: 0 };
  const mean = values.reduce((a,b)=>a+b,0)/values.length, med = median(values), mv = modeValue(values);
  const recent = values.slice(-3).reduce((a,b)=>a+b,0)/Math.min(3,values.length);
  const variance = values.reduce((s,x)=>s+(x-mean)**2,0)/values.length, std = Math.sqrt(variance);
  const target = 0.48*mv.value + 0.27*med + 0.25*recent;
  const reliability = clip(values.length/3) * clip(0.42 + 0.58*mv.support) * clip(1-std/scaleStd);
  return { count: values.length, mean, med, mode: mv.value, recent, std, target, reliability };
}

function sideRoadStats(completed, side) {
  const items = completed.filter(x => x.side === side).slice(-8);
  if (!items.length) return { count:0, verticalTarget:0, logicalTarget:0, tailTarget:0, reliability:0, verticalReliability:0, logicalReliability:0, tailReliability:0 };
  const v = summarize(items.map(x=>x.verticalHeight), 5), l = summarize(items.map(x=>x.visibleLength), 7), t = summarize(items.map(x=>x.tailLength), 5);
  return {
    count: items.length,
    verticalTarget: v.target, logicalTarget: l.target, tailTarget: t.target,
    verticalReliability: v.reliability, logicalReliability: l.reliability, tailReliability: t.reliability,
    reliability: clip(0.58*v.reliability + 0.32*l.reliability + 0.10*t.reliability),
    verticalStd: v.std, logicalStd: l.std
  };
}

function sideHazard(completed, side, currentVertical, currentLogical) {
  const values = completed.filter(x => x.side === side);
  if (!values.length) return { turn: 0.5, cont: 0.5, support: 0 };
  let reached=0, ended=0, exceeded=0;
  for (const s of values) {
    const comparable = s.verticalHeight >= currentVertical || s.visibleLength >= currentLogical;
    if (!comparable) continue;
    reached++;
    if (Math.abs(s.verticalHeight-currentVertical) < 0.01 && s.visibleLength <= currentLogical) ended++;
    if (s.verticalHeight > currentVertical || s.visibleLength > currentLogical) exceeded++;
  }
  const prior = 1.25, denom = reached + prior*2;
  return { turn: clip((ended+prior)/denom), cont: clip((exceeded+prior)/denom), support: clip(reached/4) };
}

function pairRoadStats(completed, fromSide) {
  const toSide = fromSide === "B" ? "P" : "B", pairs=[];
  for(let i=0;i<completed.length-1;i++) if(completed[i].side===fromSide && completed[i+1].side===toSide) pairs.push([completed[i].verticalHeight,completed[i+1].verticalHeight]);
  const use=pairs.slice(-6);
  if(!use.length)return{count:0,fromTarget:0,toTarget:0,reliability:0};
  const from=use.map(x=>x[0]),to=use.map(x=>x[1]),fm=modeValue(from),tm=modeValue(to);
  return { count:use.length, fromTarget:.58*fm.value+.42*median(from), toTarget:.58*tm.value+.42*median(to), reliability:clip(use.length/3)*clip(.45+.275*fm.support+.275*tm.support) };
}

function closeness(a,b,span=3) {
  if(!b)return .5;
  return clip(1-Math.abs(a-b)/Math.max(span,b*.75+1));
}

function bigRoadGeometry(road) {
  const cells=road.visibleCells;
  if(!cells.length)return{occupancy:0,bankerRatio:.5,topOccupancy:0,bottomOccupancy:0,tailRatio:0,activeColumns:0,avgFill:0,maxFill:0,fillStd:0,profileRegularity:.5,currentRow:0,currentViewCol:0,currentInTail:0,currentAtBottom:0,blockedBelow:0};
  const columnCounts=Array(BIG_ROAD_COLS).fill(0),profiles=Array(BIG_ROAD_COLS).fill(0);
  let top=0,bottom=0,tail=0,banker=0;
  for(const c of cells){const vc=c.col-road.viewStartCol;if(vc<0||vc>=BIG_ROAD_COLS)continue;columnCounts[vc]++;profiles[vc]|=(1<<c.row);if(c.row===0)top++;if(c.row===BIG_ROAD_ROWS-1)bottom++;if(c.col>c.originCol)tail++;if(c.side==="B")banker++;}
  const active=columnCounts.filter(x=>x>0),avg=active.length?active.reduce((a,b)=>a+b,0)/active.length:0,max=active.length?Math.max(...active):0,variance=active.length?active.reduce((s,x)=>s+(x-avg)**2,0)/active.length:0;
  const usedProfiles=profiles.filter(x=>x!==0);let simTotal=0,simN=0;
  for(let i=1;i<usedProfiles.length;i++){const xor=usedProfiles[i]^usedProfiles[i-1];let bits=0;for(let b=0;b<BIG_ROAD_ROWS;b++)bits+=(xor>>b)&1;simTotal+=1-bits/BIG_ROAD_ROWS;simN++;}
  const current=road.currentCell,blockedBelow=current?(current.row>=BIG_ROAD_ROWS-1||road.occupied.has(cellKey(current.row+1,current.col))):false;
  return {occupancy:clip(cells.length/(BIG_ROAD_ROWS*BIG_ROAD_COLS)),bankerRatio:banker/cells.length,topOccupancy:clip(top/BIG_ROAD_COLS),bottomOccupancy:clip(bottom/BIG_ROAD_COLS),tailRatio:tail/cells.length,activeColumns:clip(active.length/BIG_ROAD_COLS),avgFill:clip(avg/BIG_ROAD_ROWS),maxFill:clip(max/BIG_ROAD_ROWS),fillStd:clip(Math.sqrt(variance)/BIG_ROAD_ROWS),profileRegularity:simN?clip(simTotal/simN):.5,currentRow:current?current.row:0,currentViewCol:current?current.col-road.viewStartCol:0,currentInTail:current?+(current.col>current.originCol):0,currentAtBottom:current?+(current.row===BIG_ROAD_ROWS-1):0,blockedBelow:+blockedBelow};
}

function candidateMoveInfo(baseRoad,candidate,seq) {
  const after=buildBigRoad([...seq,candidate]),added=after.cells.length>baseRoad.cells.length?after.cells.at(-1):null;
  if(!added)return{after,row:0,col:0,viewCol:0,down:0,right:0,newColumn:0};
  return {after,row:added.row,col:added.col,viewCol:added.col-after.viewStartCol,down:+(added.move==="down"),right:+(added.move==="tail-bottom"||added.move==="tail-collision"),newColumn:+(added.move==="new-column")};
}

function gridCandidateQuality(baseRoad,afterRoad,candidate) {
  const before=bigRoadGeometry(baseRoad),after=bigRoadGeometry(afterRoad),current=baseRoad.currentCell,isContinue=!!current&&candidate===current.side;
  const regularityGain=clip(.5+(after.profileRegularity-before.profileRegularity)*2.2),fillSmooth=clip(1-after.fillStd),occupancyHeadroom=clip(1-after.occupancy);
  const spatial=clip(.56*regularityGain+.28*fillSmooth+.16*occupancyHeadroom);
  return isContinue?clip(.82*spatial+.18*(1-after.blockedBelow)):spatial;
}

function derivedMark(h,c,row,newCol,off){if(newCol){if(c<off+1)return"";return h[c-1]===h[c-1-off]?"R":"U";}if(c<off)return"";const ref=h[c-off];return(ref>=row)===(ref>=row-1)?"R":"U";}
function buildDerivedRoads(seq){const road=buildBigRoad(seq),streaks=road.streaks,out={big_eye:[],small_road:[],cockroach_road:[]},offsets={big_eye:1,small_road:2,cockroach_road:3},heights=[];for(let i=0;i<streaks.length;i++){heights.push(streaks[i].verticalHeight);for(const[name,off]of Object.entries(offsets)){const m=derivedMark(heights,i,streaks[i].verticalHeight,true,off);if(m)out[name].push(m);}for(let r=2;r<=streaks[i].verticalHeight;r++)for(const[name,off]of Object.entries(offsets)){const m=derivedMark(heights,i,r,false,off);if(m)out[name].push(m);}}return out;}
function regularity(values,w=8){const a=values.slice(-w).filter(x=>x==="R"||x==="U");return a.length?[a.filter(x=>x==="R").length/a.length,a.length]:[.5,0];}
function derivedInfo(seq,w=8){const roads=buildDerivedRoads(seq),[be,bn]=regularity(roads.big_eye,w),[sm,sn]=regularity(roads.small_road,w),[cr,cn]=regularity(roads.cockroach_road,w),mean=(be+sm+cr)/3;return{be,sm,cr,bn,sn,cn,consensus:clip(1-(Math.abs(be-mean)+Math.abs(sm-mean)+Math.abs(cr-mean))/1.5),support:clip((bn+sn+cn)/(w*3))};}

function bigRoadCandidates(seq) {
  const road=buildBigRoad(seq),current=road.currentStreak;
  if(!current)return{B:.5,P:.5,support:0,currentSide:"",currentComplete:.5,continuationNeed:.5,road,geometry:bigRoadGeometry(road),continueImpulse:0,reverseImpulse:0,fastDirectional:0};
  const completed=visibleCompletedStreaks(road),currentSide=current.side,opposite=currentSide==="B"?"P":"B",sign=sideSign(currentSide);
  const own=sideRoadStats(completed,currentSide),opp=sideRoadStats(completed,opposite),currentVertical=Math.min(BIG_ROAD_ROWS,current.verticalHeight),currentLogical=Math.min(current.logicalLength,BIG_ROAD_COLS);
  const hazard=sideHazard(completed,currentSide,currentVertical,currentLogical),pair=pairRoadStats(completed,currentSide),ownV=own.verticalTarget||currentVertical,ownL=own.logicalTarget||currentLogical;
  const currentComplete=own.count?clip(.68*closeness(currentVertical,ownV,2.2)+.32*closeness(currentLogical,ownL,3.2)):.5;
  const verticalNeed=own.count?clip((ownV-currentVertical+.20)/Math.max(1.15,ownV)):.30,logicalNeed=own.count?clip((ownL-currentLogical+.20)/Math.max(1.25,ownL)):.30;
  const continuationNeed=clip(.72*verticalNeed+.28*logicalNeed);
  const contInfo=candidateMoveInfo(road,currentSide,seq),revInfo=candidateMoveInfo(road,opposite,seq),contStreak=contInfo.after.currentStreak,nextVertical=contStreak?contStreak.verticalHeight:currentVertical,nextLogical=contStreak?contStreak.logicalLength:currentLogical;
  const nextFit=own.count?clip(.70*closeness(nextVertical,ownV,2.0)+.30*closeness(nextLogical,ownL,3.0)):.5;
  const pairFit=pair.count?closeness(currentVertical,pair.fromTarget,2.0):.5;
  const overVertical=own.count?clip(Math.max(0,currentVertical-ownV)/Math.max(1.5,ownV*.7)):0,overLogical=own.count?clip(Math.max(0,currentLogical-ownL)/Math.max(2,ownL*.7)):0,overComplete=clip(.72*overVertical+.28*overLogical);
  const continueGrid=gridCandidateQuality(road,contInfo.after,currentSide),reverseGrid=gridCandidateQuality(road,revInfo.after,opposite);
  const cdi=derivedInfo([...seq,currentSide],8),rdi=derivedInfo([...seq,opposite],8),continueDerived=clip(.62*cdi.consensus+.38*cdi.support),reverseDerived=clip(.62*rdi.consensus+.38*rdi.support);
  const geometry=bigRoadGeometry(road);

  const templateReady=clip(.60*own.reliability+.40*Math.min(1,own.count/3));
  const underTarget=clip((ownV-currentVertical)/Math.max(1,ownV));
  const atTarget=own.count?clip(1-Math.abs(currentVertical-ownV)/1.25):0;
  const nextImprovement=own.count?clip((Math.abs(currentVertical-ownV)-Math.abs(nextVertical-ownV)+.15)/1.25):0;
  const continueImpulse=clip(templateReady*(.46*underTarget+.34*nextImprovement+.20*hazard.cont));
  const pairReady=clip(pair.reliability*pairFit);
  const blockedTurn=clip(geometry.blockedBelow*(.55*atTarget+.45*overComplete));
  const reverseImpulse=clip(templateReady*(.38*atTarget+.20*overComplete)+.30*pairReady+.12*blockedTurn);

  const continueBase=.16*hazard.cont+.20*nextFit+.11*continuationNeed+.10*own.reliability+.19*continueGrid+.08*continueDerived+.04*(1-overComplete);
  const reverseBase=.16*hazard.turn+.16*currentComplete+.12*pairFit+.10*pair.reliability+.07*opp.reliability+.19*reverseGrid+.07*reverseDerived+.03*overComplete;
  const continueScore=clip(continueBase+.26*continueImpulse-.10*reverseImpulse);
  const reverseScore=clip(reverseBase+.29*reverseImpulse-.09*continueImpulse);
  const scores=currentSide==="B"?{B:continueScore,P:reverseScore}:{P:continueScore,B:reverseScore};
  const support=clip(.26*own.reliability+.13*opp.reliability+.17*hazard.support+.15*pair.reliability+.14*Math.max(continueGrid,reverseGrid)+.15*Math.max(continueImpulse,reverseImpulse));
  const fastDirectional=signed(sign*(continueImpulse-reverseImpulse));
  const completionReversalDirectional=signed(-sign*reverseImpulse);
  const continuationDirectional=signed(sign*continueImpulse);
  const pairReversalDirectional=signed(-sign*pairReady);
  const blockedReversalDirectional=signed(-sign*blockedTurn);

  return {...scores,support,currentSide,currentComplete,continuationNeed,ownTarget:ownV,ownLogicalTarget:ownL,oppositeTarget:opp.verticalTarget||0,pairReliability:pair.reliability,pairFit,turnProbability:hazard.turn,continueProbability:hazard.cont,continueScore,reverseScore,continueGrid,reverseGrid,continueMove:contInfo,reverseMove:revInfo,road,geometry,continueImpulse,reverseImpulse,fastDirectional,completionReversalDirectional,continuationDirectional,pairReversalDirectional,blockedReversalDirectional,templateReady,atTarget,underTarget,overComplete};
}

function buildShoeVector(seq) {
  const used=Math.min(TOTAL_CARDS,seq.length*AVG_CARDS_PER_HAND),remaining=Math.max(0,TOTAL_CARDS-used),rr=clip(remaining/TOTAL_CARDS),pen=clip(1-rr),maturity=clip(seq.length/70),hands=clip(seq.length/(TOTAL_CARDS/AVG_CARDS_PER_HAND));
  const v=[rr,pen,rr,maturity,...Array(10).fill(1),0,0,clip(1-pen/.35),clip(1-Math.abs(pen-.5)/.35),clip((pen-.55)/.35),hands,rr,clip(Math.log1p(seq.length)/Math.log1p(TOTAL_CARDS/AVG_CARDS_PER_HAND)),tieRatio(seq),tieRatio(seq,8),tieRatio(seq,16),balance(seq,Math.max(1,bp(seq).length)),binaryEntropy(seq,12),outcomeEntropy(seq,12),outcomeEntropy(seq,24),clip(seq.length/32),1,clip(Math.sqrt(seq.length)/Math.sqrt(TOTAL_CARDS/AVG_CARDS_PER_HAND))];
  for(const w of [4,6,12,24,32])v.push(tieRatio(seq,w));
  for(const w of [4,6,8,16,24,32])v.push(binaryEntropy(seq,w));
  for(const w of [6,8,16,32])v.push(outcomeEntropy(seq,w));
  for(const w of [6,8,16,24,32])v.push(balance(seq,w));
  v.push(pen*pen,Math.sqrt(pen),rr*rr,Math.sqrt(rr),clip(1-pen/.18),clip(1-Math.abs(pen-.3)/.22),clip(1-Math.abs(pen-.62)/.24),clip((pen-.72)/.22),clip(seq.length/8),clip(seq.length/16),clip(seq.length/24),clip(seq.length/48));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(tieRatio(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(binaryEntropy(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(outcomeEntropy(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(balance(seq,w));
  v.push(pen**3,rr**3,Math.sqrt(Math.sqrt(pen)),Math.sqrt(Math.sqrt(rr)),clip(1-Math.abs(pen-.125)/.125),clip(1-Math.abs(pen-.375)/.125),clip(1-Math.abs(pen-.625)/.125),clip(1-Math.abs(pen-.875)/.125),clip(seq.length/4),clip(seq.length/12),clip(seq.length/20),clip(seq.length/32),clip(.5+(tieRatio(seq,8)-tieRatio(seq,32))/2),clip(.5+(binaryEntropy(seq,8)-binaryEntropy(seq,32))/2),clip(.5+(balance(seq,8)-balance(seq,32))/2),clip(Math.log1p(seq.length)/Math.log1p(128)));
  if(v.length!==SHOE_DIM)throw new Error(`shoe mismatch ${v.length}`);
  return v;
}

const ROAD_INDEX={};
function buildRoadVector(seq) {
  const rs=runs(seq),current=rs.at(-1)||["",0],prior=i=>rs.length>i?rs.at(-1-i):["",0],d8=derivedInfo(seq,8),s4=runStats(seq,4),s8=runStats(seq,8),s12=runStats(seq,12),cand=bigRoadCandidates(seq),g=cand.geometry;
  const hs=clip(1-runVolatility(seq,6)*.55-Math.abs(turnRate(seq,8)-turnRate(seq,24))*.45),road=[];
  const push=(name,value)=>{if(!(name in ROAD_INDEX))ROAD_INDEX[name]=road.length;road.push(Number.isFinite(+value)?+value:0);};
  push("current_side_banker",sideSign(current[0]));push("current_run_norm",clip(current[1]/12));for(let i=1;i<=6;i++)push(`previous_run_${i}`,clip(prior(i)[1]/12));
  push("side_turn_probability",cand.turnProbability??.5);push("side_continue_probability",cand.continueProbability??.5);push("structure_stability",hs);push("run_volatility",runVolatility(seq,6));push("run_trend",runTrend(seq,5));
  push("derived_big_eye",d8.be);push("derived_small",d8.sm);push("derived_cockroach",d8.cr);push("derived_consensus",d8.consensus);push("derived_support",d8.support);push("same2",sameTail(seq,2));push("same3",sameTail(seq,3));push("alt4",alternatingTail(seq,4));push("alt6",alternatingTail(seq,6));
  for(const w of [2,4,6,8,10,12,16,20,24,32,48,64])push(`banker_bias_${w}`,(bankerRatio(seq,w)-.5)*2);
  for(const w of [2,4,6,8,10,12,16,20,24,32,48,64])push(`turn_rate_${w}`,turnRate(seq,w));
  for(const w of [4,8,16,24,32]){const d=derivedInfo(seq,w);push(`be_${w}`,d.be);push(`sm_${w}`,d.sm);push(`cr_${w}`,d.cr);}
  push("run_avg4",s4.avg);push("run_avg8",s8.avg);push("run_avg12",s12.avg);push("run_max8",s8.max);push("run_max12",s12.max);push("run_std8",s8.std);push("run_std12",s12.std);push("run_trend8",runTrend(seq,8));push("run_trend12",runTrend(seq,12));
  for(const w of [3,5,8,10])push(`alternating_${w}`,alternatingTail(seq,w));for(const w of [4,5,6,8,10])push(`same_${w}`,sameTail(seq,w));
  push("banker_delta_4_16",bankerRatio(seq,4)-bankerRatio(seq,16));push("banker_delta_8_32",bankerRatio(seq,8)-bankerRatio(seq,32));push("banker_delta_16_64",bankerRatio(seq,16)-bankerRatio(seq,64));push("turn_delta_4_16",turnRate(seq,4)-turnRate(seq,16));push("turn_delta_8_32",turnRate(seq,8)-turnRate(seq,32));push("turn_delta_16_64",turnRate(seq,16)-turnRate(seq,64));
  push("bigroad_candidate_B",cand.B-.5);push("bigroad_candidate_P",cand.P-.5);push("bigroad_candidate_gap_B",cand.B-cand.P);push("bigroad_support",cand.support);push("bigroad_current_complete",cand.currentComplete);push("bigroad_continuation_need",cand.continuationNeed);push("bigroad_own_vertical_target",clip((cand.ownTarget||0)/BIG_ROAD_ROWS));push("bigroad_own_logical_target",clip((cand.ownLogicalTarget||0)/BIG_ROAD_COLS));push("bigroad_opposite_vertical_target",clip((cand.oppositeTarget||0)/BIG_ROAD_ROWS));push("bigroad_pair_reliability",cand.pairReliability||0);push("bigroad_continue_score",cand.continueScore||.5);push("bigroad_reverse_score",cand.reverseScore||.5);push("bigroad_continue_grid_quality",cand.continueGrid||.5);push("bigroad_reverse_grid_quality",cand.reverseGrid||.5);
  push("fast_directional_signal",cand.fastDirectional||0);push("continuation_impulse_directional",cand.continuationDirectional||0);push("completion_reversal_directional",cand.completionReversalDirectional||0);push("pair_reversal_directional",cand.pairReversalDirectional||0);push("blocked_reversal_directional",cand.blockedReversalDirectional||0);push("template_ready",cand.templateReady||0);push("column_at_target",cand.atTarget||0);push("column_under_target",cand.underTarget||0);push("column_over_complete",cand.overComplete||0);
  push("grid_current_row",clip(g.currentRow/(BIG_ROAD_ROWS-1)));push("grid_current_col",clip(g.currentViewCol/(BIG_ROAD_COLS-1)));push("grid_visible_occupancy",g.occupancy);push("grid_visible_banker_ratio",(g.bankerRatio-.5)*2);push("grid_top_occupancy",g.topOccupancy);push("grid_bottom_occupancy",g.bottomOccupancy);push("grid_tail_ratio",g.tailRatio);push("grid_current_in_tail",g.currentInTail);push("grid_current_at_bottom",g.currentAtBottom);push("grid_blocked_below",g.blockedBelow);push("grid_continue_moves_down",cand.continueMove?.down||0);push("grid_continue_moves_right",cand.continueMove?.right||0);push("grid_continue_row",clip((cand.continueMove?.row||0)/(BIG_ROAD_ROWS-1)));push("grid_continue_col",clip((cand.continueMove?.viewCol||0)/(BIG_ROAD_COLS-1)));push("grid_reverse_col",clip((cand.reverseMove?.viewCol||0)/(BIG_ROAD_COLS-1)));push("grid_active_columns",g.activeColumns);push("grid_avg_column_fill",g.avgFill);push("grid_max_column_fill",g.maxFill);push("grid_column_fill_std",g.fillStd);push("grid_profile_regularity",g.profileRegularity);
  while(road.length<ROAD_DIM)push(`reserved_${road.length}`,0);
  if(road.length!==ROAD_DIM)throw new Error(`road mismatch ${road.length}`);
  return{vector:road,candidates:cand,bigRoad:cand.road};
}

function context256(seq){const shoe=buildShoeVector(seq),roadData=buildRoadVector(seq),vector=[...shoe,...roadData.vector];if(vector.length!==DIM)throw new Error(`context mismatch ${vector.length}`);return{vector,candidates:roadData.candidates,bigRoad:roadData.bigRoad};}

function frozenPrior(){
  const A={B:Array(DIM).fill(RIDGE),P:Array(DIM).fill(RIDGE)},b={B:Array(DIM).fill(0),P:Array(DIM).fill(0)};
  const setDirectional=(name,weight,precision=1)=>{const local=ROAD_INDEX[name];if(local===undefined)return;const i=SHOE_DIM+local;A.B[i]=precision;A.P[i]=precision;b.B[i]=weight;b.P[i]=-weight;};
  const setCandidate=(name,arm,weight,precision=1)=>{const local=ROAD_INDEX[name];if(local===undefined)return;const i=SHOE_DIM+local;A[arm][i]=precision;b[arm][i]=weight;};
  setDirectional("banker_bias_8",.030,1.25);setDirectional("banker_bias_16",.022,1.30);setDirectional("banker_bias_32",.014,1.35);setDirectional("banker_delta_4_16",.028,1.20);setDirectional("banker_delta_8_32",.020,1.25);
  setDirectional("bigroad_candidate_gap_B",.62,.88);setCandidate("bigroad_candidate_B","B",.19,.94);setCandidate("bigroad_candidate_P","P",.19,.94);
  setDirectional("fast_directional_signal",.56,.86);setDirectional("continuation_impulse_directional",.30,.92);setDirectional("completion_reversal_directional",.34,.90);setDirectional("pair_reversal_directional",.22,.96);setDirectional("blocked_reversal_directional",.16,1.00);
  setDirectional("grid_visible_banker_ratio",.012,1.40);setDirectional("current_side_banker",.008,1.45);
  return{A,b};
}

function modelX(raw){return raw.map(v=>Number.isFinite(+v)?+v:0);}
function scoreArm(arm,x,prior){const A=prior.A[arm],b=prior.b[arm];let mean=0,uncertaintySquared=0;for(let i=0;i<DIM;i++){const a=Math.max(1e-9,A[i]);mean+=x[i]*(b[i]/a);uncertaintySquared+=(x[i]*x[i])/a;}const uncertainty=Math.sqrt(Math.max(0,uncertaintySquared));return{mean,uncertainty,score:mean+ALPHA*uncertainty,effectiveAlpha:ALPHA};}
function deterministicTie(seq){let h=0;const token="FROZEN256_BIGROAD_6X15_SENSITIVE_V17|"+seq.join("");for(let i=0;i<token.length;i++)h=(h*31+token.charCodeAt(i))>>>0;return h%2?"B":"P";}

function choose(seq){
  const ctx=context256(seq),x=modelX(ctx.vector),prior=frozenPrior(),scores={B:scoreArm("B",x,prior),P:scoreArm("P",x,prior)},gap=scores.B.score-scores.P.score;
  let direction,reason;if(Math.abs(gap)<=SCORE_TIE_EPS){direction=deterministicTie(seq);reason="固定歷史平手判定";}else{direction=gap>0?"B":"P";reason="256維＋6×15即時雙臂判斷";}
  const rawPB=1/(1+Math.exp(-Math.max(-8,Math.min(8,gap/SCORE_TEMP)))),pB=clip(rawPB,PROB_MIN,PROB_MAX),pP=1-pB,confidence=direction==="B"?pB:pP;
  const previous=state.last_selected;state.selection_streak=previous===direction?(state.selection_streak||0)+1:1;state.last_selected=direction;
  const c=ctx.candidates;let regime="混合";
  if(c.support>=.24){const currentContinuation=c.currentSide&&direction===c.currentSide;if(Math.abs(c.B-c.P)>=.040||Math.max(c.continueImpulse||0,c.reverseImpulse||0)>=.42)regime=currentContinuation?"大路延續":"大路反轉";else regime="大路觀察";}
  const strength=clip(.40+.30*c.support+.18*Math.max(c.continueImpulse||0,c.reverseImpulse||0)+Math.min(.18,Math.abs(gap)*.72));
  return{direction,reason,x,scores,gap,confidence,probabilities:{B:pB,P:pP},regime,strength,candidates:c,bigRoad:{rows:BIG_ROAD_ROWS,cols:BIG_ROAD_COLS,viewStartCol:ctx.bigRoad.viewStartCol,viewEndCol:ctx.bigRoad.viewEndCol,maxCol:ctx.bigRoad.maxCol,grid:ctx.bigRoad.grid},frozen:{bootstrap:false,walk_forward:false,replay:false,settle_previous:false,update_ab:false,decay:false,alpha:ALPHA,ridge:RIDGE,context_dim:DIM}};
}

function save(){try{if(typeof localStorage!=="undefined")localStorage.setItem(STORAGE_KEY,JSON.stringify({history:state.history,last_selected:state.last_selected,selection_streak:state.selection_streak}));}catch(_){}}
function load(){try{if(typeof localStorage==="undefined")return;let text=localStorage.getItem(STORAGE_KEY);if(!text){for(const key of LEGACY_KEYS){const t=localStorage.getItem(key);if(t){text=t;break;}}}const raw=JSON.parse(text||"null");if(!raw||!Array.isArray(raw.history))return;state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500);state.last_selected=["B","P"].includes(raw.last_selected)?raw.last_selected:"";state.selection_streak=Math.max(0,+raw.selection_streak||0);}catch(_){}}
function clearAllStorage(){try{if(typeof localStorage==="undefined")return;localStorage.removeItem(STORAGE_KEY);for(const key of LEGACY_KEYS)localStorage.removeItem(key);}catch(_){}}
function setMessage(text,warning=false){const box=el("message");if(!box)return;box.textContent=text;box.classList.toggle("warning",warning);}
function addOutcome(outcome){if(!["B","P","T"].includes(outcome))return;state.history.push(outcome);state.active=false;state.lastPrediction=null;save();setMessage(`已記錄第 ${state.history.length} 局`);render();}
function startAnalysis(){if(!state.history.length){setMessage("請先輸入牌局紀錄",true);return;}state.lastPrediction=choose(state.history);state.active=true;save();setMessage(`第 ${state.history.length+1} 局分析完成`);render();}
function backOne(){if(!state.history.length){setMessage("目前沒有可返回的紀錄",true);return;}state.history.pop();state.active=false;state.lastPrediction=null;save();setMessage("已返回上一局");render();}
function endAnalysis(){state={history:[],active:false,lastPrediction:null,last_selected:"",selection_streak:0};clearAllStorage();setMessage("本靴資料已清空");render();}
function renderHistory(){const box=el("historyTrack");if(!box)return;if(!state.history.length){box.innerHTML='<div class="empty-history">尚未輸入牌局</div>';return;}box.innerHTML=state.history.map((v,i)=>`<span class="history-chip ${v}" title="第 ${i+1} 局">${v}</span>`).join("");box.scrollLeft=box.scrollWidth;}
function renderPrediction(){const p=state.active?state.lastPrediction:null,orb=el("directionOrb");if(!orb)return;if(!p){el("directionText").textContent="—";el("directionCode").textContent="等待分析";el("confidence").textContent="—";el("regime").textContent="—";el("strength").textContent="—";orb.className="direction-orb idle";return;}const isB=p.direction==="B";el("directionText").textContent=isB?"莊":"閒";el("directionCode").textContent=isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";el("regime").textContent=p.regime;el("strength").textContent=p.strength>=.68?"穩定":p.strength>=.52?"中等":"保守";orb.className="direction-orb "+(isB?"banker":"player");}
function render(){if(el("roundPill"))el("roundPill").textContent=`${state.history.length} 局`;if(el("modePill"))el("modePill").textContent=state.active?"分析完成":"準備中";if(el("roundCount"))el("roundCount").textContent=state.history.length;renderPrediction();renderHistory();}

if(el("btnB"))el("btnB").addEventListener("click",()=>addOutcome("B"));
if(el("btnP"))el("btnP").addEventListener("click",()=>addOutcome("P"));
if(el("btnT"))el("btnT").addEventListener("click",()=>addOutcome("T"));
if(el("btnStart"))el("btnStart").addEventListener("click",startAnalysis);
if(el("btnBack"))el("btnBack").addEventListener("click",backOne);
if(el("btnEnd"))el("btnEnd").addEventListener("click",endAnalysis);

if(typeof window!=="undefined")window.__BGS256_TEST__={runs,buildBigRoad,bigRoadGeometry,bigRoadCandidates,context256,choose,constants:{DIM,SHOE_DIM,ROAD_DIM,ALPHA,RIDGE,PROB_MIN,PROB_MAX,BIG_ROAD_ROWS,BIG_ROAD_COLS}};
load();render();
})();
