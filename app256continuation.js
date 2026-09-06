(() => {
"use strict";

const BASE = (typeof window !== "undefined") ? window.__BGS256_TEST__ : null;
if (!BASE) return;

const VERSION = "V20_TRANSITION_FORECAST";
const STORAGE_KEY = "bgs256d_transition_forecast_v20";
const SCORE_TEMP = 0.42;
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const MAX_ADJUSTMENT = 0.12;

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const bp = seq => seq.filter(x => x === "B" || x === "P");
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

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

function transitionRate(arr) {
  if (!arr || arr.length < 2) return 0.5;
  let turns = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] !== arr[i - 1]) turns++;
  return turns / (arr.length - 1);
}

function transitionSequence(seq) {
  const a = bp(seq), out = [];
  for (let i = 1; i < a.length; i++) out.push(a[i] === a[i - 1] ? "S" : "X");
  return out;
}

function persistenceShift(seq) {
  const a = bp(seq);
  if (a.length < 7) return { value: 0.5, support: clip(a.length / 10), recentTurn: 0.5, previousTurn: 0.5 };
  const recent = a.slice(-5);
  const previous = a.slice(Math.max(0, a.length - 13), Math.max(0, a.length - 5));
  const recentTurn = transitionRate(recent);
  const previousTurn = transitionRate(previous);
  const sameRecent = 1 - recentTurn;
  const samePrevious = 1 - previousTurn;
  return {
    value: clip(0.5 + (sameRecent - samePrevious) * 0.75),
    support: clip(Math.min(recent.length - 1, previous.length - 1) / 5),
    recentTurn,
    previousTurn
  };
}

function motifOrderStats(tokens, order) {
  if (tokens.length <= order) return null;
  const context = tokens.slice(-order).join("");
  let same = 0, sw = 0, total = 0;
  const start = Math.max(0, tokens.length - 96);
  for (let i = start; i + order < tokens.length; i++) {
    if (tokens.slice(i, i + order).join("") !== context) continue;
    const age = tokens.length - 1 - (i + order);
    const weight = Math.pow(0.965, age);
    total += weight;
    if (tokens[i + order] === "S") same += weight;
    else sw += weight;
  }
  if (total <= 0) return null;
  const prior = 0.85;
  const denom = total + 2 * prior;
  return {
    order,
    context,
    pSame: clip((same + prior) / denom),
    pSwitch: clip((sw + prior) / denom),
    support: clip(total / 3.5),
    weightedSamples: total
  };
}

function transitionMotifBackoff(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };

  const details = [];
  for (let order = Math.min(5, tokens.length); order >= 1; order--) {
    const item = motifOrderStats(tokens, order);
    if (item) details.push(item);
  }

  if (!details.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };

  let weightedSame = 0, weightTotal = 0, supportTotal = 0;
  const directions = [];
  for (const item of details) {
    const orderWeight = ({5:1.00,4:0.82,3:0.66,2:0.50,1:0.34})[item.order] || 0.25;
    const w = orderWeight * (0.25 + 0.75 * item.support);
    weightedSame += item.pSame * w;
    weightTotal += w;
    supportTotal += item.support * orderWeight;
    if (Math.abs(item.pSame - 0.5) >= 0.035) directions.push(Math.sign(item.pSame - 0.5));
  }
  const pSameRaw = weightTotal ? weightedSame / weightTotal : 0.5;
  const support = clip(supportTotal / 2.4);
  const pSame = clip(0.5 + (pSameRaw - 0.5) * (0.35 + 0.65 * support));
  const agreement = directions.length ? Math.abs(directions.reduce((a,b)=>a+b,0)) / directions.length : 0;
  return {
    pSame,
    pSwitch: 1 - pSame,
    support,
    order: details[0]?.order || 0,
    agreement,
    details
  };
}

function transitionDepthForecast(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, token: "", depth: 0 };
  const token = tokens.at(-1);
  let depth = 1;
  for (let i = tokens.length - 2; i >= 0 && tokens[i] === token; i--) depth++;

  const completedRuns = [];
  let side = tokens[0], n = 1;
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === side) n++;
    else { completedRuns.push({ token: side, length: n }); side = tokens[i]; n = 1; }
  }

  const items = completedRuns.filter(r => r.token === token).slice(-18);
  let reached = 0, continued = 0, ended = 0;
  for (let i = 0; i < items.length; i++) {
    if (items[i].length < depth) continue;
    const recency = Math.pow(0.95, items.length - 1 - i);
    reached += recency;
    if (items[i].length > depth) continued += recency;
    else ended += recency;
  }
  const prior = 0.9, denom = reached + 2 * prior;
  const pPersist = denom > 0 ? clip((continued + prior) / denom) : 0.5;
  const support = clip(reached / 4);
  const adjustedPersist = clip(0.5 + (pPersist - 0.5) * (0.35 + 0.65 * support));
  const pSame = token === "S" ? adjustedPersist : 1 - adjustedPersist;
  return { pSame, pSwitch: 1 - pSame, support, token, depth, pPersist, reached, continued, ended };
}

function empiricalExhaustion(completed, side, currentLength) {
  const items = completed.filter(s => s.side === side).slice(-18);
  let reached = 0, endedHere = 0, exceeded = 0;
  for (let i = 0; i < items.length; i++) {
    const length = items[i].logicalLength;
    if (length < currentLength) continue;
    const recency = Math.pow(0.95, items.length - 1 - i);
    reached += recency;
    if (length === currentLength) endedHere += recency;
    else exceeded += recency;
  }
  if (reached <= 0) return { value: 0.5, turnShare: 0.5, continueShare: 0.5, support: 0, reached: 0 };
  const prior = 0.8, denom = reached + 2 * prior;
  const turnShare = clip((endedHere + prior) / denom);
  const continueShare = clip((exceeded + prior) / denom);
  return {
    value: turnShare,
    turnShare,
    continueShare,
    support: clip(reached / 4),
    reached,
    endedHere,
    exceeded
  };
}

function expectedBranchFutureQuality(seq, first) {
  const opposite = first === "B" ? "P" : "B";
  const road1 = BASE.buildBigRoad([...seq, first]);
  const current = road1.currentStreak;
  if (!current) return { quality: 0.5, pSame: 0.5, support: 0, sameQuality: 0.5, switchQuality: 0.5 };

  const completed = road1.streaks.length > 1 ? road1.streaks.slice(0, -1) : [];
  const stage = BASE.stageSurvival(completed, first, current.logicalLength);
  const context = BASE.contextualStageStats(completed, completed.length, first, current.logicalLength);
  const support = clip(0.56 * (stage.support || 0) + 0.44 * (context.support || 0));
  const rawPSame = clip(0.56 * stage.cont + 0.44 * context.cont);
  const pSame = clip(0.5 + (rawPSame - 0.5) * (0.35 + 0.65 * support));

  const second = BASE.bigRoadCandidates([...seq, first]);
  const sameQuality = clip(second?.[first] ?? 0.5);
  const switchQuality = clip(second?.[opposite] ?? 0.5);
  const quality = clip(pSame * sameQuality + (1 - pSame) * switchQuality);
  return { quality, pSame, support, sameQuality, switchQuality };
}

function continuationSignals(seq, basePrediction) {
  const rs = runs(seq);
  const current = rs.at(-1) || ["", 0];
  const currentSide = current[0];
  const currentLength = current[1];
  const sign = sideSign(currentSide);
  const opposite = currentSide === "B" ? "P" : "B";
  const cand = basePrediction?.candidates || BASE.bigRoadCandidates(seq);
  const road = BASE.buildBigRoad(seq);
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];

  const stageNow = BASE.stageSurvival(completed, currentSide, currentLength);
  const stageNext = BASE.stageSurvival(completed, currentSide, currentLength + 1);
  const shift = persistenceShift(seq);
  const exhaustion = empiricalExhaustion(completed, currentSide, currentLength);
  const motif = transitionMotifBackoff(seq);
  const depth = transitionDepthForecast(seq);

  const sameFuture = expectedBranchFutureQuality(seq, currentSide);
  const reverseFuture = expectedBranchFutureQuality(seq, opposite);
  const expectedDiff = sameFuture.quality - reverseFuture.quality;
  const expectedSupport = clip(0.5 * sameFuture.support + 0.5 * reverseFuture.support);
  const branchCurrentAdv = clip(0.5 + expectedDiff * 1.8);
  const branchReverseAdv = clip(0.5 - expectedDiff * 1.8);

  const contextCont = cand?.contextCont ?? 0.5;
  const contextTurn = cand?.contextTurn ?? 0.5;
  const forwardCurrent = cand?.forwardDirectional == null ? 0.5 : clip(0.5 + sign * cand.forwardDirectional * 0.5);

  const motifSigned = (motif.pSame - motif.pSwitch) * motif.support * (0.6 + 0.4 * motif.agreement);
  const depthSigned = (depth.pSame - depth.pSwitch) * depth.support;
  const expectedSigned = expectedDiff * 2 * expectedSupport;
  const stageSigned = (stageNow.cont - stageNow.turn) * (stageNow.support || 0);
  const sameVsSwitch = signed(
    0.42 * motifSigned +
    0.20 * depthSigned +
    0.21 * expectedSigned +
    0.17 * stageSigned
  );
  const transitionDirectional = signed(sign * sameVsSwitch);

  const earlyGate = clip((4 - currentLength) / 3);
  const startRaw = clip(
    0.22 * stageNow.cont +
    0.17 * contextCont +
    0.20 * motif.pSame +
    0.13 * depth.pSame +
    0.16 * branchCurrentAdv +
    0.07 * shift.value +
    0.05 * forwardCurrent
  );
  const startSupport = clip(
    0.20 * (stageNow.support || 0) +
    0.15 * (cand?.contextSupport || 0) +
    0.25 * motif.support +
    0.15 * depth.support +
    0.10 * expectedSupport +
    0.15 * clip(completed.length / 8)
  );
  const startSignal = clip(earlyGate * startSupport * clip((startRaw - 0.50) / 0.26));

  const survivalDrop = clip(0.5 + (stageNow.cont - stageNext.cont) * 1.65);
  const maturityGate = clip((currentLength - 1) / 2.5);
  const breakRaw = clip(
    0.20 * stageNow.turn +
    0.16 * contextTurn +
    0.20 * motif.pSwitch +
    0.14 * depth.pSwitch +
    0.12 * survivalDrop +
    0.10 * exhaustion.turnShare +
    0.08 * branchReverseAdv
  );
  const breakSupport = clip(
    0.20 * (stageNow.support || 0) +
    0.14 * (cand?.contextSupport || 0) +
    0.24 * motif.support +
    0.16 * depth.support +
    0.16 * exhaustion.support +
    0.10 * expectedSupport
  );
  const breakSignal = clip(maturityGate * breakSupport * clip((breakRaw - 0.50) / 0.26));

  const continuationDirectional = signed(sign * (startSignal - breakSignal));
  const directional = signed(0.66 * transitionDirectional + 0.34 * continuationDirectional);

  return {
    currentSide,
    currentLength,
    startSignal,
    breakSignal,
    startRaw,
    breakRaw,
    startSupport,
    breakSupport,
    stageNow,
    stageNext,
    survivalDrop,
    shift,
    exhaustion,
    motif,
    depth,
    sameFuture,
    reverseFuture,
    expectedDiff,
    expectedSupport,
    transitionDirectional,
    continuationDirectional,
    sameVsSwitch,
    directional
  };
}

function enhancedChoose(seq) {
  const basePrediction = BASE.choose(seq);
  const sig = continuationSignals(seq, basePrediction);
  const adjustment = signed(sig.directional) * MAX_ADJUSTMENT;
  const adjustedGap = basePrediction.gap + adjustment;
  let direction;
  if (Math.abs(adjustedGap) <= 1e-9) direction = basePrediction.direction;
  else direction = adjustedGap > 0 ? "B" : "P";

  const rawPB = 1 / (1 + Math.exp(-Math.max(-8, Math.min(8, adjustedGap / SCORE_TEMP))));
  const pB = clip(rawPB, PROB_MIN, PROB_MAX), pP = 1 - pB;
  const confidence = direction === "B" ? pB : pP;

  let stateLabel = "前瞻平衡";
  if (sig.currentLength === 1 && sig.motif.support >= 0.28 && sig.motif.pSwitch >= 0.60) stateLabel = "交錯前兆";
  else if (sig.startSignal >= 0.24 && sig.startSignal > sig.breakSignal + 0.06) stateLabel = "延續前兆";
  else if (sig.breakSignal >= 0.24 && sig.breakSignal > sig.startSignal + 0.06) stateLabel = "延續衰竭";

  const strength = clip(
    (basePrediction.strength || 0.5) * 0.76 +
    0.09 * Math.max(sig.startSupport, sig.breakSupport) +
    0.08 * sig.motif.support +
    0.04 * sig.depth.support +
    0.03 * Math.abs(sig.directional)
  );

  return {
    ...basePrediction,
    direction,
    gap: adjustedGap,
    confidence,
    probabilities: { B: pB, P: pP },
    regime: stateLabel,
    strength,
    continuationSignals: sig,
    v20: {
      version: VERSION,
      baseGap: basePrediction.gap,
      adjustment,
      transitionDirectional: sig.transitionDirectional,
      continuationDirectional: sig.continuationDirectional,
      motifPSame: sig.motif.pSame,
      motifPSwitch: sig.motif.pSwitch,
      motifSupport: sig.motif.support,
      depthPSame: sig.depth.pSame,
      depthPSwitch: sig.depth.pSwitch,
      startSignal: sig.startSignal,
      breakSignal: sig.breakSignal
    }
  };
}

function readHistory() {
  if (typeof localStorage === "undefined") return [];
  const keys = [
    "bgs256d_frozen_6x15_forward_v18",
    "bgs256d_frozen_6x15_sensitive_v17",
    "bgs256d_frozen_6x15_bigroad_v16"
  ];
  for (const key of keys) {
    try {
      const raw = JSON.parse(localStorage.getItem(key) || "null");
      if (raw && Array.isArray(raw.history)) return raw.history.filter(x => ["B","P","T"].includes(x)).slice(-500);
    } catch (_) {}
  }
  return [];
}

function saveSelection(direction) {
  if (typeof localStorage === "undefined") return;
  try {
    const old = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const streak = old.last_selected === direction ? Math.max(1, (+old.selection_streak || 0) + 1) : 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ last_selected: direction, selection_streak: streak }));
  } catch (_) {}
}

function renderPrediction(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;
  const isB = p.direction === "B";
  el("directionText").textContent = isB ? "莊" : "閒";
  el("directionCode").textContent = isB ? "BANKER" : "PLAYER";
  el("confidence").textContent = (p.confidence * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent = p.strength >= .68 ? "穩定" : p.strength >= .52 ? "中等" : "保守";
  orb.className = "direction-orb " + (isB ? "banker" : "player");
  if (el("modePill")) el("modePill").textContent = "分析完成";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) el("message").textContent = `第 ${historyLength + 1} 局分析完成`;
}

function installUIOverride() {
  if (typeof document === "undefined") return;
  const oldBtn = document.getElementById("btnStart");
  if (!oldBtn) return;
  const btn = oldBtn.cloneNode(true);
  oldBtn.replaceWith(btn);
  btn.addEventListener("click", () => {
    const history = readHistory();
    if (!history.length) {
      const msg = document.getElementById("message");
      if (msg) { msg.textContent = "請先輸入牌局紀錄"; msg.classList.add("warning"); }
      return;
    }
    const p = enhancedChoose(history);
    saveSelection(p.direction);
    renderPrediction(p, history.length);
  });

  const end = document.getElementById("btnEnd");
  if (end) end.addEventListener("click", () => {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
  });
}

if (typeof window !== "undefined") {
  window.__BGS256_CONTINUATION_TEST__ = {
    continuationSignals,
    enhancedChoose,
    persistenceShift,
    empiricalExhaustion,
    transitionSequence,
    transitionMotifBackoff,
    transitionDepthForecast,
    expectedBranchFutureQuality,
    version: VERSION
  };
}

installUIOverride();
})();
