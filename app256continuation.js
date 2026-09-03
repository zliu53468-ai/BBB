(() => {
"use strict";

const BASE = (typeof window !== "undefined") ? window.__BGS256_TEST__ : null;
if (!BASE) return;

const VERSION = "V19_CONTINUATION_START_BREAK";
const STORAGE_KEY = "bgs256d_continuation_start_break_v19";
const SCORE_TEMP = 0.42;
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;

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

function persistenceShift(seq) {
  const a = bp(seq);
  if (a.length < 7) return { value: 0.5, support: clip(a.length / 10) };
  const recent = a.slice(-5);
  const previous = a.slice(Math.max(0, a.length - 13), Math.max(0, a.length - 5));
  const recentTurn = transitionRate(recent);
  const previousTurn = transitionRate(previous);
  const sameRecent = 1 - recentTurn;
  const samePrevious = 1 - previousTurn;
  const acceleration = (sameRecent - samePrevious);
  return {
    value: clip(0.5 + acceleration * 0.95),
    support: clip(Math.min(recent.length - 1, previous.length - 1) / 5),
    recentTurn,
    previousTurn
  };
}

function empiricalExhaustion(completed, side, currentLength) {
  const items = completed.filter(s => s.side === side).slice(-16);
  if (!items.length) return { value: 0.5, support: 0, tailShare: 0.5 };
  let weight = 0, endedByNow = 0, exceeded = 0;
  for (let i = 0; i < items.length; i++) {
    const recency = Math.pow(0.95, items.length - 1 - i);
    weight += recency;
    if (items[i].logicalLength <= currentLength) endedByNow += recency;
    if (items[i].logicalLength > currentLength) exceeded += recency;
  }
  const prior = 0.8;
  const tailShare = (endedByNow + prior) / (weight + 2 * prior);
  const continueShare = (exceeded + prior) / (weight + 2 * prior);
  return {
    value: clip(tailShare),
    continueShare: clip(continueShare),
    support: clip(weight / 5),
    tailShare: clip(tailShare)
  };
}

function continuationSignals(seq, basePrediction) {
  const a = bp(seq);
  const rs = runs(seq);
  const current = rs.at(-1) || ["", 0];
  const currentSide = current[0];
  const currentLength = current[1];
  const sign = sideSign(currentSide);
  const cand = basePrediction?.candidates || BASE.bigRoadCandidates(seq);
  const road = BASE.buildBigRoad(seq);
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];

  const stageNow = BASE.stageSurvival(completed, currentSide, currentLength);
  const stageNext = BASE.stageSurvival(completed, currentSide, currentLength + 1);
  const shift = persistenceShift(seq);
  const exhaustion = empiricalExhaustion(completed, currentSide, currentLength);

  const sameFuture = BASE.branchFutureQuality(seq, currentSide);
  const opposite = currentSide === "B" ? "P" : "B";
  const reverseFuture = BASE.branchFutureQuality(seq, opposite);
  const branchCurrentAdv = clip(0.5 + (sameFuture - reverseFuture) * 1.55);
  const branchReverseAdv = clip(0.5 + (reverseFuture - sameFuture) * 1.55);

  const twoStepCurrent = cand?.twoStepDirectional == null ? 0.5 : clip(0.5 + sign * cand.twoStepDirectional * 0.5);
  const forwardCurrent = cand?.forwardDirectional == null ? 0.5 : clip(0.5 + sign * cand.forwardDirectional * 0.5);
  const contextCont = cand?.contextCont ?? 0.5;
  const contextTurn = cand?.contextTurn ?? 0.5;

  const earlyGate = clip((4 - currentLength) / 3);
  const startRaw = clip(
    0.25 * stageNow.cont +
    0.20 * contextCont +
    0.18 * branchCurrentAdv +
    0.15 * shift.value +
    0.12 * twoStepCurrent +
    0.10 * forwardCurrent
  );
  const startSupport = clip(
    0.30 * (stageNow.support || 0) +
    0.22 * (cand?.contextSupport || 0) +
    0.18 * shift.support +
    0.15 * (cand?.support || 0) +
    0.15 * clip(completed.length / 8)
  );
  const startSignal = clip(earlyGate * startSupport * clip((startRaw - 0.50) / 0.30));

  const survivalDrop = clip(0.5 + (stageNow.cont - stageNext.cont) * 1.8);
  const maturityGate = clip((currentLength - 1) / 3);
  const overshoot = cand?.ownTarget ? clip((currentLength - cand.ownTarget + 0.25) / Math.max(2, cand.ownTarget * 0.75)) : 0;
  const breakRaw = clip(
    0.25 * stageNow.turn +
    0.19 * contextTurn +
    0.18 * survivalDrop +
    0.15 * exhaustion.value +
    0.13 * branchReverseAdv +
    0.10 * overshoot
  );
  const breakSupport = clip(
    0.32 * (stageNow.support || 0) +
    0.22 * (cand?.contextSupport || 0) +
    0.20 * exhaustion.support +
    0.14 * (cand?.support || 0) +
    0.12 * clip(completed.length / 8)
  );
  const breakSignal = clip(maturityGate * breakSupport * clip((breakRaw - 0.50) / 0.30));

  const net = signed(startSignal - breakSignal);
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
    sameFuture,
    reverseFuture,
    net,
    directional: signed(sign * net)
  };
}

function enhancedChoose(seq) {
  const basePrediction = BASE.choose(seq);
  const sig = continuationSignals(seq, basePrediction);
  const adjustment = signed(sig.directional) * 0.16;
  const adjustedGap = basePrediction.gap + adjustment;
  let direction;
  if (Math.abs(adjustedGap) <= 1e-9) direction = basePrediction.direction;
  else direction = adjustedGap > 0 ? "B" : "P";

  const rawPB = 1 / (1 + Math.exp(-Math.max(-8, Math.min(8, adjustedGap / SCORE_TEMP))));
  const pB = clip(rawPB, PROB_MIN, PROB_MAX), pP = 1 - pB;
  const confidence = direction === "B" ? pB : pP;

  let stateLabel = "前瞻平衡";
  if (sig.startSignal >= 0.28 && sig.startSignal > sig.breakSignal + 0.08) stateLabel = "延續前兆";
  else if (sig.breakSignal >= 0.28 && sig.breakSignal > sig.startSignal + 0.08) stateLabel = "延續衰竭";

  const strength = clip(
    (basePrediction.strength || 0.5) * 0.78 +
    0.12 * Math.max(sig.startSupport, sig.breakSupport) +
    0.10 * Math.max(sig.startSignal, sig.breakSignal)
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
    v19: {
      version: VERSION,
      baseGap: basePrediction.gap,
      adjustment,
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
    version: VERSION
  };
}

installUIOverride();
})();
