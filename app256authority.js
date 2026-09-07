(() => {
"use strict";

const BASE = (typeof window !== "undefined") ? window.__BGS256_TEST__ : null;
const V20 = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!BASE || !V20) return;

const VERSION = "V21_TRANSITION_AUTHORITY";
const STORAGE_KEY = "bgs256d_transition_authority_v21";
const SCORE_TEMP = 0.42;
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

function currentState(seq) {
  const rs = BASE.runs(seq);
  const current = rs.at(-1) || ["", 0];
  return { side: current[0], length: current[1], sign: sideSign(current[0]) };
}

function authoritySignals(seq, basePrediction) {
  const current = currentState(seq);
  const motif = V20.transitionMotifBackoff(seq);
  const depth = V20.transitionDepthForecast(seq);
  const v20 = V20.continuationSignals(seq, basePrediction);
  const sameFuture = V20.expectedBranchFutureQuality(seq, current.side);
  const opposite = current.side === "B" ? "P" : "B";
  const reverseFuture = V20.expectedBranchFutureQuality(seq, opposite);

  const motifEdge = signed((motif.pSame - motif.pSwitch) * 2 * motif.support * (0.55 + 0.45 * (motif.agreement || 0)));
  const depthEdge = signed((depth.pSame - depth.pSwitch) * 2 * depth.support);
  const expectedSupport = clip(0.5 * (sameFuture.support || 0) + 0.5 * (reverseFuture.support || 0));
  const expectedEdge = signed((sameFuture.quality - reverseFuture.quality) * 2 * expectedSupport);

  const stage = basePrediction?.candidates || BASE.bigRoadCandidates(seq);
  const stageSupport = clip(0.5 * (stage.stageSupport || 0) + 0.5 * (stage.contextSupport || 0));
  const stageEdge = signed(((stage.stageCont ?? 0.5) - (stage.stageTurn ?? 0.5)) * stageSupport);

  const token = depth.token || "";
  const transitionDepth = depth.depth || 0;
  const recentPersistence = clip((transitionDepth - 1) / 3);
  const recentEdge = token === "S" ? recentPersistence : token === "X" ? -recentPersistence : 0;
  const xPhase = current.length === 1 && token === "X" && transitionDepth >= 2;
  const sPhase = token === "S" && transitionDepth >= 2;

  let sameSwitchEdge;
  if (xPhase) {
    sameSwitchEdge = signed(
      0.34 * motifEdge +
      0.18 * depthEdge +
      0.18 * expectedEdge +
      0.30 * recentEdge
    );
  } else if (sPhase) {
    sameSwitchEdge = signed(
      0.34 * motifEdge +
      0.20 * depthEdge +
      0.20 * expectedEdge +
      0.14 * recentEdge +
      0.12 * stageEdge
    );
  } else {
    sameSwitchEdge = signed(
      0.38 * motifEdge +
      0.20 * depthEdge +
      0.20 * expectedEdge +
      0.12 * recentEdge +
      0.10 * stageEdge
    );
  }

  const evidenceSupport = clip(
    0.28 * (motif.support || 0) +
    0.18 * (depth.support || 0) +
    0.18 * expectedSupport +
    0.12 * stageSupport +
    0.24 * recentPersistence
  );

  let authority = evidenceSupport;
  if (xPhase) authority = Math.max(authority, clip(recentPersistence * 0.72));
  if (sPhase) authority = Math.max(authority, clip(recentPersistence * 0.58));

  const directional = signed(current.sign * sameSwitchEdge);
  const baseGap = Number(basePrediction?.gap || 0);
  const baseSign = Math.sign(baseGap);
  const authoritySign = Math.sign(directional);
  const conflict = baseSign !== 0 && authoritySign !== 0 && baseSign !== authoritySign;

  let baseRetention = 1;
  if (conflict && xPhase && sameSwitchEdge < -0.05) {
    baseRetention = clip(0.46 - 0.34 * authority, 0.18, 0.46);
  } else if (conflict && sPhase && sameSwitchEdge > 0.05) {
    baseRetention = clip(0.58 - 0.28 * authority, 0.28, 0.58);
  } else if (conflict && authority >= 0.28) {
    baseRetention = clip(0.78 - 0.34 * authority, 0.42, 0.78);
  }

  let authoritySpan = 0.055 + 0.145 * authority;
  if (xPhase && transitionDepth >= 3) authoritySpan += 0.025;
  authoritySpan = clip(authoritySpan, 0.04, 0.23);
  const authorityGap = signed(directional) * authoritySpan;

  return {
    current,
    motif,
    depth,
    v20,
    sameFuture,
    reverseFuture,
    motifEdge,
    depthEdge,
    expectedEdge,
    stageEdge,
    recentEdge,
    recentPersistence,
    sameSwitchEdge,
    directional,
    evidenceSupport,
    authority,
    xPhase,
    sPhase,
    conflict,
    baseRetention,
    authorityGap
  };
}

function authorityChoose(seq) {
  const base = BASE.choose(seq);
  const sig = authoritySignals(seq, base);
  const retainedBaseGap = base.gap * sig.baseRetention;
  const softV20 = signed(sig.v20?.continuationDirectional || 0) * 0.018;
  const adjustedGap = retainedBaseGap + sig.authorityGap + softV20;

  let direction;
  if (Math.abs(adjustedGap) <= 1e-9) direction = base.direction;
  else direction = adjustedGap > 0 ? "B" : "P";

  const rawPB = 1 / (1 + Math.exp(-Math.max(-8, Math.min(8, adjustedGap / SCORE_TEMP))));
  const pB = clip(rawPB, PROB_MIN, PROB_MAX), pP = 1 - pB;
  const confidence = direction === "B" ? pB : pP;

  let regime = "前瞻平衡";
  if (sig.xPhase && sig.sameSwitchEdge < -0.08 && sig.authority >= 0.20) regime = "交錯接管";
  else if (sig.sPhase && sig.sameSwitchEdge > 0.08 && sig.authority >= 0.20) regime = "延續接管";
  else if (sig.conflict && sig.authority >= 0.24) regime = "轉換修正";
  else if (sig.v20?.startSignal >= 0.24) regime = "延續前兆";
  else if (sig.v20?.breakSignal >= 0.24) regime = "延續衰竭";

  const strength = clip(
    (base.strength || 0.5) * 0.62 +
    0.18 * sig.authority +
    0.10 * Math.abs(sig.sameSwitchEdge) +
    0.10 * (1 - sig.baseRetention)
  );

  return {
    ...base,
    direction,
    gap: adjustedGap,
    confidence,
    probabilities: { B: pB, P: pP },
    regime,
    strength,
    transitionAuthority: sig,
    v21: {
      version: VERSION,
      baseGap: base.gap,
      retainedBaseGap,
      baseRetention: sig.baseRetention,
      authorityGap: sig.authorityGap,
      softV20,
      adjustedGap,
      xPhase: sig.xPhase,
      sPhase: sig.sPhase,
      conflict: sig.conflict,
      transitionDepth: sig.depth.depth,
      transitionToken: sig.depth.token,
      motifPSame: sig.motif.pSame,
      motifPSwitch: sig.motif.pSwitch,
      authority: sig.authority,
      sameSwitchEdge: sig.sameSwitchEdge
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
      if (raw && Array.isArray(raw.history)) return raw.history.filter(x => ["B", "P", "T"].includes(x)).slice(-500);
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
    const p = authorityChoose(history);
    saveSelection(p.direction);
    renderPrediction(p, history.length);
  });

  const end = document.getElementById("btnEnd");
  if (end) end.addEventListener("click", () => {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
  });
}

if (typeof window !== "undefined") {
  window.__BGS256_AUTHORITY_TEST__ = {
    authoritySignals,
    authorityChoose,
    version: VERSION
  };
}

installUIOverride();
})();
