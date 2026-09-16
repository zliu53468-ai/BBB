(() => {
"use strict";

const CORE = window.__BGS256_CONTINUATION_TEST__;
const RES = window.__BGS_RESIDUAL_BIAS__;
if (!CORE || !RES) return;

const VERSION = "XGB_RESIDUAL_BIAS_V1_2_STREAK";
const MODEL_URL = "residual_bias_model.json";
const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";

let policy = { trained: false, switch_margin: 0, table_size: 50, model_version: "" };
let previousBeforeClick = "";

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));

function readHistory() {
  for (const key of [
    "bgs256d_frozen_6x15_forward_v18",
    "bgs256d_frozen_6x15_sensitive_v17",
    "bgs256d_frozen_6x15_bigroad_v16"
  ]) {
    try {
      const raw = JSON.parse(localStorage.getItem(key) || "null");
      if (raw && Array.isArray(raw.history)) return raw.history.filter(x => ["B", "P", "T"].includes(x)).slice(-500);
    } catch (_) {}
  }
  return [];
}

function readPreviousDirection() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const side = String(raw.last_selected || "").toUpperCase();
    return side === "B" || side === "P" ? side : "";
  } catch (_) {
    return "";
  }
}

function saveSelection(direction) {
  try {
    const old = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const streak = old.last_selected === direction ? Math.max(1, (+old.selection_streak || 0) + 1) : 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ last_selected: direction, selection_streak: streak }));
  } catch (_) {}
}

function applySwitchMargin(probB, previous, margin) {
  const p = clip(probB);
  const m = clip(margin, 0, 0.05);
  if (previous === "B") return p < 0.5 - m ? "P" : "B";
  if (previous === "P") return p > 0.5 + m ? "B" : "P";
  return p > 0.5 ? "B" : "P";
}

function applyStreakPolicy(prediction, previous) {
  const rb = prediction?.residualBias || {};
  if (!policy.trained || !rb.active) return prediction;

  const finalPB = clip(rb.finalPB ?? prediction?.probabilities?.B ?? 0.5);
  const rawDirection = String(prediction.direction || (finalPB > 0.5 ? "B" : "P"));
  const direction = applySwitchMargin(finalPB, previous, policy.switch_margin);
  const held = direction !== rawDirection;

  if (!held) {
    return {
      ...prediction,
      residualBias: { ...rb, streakPolicyVersion: VERSION, switchMargin: policy.switch_margin, heldByMargin: false }
    };
  }

  const finalPP = 1 - finalPB;
  return {
    ...prediction,
    direction,
    confidence: direction === "B" ? finalPB : finalPP,
    probabilities: { B: finalPB, P: finalPP },
    regime: "XGB臨界區維持",
    residualBias: {
      ...rb,
      streakPolicyVersion: VERSION,
      switchMargin: policy.switch_margin,
      rawDirection,
      finalDirection: direction,
      previousDirection: previous,
      heldByMargin: true
    }
  };
}

function render(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;
  const isB = p.direction === "B";
  el("directionText").textContent = isB ? "莊" : "閒";
  el("directionCode").textContent = isB ? "BANKER" : "PLAYER";
  el("confidence").textContent = (clip(p.confidence) * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent = p.strength >= .68 ? "穩定" : p.strength >= .52 ? "中等" : "保守";
  orb.className = "direction-orb " + (isB ? "banker" : "player");
  if (el("modePill")) el("modePill").textContent = policy.trained ? "XGB V1.2 連勝品質" : "分析完成";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) el("message").textContent = `第 ${historyLength + 1} 局分析完成`;
}

async function loadPolicy() {
  try {
    const response = await fetch(MODEL_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bundle = await response.json();
    policy = {
      trained: Boolean(bundle?.trained && String(bundle?.model_version || "").startsWith("XGB_RESIDUAL_BIAS_V1_2")),
      switch_margin: clip(bundle?.switch_margin ?? 0, 0, 0.05),
      table_size: Math.max(10, +bundle?.table_size || 50),
      model_version: String(bundle?.model_version || "")
    };
  } catch (_) {
    policy = { trained: false, switch_margin: 0, table_size: 50, model_version: "" };
  }
}

function install() {
  const btn = document.getElementById("btnStart");
  if (!btn) return;

  btn.addEventListener("click", () => {
    previousBeforeClick = readPreviousDirection();
  }, true);

  btn.addEventListener("click", () => {
    if (!policy.trained) return;
    const history = readHistory();
    if (!history.length) return;

    const core = CORE.hazardChoose(history);
    const raw = RES.applyCorrection(history, core);
    const adjusted = applyStreakPolicy(raw, previousBeforeClick);
    saveSelection(adjusted.direction);
    render(adjusted, history.length);
  });

  const end = document.getElementById("btnEnd");
  if (end) end.addEventListener("click", () => {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
    previousBeforeClick = "";
  });

  window.__BGS_STREAK_POLICY__ = {
    version: VERSION,
    loadPolicy,
    applySwitchMargin,
    applyStreakPolicy,
    getStatus: () => ({ ...policy })
  };
}

loadPolicy();
install();
})();