(() => {
"use strict";

const CORE = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "XGB_RESIDUAL_BIAS_V1";
const MODEL_URL = "residual_bias_model.json";
const FEATURE_NAMES = [
  "core_p_b",
  "round_index",
  "estimated_total_hands",
  "remaining_ratio",
  "sx_markov_p_same",
  "stage",
  "depth"
];
const MAX_DELTA_DEFAULT = 0.10;
const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_xgb_residual_training_v1";
const PENDING_KEY = "bgs_xgb_residual_pending_v1";
const SHOE_KEY = "bgs_xgb_residual_shoe_id_v1";
const CUT_KEY = "bgs_xgb_estimated_total_hands_v1";
const MAX_TRAINING_ROWS = 10000;

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const bp = seq => seq.filter(x => x === "B" || x === "P");

let modelBundle = null;
let modelLoaded = false;
let modelLoadError = "";

function transitionSequence(seq) {
  const values = bp(seq), out = [];
  for (let i = 1; i < values.length; i++) out.push(values[i] === values[i - 1] ? "S" : "X");
  return out;
}

function sxMarkovPSame(seq, window = 24, prior = 1.0) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return 0.5;
  const current = tokens.at(-1);
  const start = Math.max(0, tokens.length - 1 - Math.max(2, window));
  let same = 0, sw = 0;
  for (let i = start; i < tokens.length - 1; i++) {
    if (tokens[i] !== current) continue;
    if (tokens[i + 1] === "S") same++;
    else if (tokens[i + 1] === "X") sw++;
  }
  return clip((same + prior) / (same + sw + 2 * prior));
}

function currentStage(seq) {
  const values = bp(seq);
  if (!values.length) return 0;
  const side = values.at(-1);
  let n = 1;
  for (let i = values.length - 2; i >= 0 && values[i] === side; i--) n++;
  return n;
}

function currentDepth(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return 0;
  const token = tokens.at(-1);
  let n = 1;
  for (let i = tokens.length - 2; i >= 0 && tokens[i] === token; i--) n++;
  return n;
}

function getEstimatedTotalHands() {
  try {
    const configured = +(window.__BGS_RESIDUAL_CONFIG__?.estimatedTotalHands || 0);
    if (configured >= 40 && configured <= 90) return configured;
    const stored = +localStorage.getItem(CUT_KEY);
    if (stored >= 40 && stored <= 90) return stored;
  } catch (_) {}
  return 60;
}

function setEstimatedTotalHands(value) {
  const parsed = Math.round(+value || 0);
  if (parsed < 40 || parsed > 90) throw new Error("estimatedTotalHands must be between 40 and 90");
  try { localStorage.setItem(CUT_KEY, String(parsed)); } catch (_) {}
  return parsed;
}

function buildFeatures(seq, corePrediction, signal = null) {
  const probabilities = corePrediction?.probabilities || {};
  const corePB = clip(+probabilities.B || 0.5, 0, 1);
  const roundIndex = Math.max(1, Math.min(70, seq.length + 1));
  const estimatedTotalHands = getEstimatedTotalHands();
  const remainingRatio = clip((estimatedTotalHands - (roundIndex - 1)) / Math.max(1, estimatedTotalHands));
  const stage = Number.isFinite(+signal?.state?.length) ? +signal.state.length : currentStage(seq);
  const depth = Number.isFinite(+signal?.depth?.depth) ? +signal.depth.depth : currentDepth(seq);
  const features = {
    core_p_b: corePB,
    round_index: roundIndex,
    estimated_total_hands: estimatedTotalHands,
    remaining_ratio: remainingRatio,
    sx_markov_p_same: sxMarkovPSame(seq),
    stage,
    depth
  };
  return features;
}

function featureVector(features) {
  return FEATURE_NAMES.map(name => {
    const value = +features[name];
    return Number.isFinite(value) ? value : 0;
  });
}

function findChild(node, nodeId) {
  const children = Array.isArray(node?.children) ? node.children : [];
  return children.find(child => +child.nodeid === +nodeId) || null;
}

function splitIndex(split) {
  const text = String(split ?? "");
  if (/^f\d+$/.test(text)) return +text.slice(1);
  return FEATURE_NAMES.indexOf(text);
}

function evaluateTree(tree, vector) {
  let node = tree;
  let guard = 0;
  while (node && guard++ < 256) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) return +node.leaf || 0;
    const index = splitIndex(node.split);
    const value = index >= 0 ? Math.fround(vector[index]) : NaN;
    const splitCondition = Math.fround(+node.split_condition);
    let nextId;
    if (!Number.isFinite(value)) nextId = node.missing;
    else nextId = value < splitCondition ? node.yes : node.no;
    node = findChild(node, nextId);
  }
  return 0;
}

function predictRawDelta(features) {
  if (!modelBundle?.trained || !Array.isArray(modelBundle.trees)) return 0;
  const vector = featureVector(features);
  let result = +modelBundle.base_score || 0;
  for (const tree of modelBundle.trees) result += evaluateTree(tree, vector);
  return Number.isFinite(result) ? result : 0;
}

function applyCorrection(seq, corePrediction) {
  const signal = corePrediction?.singleHazard || null;
  const features = buildFeatures(seq, corePrediction, signal);
  const rawDelta = predictRawDelta(features);
  const maxDelta = clip(modelBundle?.max_delta ?? MAX_DELTA_DEFAULT, 0, 0.10);
  const delta = clip(rawDelta, -maxDelta, maxDelta);
  const corePB = features.core_p_b;
  const finalPB = clip(corePB + delta, 0, 1);
  const direction = finalPB > 0.5 ? "B" : "P";
  const finalPP = 1 - finalPB;
  const confidence = direction === "B" ? finalPB : finalPP;
  const coreDirection = String(corePrediction?.direction || (corePB > 0.5 ? "B" : "P"));
  const active = Boolean(modelBundle?.trained);
  if (!active) {
    return {
      ...corePrediction,
      residualBias: {
        version: VERSION,
        active: false,
        modelLoaded,
        modelLoadError,
        coreDirection,
        corePB,
        rawDelta: 0,
        delta: 0,
        finalPB: corePB,
        finalDirection: coreDirection,
        flipped: false,
        features
      }
    };
  }
  const flipped = direction !== coreDirection;
  return {
    ...corePrediction,
    direction,
    confidence,
    probabilities: { B: finalPB, P: finalPP },
    regime: flipped ? "XGB殘差修正換邊" : corePrediction.regime,
    residualBias: {
      version: VERSION,
      active: true,
      modelLoaded,
      modelLoadError,
      coreDirection,
      corePB,
      rawDelta,
      delta,
      finalPB,
      finalDirection: direction,
      flipped,
      features
    }
  };
}

function readHistory() {
  if (typeof localStorage === "undefined") return [];
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

function getShoeId() {
  try {
    let id = String(localStorage.getItem(SHOE_KEY) || "");
    if (!id) {
      id = `shoe_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
      localStorage.setItem(SHOE_KEY, id);
    }
    return id;
  } catch (_) {
    return "browser_shoe";
  }
}

function rotateShoeId() {
  try {
    localStorage.removeItem(SHOE_KEY);
    localStorage.removeItem(PENDING_KEY);
  } catch (_) {}
}

function saveSelection(direction) {
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
  if (el("modePill")) el("modePill").textContent = p.residualBias?.active ? "XGB 修正完成" : "分析完成";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) el("message").textContent = `第 ${historyLength + 1} 局分析完成`;
}

function readTrainingRows() {
  try {
    const rows = JSON.parse(localStorage.getItem(TRAINING_KEY) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch (_) {
    return [];
  }
}

function writeTrainingRows(rows) {
  try { localStorage.setItem(TRAINING_KEY, JSON.stringify(rows.slice(-MAX_TRAINING_ROWS))); } catch (_) {}
}

function registerPrediction(seq, prediction) {
  const residual = prediction?.residualBias || {};
  const features = residual.features || buildFeatures(seq, prediction, prediction?.singleHazard || null);
  const pending = {
    shoe_id: getShoeId(),
    created_at: Date.now(),
    history_fingerprint: seq.join(""),
    core_p_b: +features.core_p_b,
    core_direction: residual.coreDirection || String(prediction?.direction || ""),
    features
  };
  try { localStorage.setItem(PENDING_KEY, JSON.stringify(pending)); } catch (_) {}
}

function settlePending(actualOutcome) {
  const actual = String(actualOutcome || "").toUpperCase();
  if (actual === "T") return;
  if (actual !== "B" && actual !== "P") return;
  let pending = null;
  try { pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null"); } catch (_) {}
  if (!pending?.features) return;
  const actualB = actual === "B" ? 1 : 0;
  const corePB = clip(+pending.features.core_p_b || 0.5);
  const row = {
    schema_version: 1,
    shoe_id: String(pending.shoe_id || getShoeId()),
    created_at: +pending.created_at || Date.now(),
    history_fingerprint: String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: actualB - corePB,
    ...pending.features
  };
  const rows = readTrainingRows();
  const duplicate = rows.length && rows.at(-1)?.shoe_id === row.shoe_id && rows.at(-1)?.history_fingerprint === row.history_fingerprint;
  if (!duplicate) rows.push(row);
  writeTrainingRows(rows);
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function rollbackTrainingIfNeeded() {
  const history = readHistory();
  const rows = readTrainingRows();
  if (!rows.length) return;
  const last = rows.at(-1);
  if (last?.shoe_id === getShoeId() && +last.round_index > history.length) {
    rows.pop();
    writeTrainingRows(rows);
  }
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({
    schema_version: 1,
    feature_names: FEATURE_NAMES,
    rows: readTrainingRows()
  }, null, 2);
}

function downloadTrainingData() {
  const blob = new Blob([exportTrainingData()], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bgs_xgb_residual_training_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function loadModel(url = MODEL_URL) {
  modelLoaded = false;
  modelLoadError = "";
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bundle = await response.json();
    if (!bundle || bundle.model_type !== "xgb_residual_regressor") throw new Error("invalid_model_bundle");
    const names = Array.isArray(bundle.feature_names) ? bundle.feature_names : [];
    if (names.join("|") !== FEATURE_NAMES.join("|")) throw new Error("feature_schema_mismatch");
    modelBundle = bundle;
    modelLoaded = true;
    return bundle;
  } catch (error) {
    modelBundle = null;
    modelLoadError = String(error?.message || error || "model_load_failed");
    return null;
  }
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
    const corePrediction = CORE.hazardChoose(history);
    const prediction = applyCorrection(history, corePrediction);
    saveSelection(prediction.direction);
    registerPrediction(history, prediction);
    renderPrediction(prediction, history.length);
  });

  const b = document.getElementById("btnB");
  const p = document.getElementById("btnP");
  const t = document.getElementById("btnT");
  if (b) b.addEventListener("click", () => settlePending("B"));
  if (p) p.addEventListener("click", () => settlePending("P"));
  if (t) t.addEventListener("click", () => settlePending("T"));

  const back = document.getElementById("btnBack");
  if (back) back.addEventListener("click", () => setTimeout(rollbackTrainingIfNeeded, 0));

  const end = document.getElementById("btnEnd");
  if (end) end.addEventListener("click", rotateShoeId);
}

if (typeof window !== "undefined") {
  window.__BGS_RESIDUAL_BIAS__ = {
    version: VERSION,
    featureNames: FEATURE_NAMES,
    buildFeatures,
    sxMarkovPSame,
    applyCorrection,
    loadModel,
    setEstimatedTotalHands,
    getEstimatedTotalHands,
    exportTrainingData,
    downloadTrainingData,
    getTrainingCount: () => readTrainingRows().length,
    getModelStatus: () => ({ loaded: modelLoaded, trained: Boolean(modelBundle?.trained), error: modelLoadError })
  };
}

loadModel();
installUIOverride();
})();