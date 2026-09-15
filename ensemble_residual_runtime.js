(() => {
"use strict";

const CORE = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "LGBM_XGB_RESIDUAL_ENSEMBLE_V1_416";
const MODEL_URL = "ensemble_residual_model.json";
const FEATURE_NAMES = [
  "core_confidence",
  "current_hand",
  "avg_cards_per_hand",
  "remaining_cards_ratio",
  "shoe_progress_delta",
  "sx_markov_p_same",
  "stage",
  "depth"
];

const TOTAL_CARDS = 416;
const MAX_HANDS = 70;
const DEFAULT_MAX_DELTA = 0.10;
const DEFAULT_ESTIMATED_CARDS_PER_HAND = 4.90;

const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_dual_residual_training_v1";
const PENDING_KEY = "bgs_dual_residual_pending_v1";
const SHOE_KEY = "bgs_dual_residual_shoe_id_v1";
const REMAINING_CARDS_KEY = "bgs_dual_remaining_cards_v1";
const EST_CARDS_PER_HAND_KEY = "bgs_dual_est_cards_per_hand_v1";
const MAX_TRAINING_ROWS = 10000;

const clip = (v, lo = 0, hi = 1) =>
  Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const bp = seq => seq.filter(x => x === "B" || x === "P");

let modelBundle = null;
let modelLoaded = false;
let modelLoadError = "";

function transitionSequence(seq) {
  const values = bp(seq), out = [];
  for (let i = 1; i < values.length; i++) {
    out.push(values[i] === values[i - 1] ? "S" : "X");
  }
  return out;
}

function sxMarkovPSame(seq, window = 24, prior = 1.0) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return 0.5;
  const current = tokens.at(-1);
  const start = Math.max(0, tokens.length - 1 - Math.max(2, Math.round(window)));
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
  const state = tokens.at(-1);
  let n = 1;
  for (let i = tokens.length - 2; i >= 0 && tokens[i] === state; i--) n++;
  return n;
}

function getEstimatedCardsPerHand() {
  try {
    const configured = +(window.__BGS_DUAL_RESIDUAL_CONFIG__?.estimatedCardsPerHand || 0);
    if (configured >= 4 && configured <= 6) return configured;
    const stored = +localStorage.getItem(EST_CARDS_PER_HAND_KEY);
    if (stored >= 4 && stored <= 6) return stored;
  } catch (_) {}
  return DEFAULT_ESTIMATED_CARDS_PER_HAND;
}

function setEstimatedCardsPerHand(value) {
  const parsed = +value;
  if (!Number.isFinite(parsed) || parsed < 4 || parsed > 6) {
    throw new Error("estimatedCardsPerHand must be between 4 and 6");
  }
  try { localStorage.setItem(EST_CARDS_PER_HAND_KEY, String(parsed)); } catch (_) {}
  return parsed;
}

function getActualRemainingCards() {
  try {
    const configured = +(window.__BGS_DUAL_RESIDUAL_CONFIG__?.remainingCards);
    if (Number.isFinite(configured) && configured >= 0 && configured <= TOTAL_CARDS) return configured;
    const raw = localStorage.getItem(REMAINING_CARDS_KEY);
    if (raw !== null && raw !== "") {
      const stored = +raw;
      if (Number.isFinite(stored) && stored >= 0 && stored <= TOTAL_CARDS) return stored;
    }
  } catch (_) {}
  return null;
}

function setRemainingCards(value) {
  const parsed = +value;
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > TOTAL_CARDS) {
    throw new Error("remainingCards must be between 0 and 416");
  }
  try { localStorage.setItem(REMAINING_CARDS_KEY, String(parsed)); } catch (_) {}
  return parsed;
}

function clearRemainingCards() {
  try { localStorage.removeItem(REMAINING_CARDS_KEY); } catch (_) {}
}

function physicalShoeContext(currentHand) {
  const exact = getActualRemainingCards();
  const estimatedCardsPerHand = getEstimatedCardsPerHand();
  if (exact !== null) {
    return {
      remainingCards: clip(exact, 0, TOTAL_CARDS),
      source: "actual",
      estimatedCardsPerHand
    };
  }
  const estimate = clip(
    TOTAL_CARDS - currentHand * estimatedCardsPerHand,
    0,
    TOTAL_CARDS
  );
  return {
    remainingCards: estimate,
    source: "estimated",
    estimatedCardsPerHand
  };
}

function buildFeatures(seq, corePrediction, signal = null) {
  const probabilities = corePrediction?.probabilities || {};
  const coreConfidence = clip(+probabilities.B || 0.5);
  const currentHand = Math.max(1, Math.min(MAX_HANDS, seq.length));
  const physical = physicalShoeContext(currentHand);
  const consumed = TOTAL_CARDS - physical.remainingCards;
  const avgCardsPerHand = consumed / currentHand;
  const remainingCardsRatio = physical.remainingCards / TOTAL_CARDS;
  const shoeProgressDelta = (currentHand / MAX_HANDS) - (consumed / TOTAL_CARDS);
  const stage = Number.isFinite(+signal?.state?.length)
    ? +signal.state.length
    : currentStage(seq);
  const depth = Number.isFinite(+signal?.depth?.depth)
    ? +signal.depth.depth
    : currentDepth(seq);

  return {
    features: {
      core_confidence: coreConfidence,
      current_hand: currentHand,
      avg_cards_per_hand: avgCardsPerHand,
      remaining_cards_ratio: remainingCardsRatio,
      shoe_progress_delta: shoeProgressDelta,
      sx_markov_p_same: sxMarkovPSame(seq),
      stage,
      depth
    },
    physical
  };
}

function featureVector(features) {
  return FEATURE_NAMES.map(name => Number.isFinite(+features[name]) ? +features[name] : 0);
}

function evaluateLightGBMNode(node, vector) {
  let current = node;
  let guard = 0;
  while (current && guard++ < 256) {
    if (Object.prototype.hasOwnProperty.call(current, "leaf_value")) {
      const leaf = +current.leaf_value;
      return Number.isFinite(leaf) ? leaf : 0;
    }
    const index = Math.trunc(+current.split_feature);
    const value = index >= 0 && index < vector.length ? +vector[index] : NaN;
    let goLeft;
    if (!Number.isFinite(value)) {
      goLeft = current.default_left !== false;
    } else {
      const threshold = +current.threshold || 0;
      const decision = String(current.decision_type || "<=");
      goLeft = decision.includes("<=") ? value <= threshold : value < threshold;
    }
    current = goLeft ? current.left_child : current.right_child;
  }
  return 0;
}

function xgbSplitIndex(split) {
  const text = String(split ?? "");
  if (/^f\d+$/.test(text)) return +text.slice(1);
  return FEATURE_NAMES.indexOf(text);
}

function findXGBChild(node, nodeId) {
  const children = Array.isArray(node?.children) ? node.children : [];
  return children.find(child => +child.nodeid === +nodeId) || null;
}

function evaluateXGBNode(root, vector) {
  let node = root;
  let guard = 0;
  while (node && guard++ < 256) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
      const leaf = +node.leaf;
      return Number.isFinite(leaf) ? leaf : 0;
    }
    const index = xgbSplitIndex(node.split);
    const value = index >= 0 && index < vector.length ? +vector[index] : NaN;
    let nextId;
    if (!Number.isFinite(value)) {
      nextId = node.missing;
    } else {
      const threshold = +node.split_condition;
      nextId = value < threshold ? node.yes : node.no;
    }
    node = findXGBChild(node, nextId);
  }
  return 0;
}

function ensembleActive() {
  return Boolean(
    modelBundle?.trained &&
    modelBundle?.lightgbm?.trained &&
    modelBundle?.xgboost?.trained
  );
}

function predictLightGBM(features) {
  if (!ensembleActive()) return 0;
  const vector = featureVector(features);
  let result = +modelBundle.lightgbm.base_score || 0;
  const trees = Array.isArray(modelBundle.lightgbm.trees) ? modelBundle.lightgbm.trees : [];
  for (const tree of trees) {
    result += evaluateLightGBMNode(tree?.tree_structure || {}, vector);
  }
  return Number.isFinite(result) ? result : 0;
}

function predictXGBoost(features) {
  if (!ensembleActive()) return 0;
  const vector = featureVector(features);
  let result = +modelBundle.xgboost.base_score || 0;
  const trees = Array.isArray(modelBundle.xgboost.trees) ? modelBundle.xgboost.trees : [];
  for (const tree of trees) result += evaluateXGBNode(tree, vector);
  return Number.isFinite(result) ? result : 0;
}

function predictComponents(features) {
  if (!ensembleActive()) {
    return {
      lightgbmDeltaRaw: 0,
      xgboostDeltaRaw: 0,
      blendedDeltaRaw: 0,
      delta: 0
    };
  }

  const lightgbmDeltaRaw = predictLightGBM(features);
  const xgboostDeltaRaw = predictXGBoost(features);
  const blend = modelBundle?.blend || {};
  let wl = Number.isFinite(+blend.lightgbm_weight) ? +blend.lightgbm_weight : 0.5;
  let wx = Number.isFinite(+blend.xgboost_weight) ? +blend.xgboost_weight : 0.5;
  const weightSum = wl + wx;
  if (!(weightSum > 0)) { wl = 0.5; wx = 0.5; }
  else { wl /= weightSum; wx /= weightSum; }

  const blendedDeltaRaw = wl * lightgbmDeltaRaw + wx * xgboostDeltaRaw;
  const maxDelta = clip(modelBundle?.max_delta ?? DEFAULT_MAX_DELTA, 0, DEFAULT_MAX_DELTA);
  const delta = Math.max(-maxDelta, Math.min(maxDelta, blendedDeltaRaw));
  return { lightgbmDeltaRaw, xgboostDeltaRaw, blendedDeltaRaw, delta };
}

function applyCorrection(seq, corePrediction) {
  const signal = corePrediction?.singleHazard || null;
  const built = buildFeatures(seq, corePrediction, signal);
  const features = built.features;
  const physical = built.physical;
  const components = predictComponents(features);
  const active = ensembleActive();
  const corePB = features.core_confidence;
  const finalPB = clip(corePB + components.delta);
  const finalPP = 1 - finalPB;
  const direction = finalPB > 0.5 ? "B" : "P";
  const confidence = direction === "B" ? finalPB : finalPP;
  const coreDirection = String(corePrediction?.direction || (corePB > 0.5 ? "B" : "P"));
  const flipped = direction !== coreDirection;

  let regime = corePrediction.regime;
  if (flipped) regime = "LGBM/XGB殘差融合換邊";
  else if (active) regime = `${corePrediction.regime}｜雙模型殘差修正`;

  return {
    ...corePrediction,
    direction,
    confidence,
    probabilities: { B: finalPB, P: finalPP },
    regime,
    dualResidual: {
      version: VERSION,
      active,
      modelLoaded,
      modelLoadError,
      coreDirection,
      corePB,
      ...components,
      finalPB,
      finalDirection: direction,
      flipped,
      features,
      physicalShoe: {
        remainingCards: physical.remainingCards,
        source: physical.source,
        estimatedCardsPerHand: physical.estimatedCardsPerHand
      }
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
      if (raw && Array.isArray(raw.history)) {
        return raw.history.filter(x => ["B", "P", "T"].includes(x)).slice(-500);
      }
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
    localStorage.removeItem(REMAINING_CARDS_KEY);
  } catch (_) {}
}

function saveSelection(direction) {
  try {
    const old = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const streak = old.last_selected === direction
      ? Math.max(1, (+old.selection_streak || 0) + 1)
      : 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      last_selected: direction,
      selection_streak: streak
    }));
  } catch (_) {}
}

function renderPrediction(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;
  const isB = p.direction === "B";
  const residual = p.dualResidual || {};
  el("directionText").textContent = isB ? "莊" : "閒";
  el("directionCode").textContent = isB ? "BANKER" : "PLAYER";
  el("confidence").textContent = (p.confidence * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent = p.strength >= .68 ? "穩定" : p.strength >= .52 ? "中等" : "保守";
  orb.className = "direction-orb " + (isB ? "banker" : "player");
  if (el("modePill")) el("modePill").textContent = residual.active ? "256D＋LGBM/XGB" : "256D基底模式";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) {
    const deltaPct = Number.isFinite(+residual.delta) ? (+residual.delta * 100).toFixed(1) : "0.0";
    const physicalLabel = residual.physicalShoe?.source === "actual" ? "實際牌數" : "估算牌數";
    el("message").textContent = `第 ${historyLength + 1} 局分析完成｜融合 Δ ${deltaPct}%｜${physicalLabel}`;
  }
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
  const residual = prediction?.dualResidual || {};
  const fallback = buildFeatures(seq, prediction, prediction?.singleHazard || null);
  const features = residual.features || fallback.features;
  const physical = residual.physicalShoe || fallback.physical;
  const pending = {
    shoe_id: getShoeId(),
    created_at: Date.now(),
    history_fingerprint: seq.join(""),
    core_direction: residual.coreDirection || String(prediction?.direction || ""),
    remaining_cards: +physical.remainingCards,
    remaining_cards_source: String(physical.source || "estimated"),
    estimated_cards_per_hand: +physical.estimatedCardsPerHand || DEFAULT_ESTIMATED_CARDS_PER_HAND,
    features
  };
  try { localStorage.setItem(PENDING_KEY, JSON.stringify(pending)); } catch (_) {}
}

function settlePending(actualOutcome) {
  const actual = String(actualOutcome || "").toUpperCase();
  let pending = null;
  try { pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null"); } catch (_) {}
  if (!pending?.features) return;

  if (actual === "T") {
    try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
    return;
  }
  if (actual !== "B" && actual !== "P") return;

  const actualB = actual === "B" ? 1 : 0;
  const corePB = clip(+pending.features.core_confidence || 0.5);
  const row = {
    schema_version: 1,
    model_type: "lgbm_xgb_residual_ensemble",
    shoe_id: String(pending.shoe_id || getShoeId()),
    created_at: +pending.created_at || Date.now(),
    history_fingerprint: String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: actualB - corePB,
    remaining_cards: +pending.remaining_cards,
    remaining_cards_source: String(pending.remaining_cards_source || "estimated"),
    estimated_cards_per_hand: +pending.estimated_cards_per_hand || DEFAULT_ESTIMATED_CARDS_PER_HAND,
    ...pending.features
  };
  const rows = readTrainingRows();
  const duplicate = rows.length &&
    rows.at(-1)?.shoe_id === row.shoe_id &&
    rows.at(-1)?.history_fingerprint === row.history_fingerprint;
  if (!duplicate) rows.push(row);
  writeTrainingRows(rows);
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function rollbackTrainingIfNeeded() {
  const history = readHistory();
  const rows = readTrainingRows();
  if (!rows.length) return;
  const last = rows.at(-1);
  if (last?.shoe_id === getShoeId() && +last.current_hand > history.length) {
    rows.pop();
    writeTrainingRows(rows);
  }
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({
    schema_version: 1,
    model_type: "lgbm_xgb_residual_ensemble",
    feature_names: FEATURE_NAMES,
    rows: readTrainingRows()
  }, null, 2);
}

function downloadTrainingData() {
  const blob = new Blob([exportTrainingData()], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bgs_dual_residual_training_${Date.now()}.json`;
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
    if (!bundle || bundle.model_type !== "lgbm_xgb_residual_ensemble") {
      throw new Error("invalid_model_bundle");
    }
    const names = Array.isArray(bundle.feature_names) ? bundle.feature_names : [];
    if (names.join("|") !== FEATURE_NAMES.join("|")) {
      throw new Error("feature_schema_mismatch");
    }
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
      if (msg) msg.textContent = "請先輸入牌局紀錄";
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
  window.__BGS_DUAL_RESIDUAL__ = {
    version: VERSION,
    featureNames: FEATURE_NAMES,
    buildFeatures,
    sxMarkovPSame,
    applyCorrection,
    predictComponents,
    loadModel,
    setRemainingCards,
    clearRemainingCards,
    getActualRemainingCards,
    setEstimatedCardsPerHand,
    getEstimatedCardsPerHand,
    exportTrainingData,
    downloadTrainingData,
    getTrainingCount: () => readTrainingRows().length,
    getModelStatus: () => ({
      loaded: modelLoaded,
      trained: ensembleActive(),
      error: modelLoadError,
      version: VERSION
    })
  };
}

loadModel();
installUIOverride();
})();
