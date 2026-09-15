(() => {
"use strict";

const CORE = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "XGB_RESIDUAL_BIAS_V7_TIE_STREAK";
const MODEL_URL = "residual_bias_model.json";
const FEATURE_NAMES = [
  "core_p_b",
  "round_index",
  "estimated_total_hands",
  "remaining_ratio",
  "stage",
  "depth",
  "sx_volatility_8",
  "sx_micro_run_4",
  "tie_density",
  "core_streak"
];

const MAX_DELTA_DEFAULT = 0.12;
const VOLATILITY_WINDOW_DEFAULT = 8;
const VOLATILITY_MIN_TOKENS_DEFAULT = 4;
const MICRO_WINDOW_DEFAULT = 4;
const TIE_WINDOW_DEFAULT = 8;
const CORE_STREAK_CAP_DEFAULT = 8;

const REGULAR_VOLATILITY_MAX_DEFAULT = 0.35;
const HIGH_VOLATILITY_START_DEFAULT = 0.85;
const HIGH_DAMPER_MAX_DEFAULT = 0.40;
const HIGH_DAMPER_MIN_DEFAULT = 0.20;
const FAST_UNLOCK_RUN_DEFAULT = 0.75;
const TIE_DAMPING_POWER_DEFAULT = 3.20;
const TIE_DAMPER_MIN_DEFAULT = 0.20;

const BET_WEIGHT_MIN_DEFAULT = 0.20;
const BET_REFERENCE_EDGE_DEFAULT = 0.08;
const ATTACK_STREAK_TRIGGER_DEFAULT = 2;
const ATTACK_SINGLE_WIN_MULTIPLIER_DEFAULT = 1.35;
const DEFENSE_TIE_DENSITY_DEFAULT = 0.25;
const DEFENSE_VOLATILITY_DEFAULT = 0.90;
const DEFENSE_DAMPER_DEFAULT = 0.40;

const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_xgb_residual_training_v1";
const PENDING_KEY = "bgs_xgb_residual_pending_v1";
const SHOE_KEY = "bgs_xgb_residual_shoe_id_v1";
const CUT_KEY = "bgs_xgb_estimated_total_hands_v1";
const CORE_STREAK_KEY = "bgs_xgb_core_streak_v1";
const MAX_TRAINING_ROWS = 10000;

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const clipSigned = (v, limit = 1) => {
  const x = Number.isFinite(+v) ? +v : 0;
  const cap = Math.abs(+limit || 1);
  return Math.max(-cap, Math.min(cap, x));
};
const bp = seq => seq.filter(x => x === "B" || x === "P");

let modelBundle = null;
let modelLoaded = false;
let modelLoadError = "";

function transitionSequence(seq) {
  // T is removed before S/X conversion, so ties never distort road geometry.
  const values = bp(seq);
  const out = [];
  for (let i = 1; i < values.length; i++) out.push(values[i] === values[i - 1] ? "S" : "X");
  return out;
}

function sxVolatility8(seq, window = VOLATILITY_WINDOW_DEFAULT, minTokens = VOLATILITY_MIN_TOKENS_DEFAULT) {
  const tokens = transitionSequence(seq).slice(-Math.max(2, Math.round(window)));
  if (tokens.length < Math.max(2, Math.round(minTokens))) return 0;
  const values = tokens.map(token => token === "S" ? 1 : 0);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return clip(Math.sqrt(variance) / 0.5);
}

function sxMicroRun4(seq, window = MICRO_WINDOW_DEFAULT) {
  const width = Math.max(2, Math.round(window));
  const tokens = transitionSequence(seq).slice(-width);
  if (tokens.length < width) return 0;
  const latest = tokens.at(-1);
  let run = 1;
  for (let i = tokens.length - 2; i >= 0; i--) {
    if (tokens[i] !== latest) break;
    run++;
  }
  const sign = latest === "S" ? 1 : -1;
  return clipSigned(sign * (run / width));
}

function tieDensity8(seq, window = TIE_WINDOW_DEFAULT) {
  const width = Math.max(1, Math.round(window));
  const recent = seq.slice(-width);
  return clip(recent.filter(x => x === "T").length / width);
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

function getCoreStreak() {
  try { return clipSigned(+localStorage.getItem(CORE_STREAK_KEY) || 0, CORE_STREAK_CAP_DEFAULT); }
  catch (_) { return 0; }
}

function setCoreStreak(value) {
  const streak = Math.trunc(clipSigned(value, CORE_STREAK_CAP_DEFAULT));
  try { localStorage.setItem(CORE_STREAK_KEY, String(streak)); } catch (_) {}
  return streak;
}

function updateCoreStreak(previous, coreDirection, actualOutcome) {
  const old = Math.trunc(Number.isFinite(+previous) ? +previous : 0);
  const direction = String(coreDirection || "").toUpperCase();
  const actual = String(actualOutcome || "").toUpperCase();
  if (actual === "T" || !["B", "P"].includes(direction) || !["B", "P"].includes(actual)) return old;
  if (direction === actual) return old > 0 ? old + 1 : 1;
  return old < 0 ? old - 1 : -1;
}

function damperConfig() {
  const cfg = modelBundle?.damper || {};
  return {
    regularMax: clip(cfg.regular_volatility_max ?? REGULAR_VOLATILITY_MAX_DEFAULT, 0, 0.95),
    highStart: clip(cfg.high_volatility_start ?? HIGH_VOLATILITY_START_DEFAULT, 0.01, 0.999999),
    highDamperMax: clip(cfg.high_damper_max ?? HIGH_DAMPER_MAX_DEFAULT, 0.20, 1.0),
    highDamperMin: clip(cfg.high_damper_min ?? HIGH_DAMPER_MIN_DEFAULT, 0, 1.0),
    fastUnlockRun: clip(cfg.fast_unlock_run ?? FAST_UNLOCK_RUN_DEFAULT, 0, 1.0),
    tieDampingPower: Math.max(0, +(cfg.tie_damping_power ?? TIE_DAMPING_POWER_DEFAULT)),
    tieDamperMin: clip(cfg.tie_damper_min ?? TIE_DAMPER_MIN_DEFAULT, 0, 1.0)
  };
}

function regimeDamper(volatility8, microRun4 = 0) {
  const cfg = damperConfig();
  const v = clip(volatility8);
  if (Math.abs(clipSigned(microRun4)) >= cfg.fastUnlockRun) return 1.0;
  const regularMax = cfg.regularMax;
  const highStart = Math.max(regularMax + 1e-6, cfg.highStart);
  const highMax = cfg.highDamperMax;
  const highMin = Math.min(highMax, cfg.highDamperMin);
  if (v <= regularMax) return 1.0;
  if (v < highStart) {
    const t = (v - regularMax) / (highStart - regularMax);
    return clip(1 - (1 - highMax) * t * t, highMax, 1.0);
  }
  const t = (v - highStart) / (1 - highStart);
  return clip(highMax - (highMax - highMin) * t * t, highMin, highMax);
}

function tieDamper(density) {
  const cfg = damperConfig();
  const d = clip(density);
  return clip(1 - cfg.tieDampingPower * d * d, cfg.tieDamperMin, 1.0);
}

function dynamicDamper(volatility8, microRun4 = 0, tieDensity = 0) {
  return Math.min(regimeDamper(volatility8, microRun4), tieDamper(tieDensity));
}

function betWeightFromProbability(finalPB, context = {}) {
  const cfg = modelBundle?.bet_weight || {};
  const minimum = clip(cfg.minimum ?? BET_WEIGHT_MIN_DEFAULT, 0, 1);
  const referenceEdge = Math.max(1e-6, +(cfg.reference_edge ?? BET_REFERENCE_EDGE_DEFAULT));
  const attackTrigger = Math.max(1, Math.round(cfg.attack_streak_trigger ?? ATTACK_STREAK_TRIGGER_DEFAULT));
  const singleWinMultiplier = Math.max(1, +(cfg.attack_single_win_multiplier ?? ATTACK_SINGLE_WIN_MULTIPLIER_DEFAULT));
  const defenseTie = clip(cfg.defense_tie_density ?? DEFENSE_TIE_DENSITY_DEFAULT);
  const defenseVol = clip(cfg.defense_volatility ?? DEFENSE_VOLATILITY_DEFAULT);
  const defenseDamper = clip(cfg.defense_damper ?? DEFENSE_DAMPER_DEFAULT);

  const edge = Math.abs(clip(finalPB) - 0.5);
  let weight = minimum + (1 - minimum) * clip(edge / referenceEdge);
  const streak = Math.trunc(Number.isFinite(+context.coreStreak) ? +context.coreStreak : 0);
  const tieDensity = clip(context.tieDensity || 0);
  const volatility = clip(context.volatility || 0);
  const damper = clip(context.damper ?? 1);
  const severeDefense = tieDensity >= defenseTie || volatility >= defenseVol || damper <= defenseDamper;
  let mode = "EDGE";

  if (severeDefense) {
    weight = minimum;
    mode = "DEFENSE";
  } else if (streak >= attackTrigger) {
    weight = 1.0;
    mode = "ATTACK";
  } else if (streak === 1) {
    weight = clip(weight * singleWinMultiplier, minimum, 1.0);
    mode = "ATTACK_WARMUP";
  } else if (streak <= -2) {
    weight = Math.max(minimum, weight * 0.60);
    mode = "LOSS_DEFENSE";
  } else if (streak === -1) {
    weight = Math.max(minimum, weight * 0.80);
    mode = "LOSS_CAUTION";
  }

  const tier = weight < 0.40 ? "LOW" : weight < 0.70 ? "MEDIUM" : "HIGH";
  return { weight, tier, edge, mode, severeDefense };
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
  const stage = Number.isFinite(+signal?.state?.length) ? Math.abs(+signal.state.length) : currentStage(seq);
  const depth = Number.isFinite(+signal?.depth?.depth) ? +signal.depth.depth : currentDepth(seq);
  return {
    core_p_b: corePB,
    round_index: roundIndex,
    estimated_total_hands: estimatedTotalHands,
    remaining_ratio: remainingRatio,
    stage,
    depth,
    sx_volatility_8: sxVolatility8(seq),
    sx_micro_run_4: sxMicroRun4(seq),
    tie_density: tieDensity8(seq),
    core_streak: getCoreStreak()
  };
}

function featureVector(features) {
  return FEATURE_NAMES.map(name => Number.isFinite(+features[name]) ? +features[name] : 0);
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
    const nextId = !Number.isFinite(value) ? node.missing : value < splitCondition ? node.yes : node.no;
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
  const active = Boolean(modelBundle?.trained);
  const rawDelta = active ? predictRawDelta(features) : 0;
  const maxDelta = clip(modelBundle?.max_delta ?? MAX_DELTA_DEFAULT, 0, MAX_DELTA_DEFAULT);
  const delta = active ? clip(rawDelta, -maxDelta, maxDelta) : 0;

  const corePB = features.core_p_b;
  const volatility = features.sx_volatility_8;
  const micro = features.sx_micro_run_4;
  const ties = features.tie_density;
  const coreStreak = features.core_streak;
  const damper = dynamicDamper(volatility, micro, ties);
  const undampedPB = clip(corePB + delta, 0, 1);
  const finalPB = clip(0.5 + ((corePB - 0.5) + delta) * damper, 0, 1);

  const direction = finalPB > 0.5 ? "B" : "P";
  const finalPP = 1 - finalPB;
  const confidence = direction === "B" ? finalPB : finalPP;
  const coreDirection = String(corePrediction?.direction || (corePB > 0.5 ? "B" : "P"));
  const flipped = direction !== coreDirection;
  const sizing = betWeightFromProbability(finalPB, { coreStreak, tieDensity: ties, volatility, damper });

  let regime = corePrediction.regime;
  if (sizing.mode === "DEFENSE") regime = `${corePrediction.regime}｜和局/震盪防禦`;
  else if (sizing.mode === "ATTACK") regime = `${corePrediction.regime}｜核心連勝進攻`;
  else if (flipped) regime = "XGB殘差修正換邊";
  else if (damper < 1.0) regime = `${corePrediction.regime}｜8局動態阻尼`;

  return {
    ...corePrediction,
    direction,
    confidence,
    probabilities: { B: finalPB, P: finalPP },
    regime,
    residualBias: {
      version: VERSION,
      active,
      modelLoaded,
      modelLoadError,
      coreDirection,
      corePB,
      rawDelta,
      delta,
      undampedPB,
      sxVolatility8: volatility,
      sxMicroRun4: micro,
      tieDensity: ties,
      coreStreak,
      damper,
      finalPB,
      finalDirection: direction,
      flipped,
      betWeight: sizing.weight,
      betWeightTier: sizing.tier,
      betWeightMode: sizing.mode,
      finalEdge: sizing.edge,
      features
    }
  };
}

function readHistory() {
  if (typeof localStorage === "undefined") return [];
  for (const key of ["bgs256d_frozen_6x15_forward_v18", "bgs256d_frozen_6x15_sensitive_v17", "bgs256d_frozen_6x15_bigroad_v16"]) {
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
  } catch (_) { return "browser_shoe"; }
}

function rotateShoeId() {
  try {
    localStorage.removeItem(SHOE_KEY);
    localStorage.removeItem(PENDING_KEY);
    localStorage.removeItem(CORE_STREAK_KEY);
  } catch (_) {}
}

function saveSelection(direction) {
  try {
    const old = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || {};
    const streak = old.last_selected === direction ? Math.max(1, (+old.selection_streak || 0) + 1) : 1;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ last_selected: direction, selection_streak: streak }));
  } catch (_) {}
}

function betTierZh(tier) {
  return tier === "HIGH" ? "高" : tier === "MEDIUM" ? "中" : "低";
}

function renderPrediction(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;
  const isB = p.direction === "B";
  const residual = p.residualBias || {};
  el("directionText").textContent = isB ? "莊" : "閒";
  el("directionCode").textContent = isB ? "BANKER" : "PLAYER";
  el("confidence").textContent = (p.confidence * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent = p.strength >= .68 ? "穩定" : p.strength >= .52 ? "中等" : "保守";
  orb.className = "direction-orb " + (isB ? "banker" : "player");
  if (el("modePill")) el("modePill").textContent = residual.active ? "XGB＋和局/連勝優化" : "和局/連勝阻尼完成";
  if (el("roundCount")) el("roundCount").textContent = historyLength;
  if (el("message")) {
    const weight = Number.isFinite(+residual.betWeight) ? Math.round(+residual.betWeight * 100) : 20;
    el("message").textContent = `第 ${historyLength + 1} 局分析完成｜注碼權重 ${weight}%（${betTierZh(residual.betWeightTier)}）`;
  }
}

function readTrainingRows() {
  try {
    const rows = JSON.parse(localStorage.getItem(TRAINING_KEY) || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch (_) { return []; }
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
  const streakBefore = Math.trunc(Number.isFinite(+pending.features.core_streak) ? +pending.features.core_streak : getCoreStreak());
  const coreDirection = String(pending.core_direction || "").toUpperCase();
  const streakAfter = updateCoreStreak(streakBefore, coreDirection, actual);
  setCoreStreak(streakAfter);

  const row = {
    schema_version: 7,
    shoe_id: String(pending.shoe_id || getShoeId()),
    created_at: +pending.created_at || Date.now(),
    history_fingerprint: String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: actualB - corePB,
    core_direction: coreDirection,
    core_correct: coreDirection === actual,
    core_streak_after: streakAfter,
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
    if (Number.isFinite(+last.core_streak)) setCoreStreak(+last.core_streak);
  }
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({ schema_version: 7, feature_names: FEATURE_NAMES, rows: readTrainingRows() }, null, 2);
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
      if (msg) {
        msg.textContent = "請先輸入牌局紀錄";
        msg.classList.add("warning");
      }
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
    transitionSequence,
    sxVolatility8,
    sxMicroRun4,
    tieDensity8,
    getCoreStreak,
    setCoreStreak,
    updateCoreStreak,
    dynamicDamper,
    betWeightFromProbability,
    applyCorrection,
    loadModel,
    setEstimatedTotalHands,
    getEstimatedTotalHands,
    exportTrainingData,
    downloadTrainingData,
    getTrainingCount: () => readTrainingRows().length,
    getModelStatus: () => ({
      loaded: modelLoaded,
      trained: Boolean(modelBundle?.trained),
      error: modelLoadError,
      version: VERSION,
      coreStreak: getCoreStreak()
    })
  };
}

loadModel();
installUIOverride();
})();
