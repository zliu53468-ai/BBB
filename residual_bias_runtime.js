(() => {
"use strict";

const CORE = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "XGB_SHOE_REGIME_3D_LIKELIHOOD_V2";
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
const MODEL_FEATURE_NAMES = [...FEATURE_NAMES, "regime_state"];
const MAX_DELTA_DEFAULT = 0.10;
const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_xgb_residual_training_v1";
const PENDING_KEY = "bgs_xgb_residual_pending_v1";
const SHOE_KEY = "bgs_xgb_residual_shoe_id_v1";
const CUT_KEY = "bgs_xgb_estimated_total_hands_v1";
const REGIME_STATE_KEY = "bgs_xgb_shoe_regime_state_v1";
const LEGACY_PF_STATE_KEY = "bgs_xgb_particle_filter_state_v1";
const MAX_TRAINING_ROWS = 10000;
const PF_DEFAULTS = {
  n_particles: 1000,
  state_dim: 1,
  Q_early: 0.005,
  Q_late: 0.02,
  early_round_end: 15,
  late_round_start: 45,
  R: 0.25,
  resample_threshold: 500,
  resampling: "systematic",
  random_state: 42,
  state_clip: 1.0,
  likelihood_weights: {
    directionality: 0.55,
    residual_alignment: 0.30,
    persistence: 0.15
  }
};

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
  return {
    core_p_b: corePB,
    round_index: roundIndex,
    estimated_total_hands: estimatedTotalHands,
    remaining_ratio: remainingRatio,
    sx_markov_p_same: sxMarkovPSame(seq),
    stage,
    depth
  };
}

function modelFeatureVector(features, regimeState) {
  const vector = FEATURE_NAMES.map(name => {
    const value = +features[name];
    return Number.isFinite(value) ? value : 0;
  });
  vector.push(Number.isFinite(+regimeState) ? +regimeState : 0);
  return vector;
}

function findChild(node, nodeId) {
  const children = Array.isArray(node?.children) ? node.children : [];
  return children.find(child => +child.nodeid === +nodeId) || null;
}

function splitIndex(split) {
  const text = String(split ?? "");
  if (/^f\d+$/.test(text)) return +text.slice(1);
  return MODEL_FEATURE_NAMES.indexOf(text);
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

function predictXGBDelta(features, regimeState) {
  const xgb = modelBundle?.xgb;
  if (!modelBundle?.trained || !xgb || !Array.isArray(xgb.trees)) return 0;
  const vector = modelFeatureVector(features, regimeState);
  let result = +xgb.base_score || 0;
  for (const tree of xgb.trees) result += evaluateTree(tree, vector);
  return Number.isFinite(result) ? result : 0;
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

function regimeConfig() {
  const cfg = modelBundle?.shoe_regime_filter || {};
  const weights = cfg.likelihood_weights || {};
  return {
    n_particles: Math.max(1, Math.round(+cfg.n_particles || PF_DEFAULTS.n_particles)),
    state_dim: 1,
    Q_early: Math.max(1e-12, +cfg.Q_early || PF_DEFAULTS.Q_early),
    Q_late: Math.max(1e-12, +cfg.Q_late || PF_DEFAULTS.Q_late),
    early_round_end: Math.round(+cfg.early_round_end || PF_DEFAULTS.early_round_end),
    late_round_start: Math.round(+cfg.late_round_start || PF_DEFAULTS.late_round_start),
    R: Math.max(1e-12, +cfg.R || PF_DEFAULTS.R),
    resample_threshold: Math.max(1, +cfg.resample_threshold || PF_DEFAULTS.resample_threshold),
    resampling: "systematic",
    random_state: Math.round(+cfg.random_state || PF_DEFAULTS.random_state) >>> 0,
    state_clip: Math.max(0.1, +cfg.state_clip || PF_DEFAULTS.state_clip),
    likelihood_weights: {
      directionality: Number.isFinite(+weights.directionality) ? +weights.directionality : PF_DEFAULTS.likelihood_weights.directionality,
      residual_alignment: Number.isFinite(+weights.residual_alignment) ? +weights.residual_alignment : PF_DEFAULTS.likelihood_weights.residual_alignment,
      persistence: Number.isFinite(+weights.persistence) ? +weights.persistence : PF_DEFAULTS.likelihood_weights.persistence
    }
  };
}

function nextUniform(state) {
  state.rng_state = (Math.imul(1664525, state.rng_state >>> 0) + 1013904223) >>> 0;
  return (state.rng_state + 0.5) / 4294967296;
}

function gaussianRandom(state) {
  const u1 = Math.max(nextUniform(state), 1e-15);
  const u2 = nextUniform(state);
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function newRegimeState(shoeId = getShoeId()) {
  const cfg = regimeConfig();
  return {
    shoe_id: String(shoeId),
    updates: 0,
    rng_state: cfg.random_state >>> 0,
    last_alignment: null,
    last_observation: 0,
    last_measurements: {
      directionality: 0,
      residual_alignment: 0,
      persistence: 0
    },
    last_effective_q: cfg.Q_early,
    last_ess: cfg.n_particles,
    last_resampled: false,
    last_turbulence_break: false,
    particles: Array(cfg.n_particles).fill(0),
    weights: Array(cfg.n_particles).fill(1 / cfg.n_particles)
  };
}

function writeRegimeState(state) {
  try { localStorage.setItem(REGIME_STATE_KEY, JSON.stringify(state)); } catch (_) {}
}

function resetShoeRegimeFilter(shoeId = getShoeId()) {
  const state = newRegimeState(shoeId);
  writeRegimeState(state);
  return state;
}

function readRegimeState() {
  const cfg = regimeConfig();
  const shoeId = getShoeId();
  try {
    const state = JSON.parse(localStorage.getItem(REGIME_STATE_KEY) || "null");
    const valid = state
      && String(state.shoe_id || "") === String(shoeId)
      && Array.isArray(state.particles)
      && Array.isArray(state.weights)
      && state.particles.length === cfg.n_particles
      && state.weights.length === cfg.n_particles
      && Number.isFinite(+state.rng_state);
    if (valid) return state;
  } catch (_) {}
  return resetShoeRegimeFilter(shoeId);
}

function regimeEstimate(state = readRegimeState()) {
  const cfg = regimeConfig();
  let total = 0, weightSum = 0;
  for (let i = 0; i < state.particles.length; i++) {
    const w = +state.weights[i] || 0;
    total += (+state.particles[i] || 0) * w;
    weightSum += w;
  }
  return clip(weightSum > 0 ? total / weightSum : 0, -cfg.state_clip, cfg.state_clip);
}

function effectiveSampleSize(state) {
  let sumSq = 0;
  for (const value of state.weights) {
    const w = +value || 0;
    sumSq += w * w;
  }
  return sumSq > 0 ? 1 / sumSq : 0;
}

function systematicResample(state) {
  const n = state.particles.length;
  const cumulative = new Array(n);
  let running = 0;
  for (let i = 0; i < n; i++) {
    running += +state.weights[i] || 0;
    cumulative[i] = running;
  }
  cumulative[n - 1] = 1;
  const start = nextUniform(state) / n;
  const particles = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const position = start + i / n;
    while (j < n - 1 && position > cumulative[j]) j++;
    particles[i] = +state.particles[j] || 0;
  }
  state.particles = particles;
  state.weights = Array(n).fill(1 / n);
}

function shoeProgress(roundIndex, estimatedTotalHands) {
  const total = Math.max(2, +estimatedTotalHands || 60);
  return clip(((+roundIndex || 1) - 1) / (total - 1), 0, 1);
}

function effectiveProcessNoise(currentRound) {
  const cfg = regimeConfig();
  const current = +currentRound || 1;
  if (current < cfg.early_round_end) return cfg.Q_early;
  if (current > cfg.late_round_start) return cfg.Q_late;
  const span = Math.max(1, cfg.late_round_start - cfg.early_round_end);
  const ratio = clip((current - cfg.early_round_end) / span, 0, 1);
  return cfg.Q_early + (cfg.Q_late - cfg.Q_early) * ratio;
}

function buildRegimeMeasurements(actualB, corePB, state) {
  const actual = +actualB >= 0.5 ? 1 : 0;
  const pB = clip(+corePB || 0.5, 0, 1);
  const predictedB = pB > 0.5;
  const actualIsB = actual >= 0.5;

  const directionality = predictedB === actualIsB ? 1 : -1;
  const residual = actual - pB;
  const residualAlignment = clip(1 - 2 * Math.abs(residual), -1, 1);
  const lastAlignment = Number.isFinite(+state.last_alignment) ? +state.last_alignment : null;
  const turbulenceBreak = lastAlignment !== null && directionality !== lastAlignment;

  if (turbulenceBreak) {
    return {
      measurements: {
        directionality: 0,
        residual_alignment: 0,
        persistence: 0
      },
      alignment: directionality,
      turbulenceBreak,
      residual
    };
  }

  return {
    measurements: {
      directionality,
      residual_alignment: residualAlignment,
      persistence: lastAlignment === directionality ? directionality : 0
    },
    alignment: directionality,
    turbulenceBreak,
    residual
  };
}

function updateShoeRegimeFilter(actualB, corePB, roundIndex, estimatedTotalHands) {
  const cfg = regimeConfig();
  const state = readRegimeState();
  const obs = buildRegimeMeasurements(actualB, corePB, state);
  const measurements = obs.measurements;
  const w = cfg.likelihood_weights;

  if (obs.turbulenceBreak) {
    state.weights = Array(state.weights.length).fill(1 / state.weights.length);
  }

  let maxLog = -Infinity;
  const logs = new Array(state.particles.length);
  for (let i = 0; i < state.particles.length; i++) {
    const particle = +state.particles[i] || 0;
    const dDirection = measurements.directionality - particle;
    const dResidual = measurements.residual_alignment - particle;
    const dPersistence = measurements.persistence - particle;
    const weightedError =
      w.directionality * dDirection * dDirection
      + w.residual_alignment * dResidual * dResidual
      + w.persistence * dPersistence * dPersistence;
    const logLike = -0.5 * weightedError / cfg.R;
    logs[i] = logLike;
    if (logLike > maxLog) maxLog = logLike;
  }

  let total = 0;
  for (let i = 0; i < state.weights.length; i++) {
    const updated = (+state.weights[i] || 0) * Math.exp(logs[i] - maxLog);
    state.weights[i] = updated;
    total += updated;
  }
  if (!Number.isFinite(total) || total <= 0) {
    state.weights = Array(state.weights.length).fill(1 / state.weights.length);
  } else {
    for (let i = 0; i < state.weights.length; i++) state.weights[i] /= total;
  }

  const essBeforeResample = effectiveSampleSize(state);
  let resampled = false;
  if (essBeforeResample < cfg.resample_threshold) {
    systematicResample(state);
    resampled = true;
  }

  const qEff = effectiveProcessNoise((+roundIndex || 1) + 1);
  const std = Math.sqrt(Math.max(qEff, 1e-12));
  for (let i = 0; i < state.particles.length; i++) {
    state.particles[i] = clip(
      (+state.particles[i] || 0) + gaussianRandom(state) * std,
      -cfg.state_clip,
      cfg.state_clip
    );
  }

  state.updates = Math.max(0, +state.updates || 0) + 1;
  state.last_alignment = obs.alignment;
  state.last_measurements = measurements;
  state.last_observation = clip(
    w.directionality * measurements.directionality
      + w.residual_alignment * measurements.residual_alignment
      + w.persistence * measurements.persistence,
    -1,
    1
  );
  state.last_residual = obs.residual;
  state.last_turbulence_break = obs.turbulenceBreak;
  state.last_ess = essBeforeResample;
  state.last_resampled = resampled;
  state.last_effective_q = qEff;
  state.last_shoe_progress = shoeProgress(roundIndex, estimatedTotalHands);
  writeRegimeState(state);
  return state;
}

function shoeRegimeStateFeature() {
  return regimeEstimate(readRegimeState());
}

function classifyRegime(value) {
  const v = +value || 0;
  if (v >= 0.45) return "stable_aligned";
  if (v >= 0.15) return "forming_aligned";
  if (v <= -0.45) return "stable_opposed";
  if (v <= -0.15) return "degrading_opposed";
  return "turbulent_neutral";
}

function applyCorrection(seq, corePrediction) {
  const signal = corePrediction?.singleHazard || null;
  const features = buildFeatures(seq, corePrediction, signal);
  const regimeState = shoeRegimeStateFeature();
  const rawDelta = predictXGBDelta(features, regimeState);
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
        version: VERSION, active: false, modelLoaded, modelLoadError,
        coreDirection, corePB, regimeState, regimeClass: classifyRegime(regimeState),
        rawDelta: 0, delta: 0, finalPB: corePB, finalDirection: coreDirection,
        flipped: false, features
      }
    };
  }
  const flipped = direction !== coreDirection;
  return {
    ...corePrediction,
    direction,
    confidence,
    probabilities: { B: finalPB, P: finalPP },
    regime: flipped ? "Shoe-Regime 8D-XGB殘差修正換邊" : corePrediction.regime,
    residualBias: {
      version: VERSION, active: true, modelLoaded, modelLoadError,
      coreDirection, corePB, regimeState, regimeClass: classifyRegime(regimeState),
      rawDelta, delta, finalPB, finalDirection: direction, flipped, features
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

function rotateShoeId() {
  try {
    localStorage.removeItem(SHOE_KEY);
    localStorage.removeItem(PENDING_KEY);
    localStorage.removeItem(REGIME_STATE_KEY);
    localStorage.removeItem(LEGACY_PF_STATE_KEY);
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
  if (el("modePill")) el("modePill").textContent = p.residualBias?.active ? "Shoe-Regime 8D-XGB 修正完成" : "分析完成";
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
    regime_state: Number.isFinite(+residual.regimeState) ? +residual.regimeState : shoeRegimeStateFeature(),
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
  const residualTarget = actualB - corePB;
  const row = {
    schema_version: 3,
    shoe_id: String(pending.shoe_id || getShoeId()),
    created_at: +pending.created_at || Date.now(),
    history_fingerprint: String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: residualTarget,
    regime_state: Number.isFinite(+pending.regime_state) ? +pending.regime_state : 0,
    ...pending.features
  };
  const rows = readTrainingRows();
  const duplicate = rows.length && rows.at(-1)?.shoe_id === row.shoe_id && rows.at(-1)?.history_fingerprint === row.history_fingerprint;
  if (!duplicate) {
    rows.push(row);
    updateShoeRegimeFilter(actualB, corePB, +pending.features.round_index || 1, +pending.features.estimated_total_hands || getEstimatedTotalHands());
  }
  writeTrainingRows(rows);
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function rebuildShoeRegimeFilter(rows = readTrainingRows()) {
  const shoeId = getShoeId();
  resetShoeRegimeFilter(shoeId);
  for (const row of rows) {
    if (String(row?.shoe_id || "") !== String(shoeId)) continue;
    const actualB = Number.isFinite(+row?.actual_b) ? +row.actual_b : (String(row?.actual_outcome || "").toUpperCase() === "B" ? 1 : 0);
    updateShoeRegimeFilter(actualB, clip(+row?.core_p_b || 0.5), +row?.round_index || 1, +row?.estimated_total_hands || getEstimatedTotalHands());
  }
}

function rollbackTrainingIfNeeded() {
  const history = readHistory();
  const rows = readTrainingRows();
  if (!rows.length) return;
  const last = rows.at(-1);
  if (last?.shoe_id === getShoeId() && +last.round_index > history.length) {
    rows.pop();
    writeTrainingRows(rows);
    rebuildShoeRegimeFilter(rows);
  }
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({
    schema_version: 3,
    feature_names: FEATURE_NAMES,
    model_feature_names: MODEL_FEATURE_NAMES,
    feature_schema: "7D_UPSTREAM_PLUS_1D_SHOE_REGIME",
    rows: readTrainingRows()
  }, null, 2);
}

function downloadTrainingData() {
  const blob = new Blob([exportTrainingData()], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bgs_xgb_shoe_regime_8d_training_${Date.now()}.json`;
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
    if (!bundle || bundle.model_type !== "xgb_shoe_regime_feature_residual") throw new Error("invalid_model_bundle");
    const names = Array.isArray(bundle.feature_names) ? bundle.feature_names : [];
    if (names.join("|") !== FEATURE_NAMES.join("|")) throw new Error("upstream_feature_schema_mismatch");
    const modelNames = Array.isArray(bundle.model_feature_names) ? bundle.model_feature_names : [];
    if (modelNames.join("|") !== MODEL_FEATURE_NAMES.join("|")) throw new Error("model_feature_schema_mismatch");
    modelBundle = bundle;
    modelLoaded = true;
    readRegimeState();
    return bundle;
  } catch (error) {
    modelBundle = null;
    modelLoadError = String(error?.message || error || "model_load_failed");
    readRegimeState();
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
    modelFeatureNames: MODEL_FEATURE_NAMES,
    buildFeatures,
    sxMarkovPSame,
    applyCorrection,
    loadModel,
    setEstimatedTotalHands,
    getEstimatedTotalHands,
    exportTrainingData,
    downloadTrainingData,
    resetShoeRegimeFilter,
    updateShoeRegimeFilter,
    getShoeRegimeState: () => shoeRegimeStateFeature(),
    getShoeRegimeStatus: () => {
      const state = readRegimeState();
      const estimate = regimeEstimate(state);
      return {
        shoeId: state.shoe_id,
        updates: +state.updates || 0,
        estimate,
        regimeClass: classifyRegime(estimate),
        effectiveSampleSize: effectiveSampleSize(state),
        lastObservation: +state.last_observation || 0,
        lastEffectiveQ: +state.last_effective_q || regimeConfig().Q_early,
        lastShoeProgress: +state.last_shoe_progress || 0,\n        lastMeasurements: state.last_measurements || null,\n        lastResidual: Number.isFinite(+state.last_residual) ? +state.last_residual : null,\n        lastTurbulenceBreak: Boolean(state.last_turbulence_break),
        lastResampled: Boolean(state.last_resampled),
        config: regimeConfig()
      };
    },
    resetParticleFilter: resetShoeRegimeFilter,
    getParticleFilterEstimate: () => shoeRegimeStateFeature(),
    getParticleFilterStatus: () => {
      const state = readRegimeState();
      return { shoeId: state.shoe_id, updates: +state.updates || 0, estimate: regimeEstimate(state), semantics: "shoe_regime_state", config: regimeConfig() };
    },
    getTrainingCount: () => readTrainingRows().length,
    getModelStatus: () => ({
      loaded: modelLoaded,
      trained: Boolean(modelBundle?.trained),
      error: modelLoadError,
      featureSchema: "7D_UPSTREAM_PLUS_1D_SHOE_REGIME"
    })
  };
}

loadModel();
installUIOverride();
})();