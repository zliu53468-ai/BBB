(() => {
"use strict";

const CORE = (typeof window !== "undefined") ? window.__BGS256_CONTINUATION_TEST__ : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "XGB_PF_BASE_MARGIN_V1";
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
const MODEL_FEATURE_NAMES = [...FEATURE_NAMES];
const MAX_DELTA_DEFAULT = 0.10;
const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_xgb_residual_training_v1";
const PENDING_KEY = "bgs_xgb_residual_pending_v1";
const SHOE_KEY = "bgs_xgb_residual_shoe_id_v1";
const CUT_KEY = "bgs_xgb_estimated_total_hands_v1";
const SHOE_PF_STATE_KEY = "bgs_xgb_shoe_particle_filter_state_v1";
const LEGACY_REGIME_STATE_KEY = "bgs_xgb_shoe_regime_state_v1";
const LEGACY_PF_STATE_KEY = "bgs_xgb_particle_filter_state_v1";
const MAX_TRAINING_ROWS = 10000;
const PF_DEFAULTS = {
  n_particles: 1000,
  decks: 8,
  point_bins: 10,
  initial_point_counts: [128, 32, 32, 32, 32, 32, 32, 32, 32, 32],
  Q_early: 0.005,
  Q_late: 0.02,
  early_round_end: 15,
  late_round_start: 45,
  R: 0.25,
  resample_threshold: 500,
  resampling: "systematic",
  random_state: 42,
  pf_delta_clip: 0.10
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

function modelFeatureVector(features) {
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

function predictXGBDelta(features, pfDelta) {
  const xgb = modelBundle?.xgb;
  if (!modelBundle?.trained || !xgb || !Array.isArray(xgb.trees)) return 0;
  const vector = modelFeatureVector(features);
  let totalDelta = Number.isFinite(+pfDelta) ? +pfDelta : 0;
  for (const tree of xgb.trees) totalDelta += evaluateTree(tree, vector);
  return Number.isFinite(totalDelta) ? totalDelta : 0;
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

function shoePFConfig() {
  const cfg = modelBundle?.shoe_particle_filter || {};
  const initial = Array.isArray(cfg.initial_point_counts) && cfg.initial_point_counts.length === 10
    ? cfg.initial_point_counts.map(v => Math.max(0, Math.round(+v || 0)))
    : PF_DEFAULTS.initial_point_counts.slice();
  return {
    n_particles: Math.max(1, Math.round(+cfg.n_particles || PF_DEFAULTS.n_particles)),
    decks: 8,
    point_bins: 10,
    initial_point_counts: initial,
    Q_early: Math.max(1e-12, +cfg.Q_early || PF_DEFAULTS.Q_early),
    Q_late: Math.max(1e-12, +cfg.Q_late || PF_DEFAULTS.Q_late),
    early_round_end: Math.round(+cfg.early_round_end || PF_DEFAULTS.early_round_end),
    late_round_start: Math.round(+cfg.late_round_start || PF_DEFAULTS.late_round_start),
    R: Math.max(1e-12, +cfg.R || PF_DEFAULTS.R),
    resample_threshold: Math.max(1, +cfg.resample_threshold || PF_DEFAULTS.resample_threshold),
    resampling: "systematic",
    random_state: Math.round(+cfg.random_state || PF_DEFAULTS.random_state) >>> 0,
    pf_delta_clip: Math.min(0.10, Math.max(0.001, +cfg.pf_delta_clip || PF_DEFAULTS.pf_delta_clip))
  };
}

function nextUniform(state) {
  state.rng_state = (Math.imul(1664525, state.rng_state >>> 0) + 1013904223) >>> 0;
  return (state.rng_state + 0.5) / 4294967296;
}

function newShoePFState(shoeId = getShoeId()) {
  const cfg = shoePFConfig();
  const particles = new Array(cfg.n_particles);
  for (let i = 0; i < cfg.n_particles; i++) particles[i] = cfg.initial_point_counts.slice();
  return {
    shoe_id: String(shoeId),
    updates: 0,
    rng_state: cfg.random_state >>> 0,
    particles,
    weights: Array(cfg.n_particles).fill(1 / cfg.n_particles),
    pf_delta: 0,
    last_effective_q: cfg.Q_early,
    last_ess: cfg.n_particles,
    last_resampled: false,
    last_residual: 0,
    last_observed_outcome: 0
  };
}

function writeShoePFState(state) {
  try { localStorage.setItem(SHOE_PF_STATE_KEY, JSON.stringify(state)); } catch (_) {}
}

function resetShoeParticleFilter(shoeId = getShoeId()) {
  const state = newShoePFState(shoeId);
  writeShoePFState(state);
  return state;
}

function readShoePFState() {
  const cfg = shoePFConfig();
  const shoeId = getShoeId();
  try {
    const state = JSON.parse(localStorage.getItem(SHOE_PF_STATE_KEY) || "null");
    const valid = state
      && String(state.shoe_id || "") === String(shoeId)
      && Array.isArray(state.particles)
      && Array.isArray(state.weights)
      && state.particles.length === cfg.n_particles
      && state.weights.length === cfg.n_particles
      && state.particles.every(p => Array.isArray(p) && p.length === 10)
      && Number.isFinite(+state.rng_state);
    if (valid) return state;
  } catch (_) {}
  return resetShoeParticleFilter(shoeId);
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
    particles[i] = state.particles[j].slice();
  }
  state.particles = particles;
  state.weights = Array(n).fill(1 / n);
}

function effectiveProcessNoise(currentRound) {
  const cfg = shoePFConfig();
  const current = +currentRound || 1;
  if (current < cfg.early_round_end) return cfg.Q_early;
  if (current > cfg.late_round_start) return cfg.Q_late;
  const span = Math.max(1, cfg.late_round_start - cfg.early_round_end);
  const ratio = clip((current - cfg.early_round_end) / span, 0, 1);
  return cfg.Q_early + (cfg.Q_late - cfg.Q_early) * ratio;
}

function drawPoint(counts, state) {
  let total = 0;
  for (const count of counts) total += +count || 0;
  if (total <= 0) return 0;
  const threshold = nextUniform(state) * total;
  let running = 0;
  for (let point = 0; point < 10; point++) {
    running += +counts[point] || 0;
    if (threshold < running) {
      counts[point] = Math.max(0, (+counts[point] || 0) - 1);
      return point;
    }
  }
  counts[9] = Math.max(0, (+counts[9] || 0) - 1);
  return 9;
}

function bankerDraws(bankerTotal, playerThird) {
  if (playerThird === null) return bankerTotal <= 5;
  if (bankerTotal <= 2) return true;
  if (bankerTotal === 3) return playerThird !== 8;
  if (bankerTotal === 4) return playerThird >= 2 && playerThird <= 7;
  if (bankerTotal === 5) return playerThird >= 4 && playerThird <= 7;
  if (bankerTotal === 6) return playerThird >= 6 && playerThird <= 7;
  return false;
}

function simulateVirtualRound(counts, state) {
  let totalRemaining = 0;
  for (const count of counts) totalRemaining += +count || 0;
  if (totalRemaining < 6) return { sign: 0, playerTotal: 0, bankerTotal: 0 };

  const player = [drawPoint(counts, state)];
  const banker = [drawPoint(counts, state)];
  player.push(drawPoint(counts, state));
  banker.push(drawPoint(counts, state));

  let playerTotal = (player[0] + player[1]) % 10;
  let bankerTotal = (banker[0] + banker[1]) % 10;
  const natural = playerTotal === 8 || playerTotal === 9 || bankerTotal === 8 || bankerTotal === 9;

  let playerThird = null;
  if (!natural) {
    if (playerTotal <= 5) {
      playerThird = drawPoint(counts, state);
      player.push(playerThird);
      playerTotal = player.reduce((a, b) => a + b, 0) % 10;
    }
    if (bankerDraws(bankerTotal, playerThird)) {
      banker.push(drawPoint(counts, state));
      bankerTotal = banker.reduce((a, b) => a + b, 0) % 10;
    }
  }

  const sign = bankerTotal > playerTotal ? 1 : playerTotal > bankerTotal ? -1 : 0;
  return { sign, playerTotal, bankerTotal };
}

function proposalLikelihood(simulated, actualB, corePB) {
  const cfg = shoePFConfig();
  const actual = +actualB >= 0.5 ? 1 : 0;
  const observedSign = actual >= 0.5 ? 1 : -1;
  const residual = actual - clip(+corePB || 0.5, 0, 1);
  const observedStrength = 0.5 + 0.5 * Math.min(1, Math.abs(residual));
  const targetScore = observedSign * observedStrength;

  let simulatedScore = 0;
  if (simulated.sign !== 0) {
    const pointMargin = Math.abs(simulated.bankerTotal - simulated.playerTotal) / 9;
    simulatedScore = simulated.sign * (0.5 + 0.5 * pointMargin);
  }

  const error = targetScore - simulatedScore;
  return Math.exp(-0.5 * error * error / cfg.R);
}

function rejuvenateParticles(state, qEff) {
  const cfg = shoePFConfig();
  const nMutations = Math.ceil(state.particles.length * clip(qEff, 0, 1));
  for (let m = 0; m < nMutations; m++) {
    const idx = Math.min(state.particles.length - 1, Math.floor(nextUniform(state) * state.particles.length));
    const counts = state.particles[idx];
    const sources = [];
    const destinations = [];
    for (let point = 0; point < 10; point++) {
      if ((+counts[point] || 0) > 0) sources.push(point);
      if ((+counts[point] || 0) < cfg.initial_point_counts[point]) destinations.push(point);
    }
    if (!sources.length || !destinations.length) continue;
    const src = sources[Math.min(sources.length - 1, Math.floor(nextUniform(state) * sources.length))];
    const validDestinations = destinations.filter(point => point !== src);
    if (!validDestinations.length) continue;
    const dst = validDestinations[Math.min(validDestinations.length - 1, Math.floor(nextUniform(state) * validDestinations.length))];
    counts[src] -= 1;
    counts[dst] += 1;
  }
}

function forecastPFDelta(state) {
  const cfg = shoePFConfig();
  let bankerMass = 0;
  let playerMass = 0;
  for (let i = 0; i < state.particles.length; i++) {
    const counts = state.particles[i].slice();
    const simulated = simulateVirtualRound(counts, state);
    const weight = +state.weights[i] || 0;
    if (simulated.sign > 0) bankerMass += weight;
    else if (simulated.sign < 0) playerMass += weight;
  }
  const decisiveMass = bankerMass + playerMass;
  if (decisiveMass <= 1e-12) return 0;
  const physicalBias = (bankerMass - playerMass) / decisiveMass;
  return clip(0.10 * physicalBias, -cfg.pf_delta_clip, cfg.pf_delta_clip);
}

function shoePFDelta() {
  const state = readShoePFState();
  const cfg = shoePFConfig();
  return clip(+state.pf_delta || 0, -cfg.pf_delta_clip, cfg.pf_delta_clip);
}

function updateShoeParticleFilter(actualB, corePB, roundIndex) {
  const state = readShoePFState();
  const logs = new Array(state.particles.length);
  let maxLog = -Infinity;

  for (let i = 0; i < state.particles.length; i++) {
    const counts = state.particles[i].slice();
    const simulated = simulateVirtualRound(counts, state);
    state.particles[i] = counts;
    const likelihood = proposalLikelihood(simulated, actualB, corePB);
    const logLike = Math.log(Math.max(likelihood, 1e-300));
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

  const ess = effectiveSampleSize(state);
  state.last_ess = ess;
  state.last_resampled = false;
  if (ess < shoePFConfig().resample_threshold) {
    systematicResample(state);
    state.last_resampled = true;
  }

  const qEff = effectiveProcessNoise(roundIndex);
  rejuvenateParticles(state, qEff);
  state.pf_delta = forecastPFDelta(state);
  state.last_effective_q = qEff;
  state.last_residual = (+actualB >= 0.5 ? 1 : 0) - clip(+corePB || 0.5, 0, 1);
  state.last_observed_outcome = +actualB >= 0.5 ? 1 : 0;
  state.updates = Math.max(0, +state.updates || 0) + 1;
  writeShoePFState(state);
  return state;
}

function applyCorrection(seq, corePrediction) {
  const signal = corePrediction?.singleHazard || null;
  const features = buildFeatures(seq, corePrediction, signal);
  const pfDelta = shoePFDelta();
  const totalDelta = predictXGBDelta(features, pfDelta);
  const maxDelta = clip(modelBundle?.max_delta ?? MAX_DELTA_DEFAULT, 0, 0.10);
  const delta = clip(totalDelta, -maxDelta, maxDelta);
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
        coreDirection, corePB, pfDelta,
        totalDelta: 0, delta: 0, finalPB: corePB, finalDirection: coreDirection,
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
    regime: flipped ? "PF Base-Margin 7D-XGB殘差修正換邊" : corePrediction.regime,
    residualBias: {
      version: VERSION, active: true, modelLoaded, modelLoadError,
      coreDirection, corePB, pfDelta,
      totalDelta, delta, finalPB, finalDirection: direction, flipped, features
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
    localStorage.removeItem(SHOE_PF_STATE_KEY);
    localStorage.removeItem(LEGACY_REGIME_STATE_KEY);
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
  if (el("modePill")) el("modePill").textContent = p.residualBias?.active ? "PF Base-Margin 7D-XGB 修正完成" : "分析完成";
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
    pf_delta: Number.isFinite(+residual.pfDelta) ? +residual.pfDelta : shoePFDelta(),
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
    schema_version: 5,
    shoe_id: String(pending.shoe_id || getShoeId()),
    created_at: +pending.created_at || Date.now(),
    history_fingerprint: String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: residualTarget,
    pf_delta: Number.isFinite(+pending.pf_delta) ? +pending.pf_delta : 0,
    ...pending.features
  };

  const rows = readTrainingRows();
  const duplicate = rows.length
    && rows.at(-1)?.shoe_id === row.shoe_id
    && rows.at(-1)?.history_fingerprint === row.history_fingerprint;

  if (!duplicate) {
    rows.push(row);
    updateShoeParticleFilter(actualB, corePB, +pending.features.round_index || 1);
  }
  writeTrainingRows(rows);
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function rebuildShoeParticleFilter(rows = readTrainingRows()) {
  const shoeId = getShoeId();
  resetShoeParticleFilter(shoeId);
  for (const row of rows) {
    if (String(row?.shoe_id || "") !== String(shoeId)) continue;
    const actualB = Number.isFinite(+row?.actual_b)
      ? +row.actual_b
      : (String(row?.actual_outcome || "").toUpperCase() === "B" ? 1 : 0);
    updateShoeParticleFilter(
      actualB,
      clip(+row?.core_p_b || 0.5),
      +row?.round_index || 1
    );
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
    rebuildShoeParticleFilter(rows);
  }
  try { localStorage.removeItem(PENDING_KEY); } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({
    schema_version: 5,
    feature_names: FEATURE_NAMES,
    model_feature_names: MODEL_FEATURE_NAMES,
    feature_schema: "7D_WITH_PF_BASE_MARGIN",
    rows: readTrainingRows()
  }, null, 2);
}

function downloadTrainingData() {
  const blob = new Blob([exportTrainingData()], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bgs_xgb_pf_base_margin_7d_training_${Date.now()}.json`;
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
    if (!bundle || bundle.model_type !== "xgb_pf_base_margin_residual") throw new Error("invalid_model_bundle");
    const names = Array.isArray(bundle.feature_names) ? bundle.feature_names : [];
    if (names.join("|") !== FEATURE_NAMES.join("|")) throw new Error("upstream_feature_schema_mismatch");
    const modelNames = Array.isArray(bundle.model_feature_names) ? bundle.model_feature_names : [];
    if (modelNames.join("|") !== MODEL_FEATURE_NAMES.join("|")) throw new Error("model_feature_schema_mismatch");
    modelBundle = bundle;
    modelLoaded = true;
    readShoePFState();
    return bundle;
  } catch (error) {
    modelBundle = null;
    modelLoadError = String(error?.message || error || "model_load_failed");
    readShoePFState();
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
    resetShoeParticleFilter,
    updateShoeParticleFilter,
    getPFDelta: () => shoePFDelta(),
    getShoeParticleFilterStatus: () => {
      const state = readShoePFState();
      return {
        shoeId: state.shoe_id,
        updates: +state.updates || 0,
        pfDelta: shoePFDelta(),
        effectiveSampleSize: effectiveSampleSize(state),
        lastResidual: Number.isFinite(+state.last_residual) ? +state.last_residual : null,
        lastEffectiveQ: +state.last_effective_q || shoePFConfig().Q_early,
        lastResampled: Boolean(state.last_resampled),
        config: shoePFConfig()
      };
    },
    resetParticleFilter: resetShoeParticleFilter,
    getParticleFilterEstimate: () => shoePFDelta(),
    getParticleFilterStatus: () => {
      const state = readShoePFState();
      return {
        shoeId: state.shoe_id,
        updates: +state.updates || 0,
        estimate: shoePFDelta(),
        semantics: "pf_delta_base_margin",
        config: shoePFConfig()
      };
    },
    getTrainingCount: () => readTrainingRows().length,
    getModelStatus: () => ({
      loaded: modelLoaded,
      trained: Boolean(modelBundle?.trained),
      error: modelLoadError,
      featureSchema: "7D_WITH_PF_BASE_MARGIN"
    })
  };
}

loadModel();
installUIOverride();
})();