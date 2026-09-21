(() => {
"use strict";

const CORE = (typeof window !== "undefined")
  ? window.__BGS256_CONTINUATION_TEST__
  : null;
if (!CORE || typeof CORE.hazardChoose !== "function") return;

const VERSION = "XGB_ANOMALY_BRAKE_11D_V1";
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
const PHYSICAL_FEATURE_NAMES = [
  "pred_card_count",
  "pred_banker_point",
  "pred_player_point",
  "anomaly_score"
];
const MODEL_FEATURE_NAMES = [
  ...FEATURE_NAMES,
  ...PHYSICAL_FEATURE_NAMES
];

const MAX_DELTA_DEFAULT = 0.10;
const BRAKE_START_DEFAULT = 0.35;
const BRAKE_FULL_DEFAULT = 0.90;

const STORAGE_KEY = "bgs256d_short_x_dynamic_v23";
const TRAINING_KEY = "bgs_xgb_residual_training_v1";
const PENDING_KEY = "bgs_xgb_residual_pending_v1";
const SHOE_KEY = "bgs_xgb_residual_shoe_id_v1";
const CUT_KEY = "bgs_xgb_estimated_total_hands_v1";
const SHOE_PF_STATE_KEY = "bgs_xgb_anomaly_brake_particle_filter_state_v5";
const PHYSICAL_OBS_KEY = "bgs_xgb_physical_observation_v1";
const LEGACY_SHOE_PF_STATE_KEYS = [
  "bgs_xgb_blind_physical_particle_filter_state_v4",
  "bgs_xgb_enhanced_shoe_particle_filter_state_v3",
  "bgs_xgb_shoe_particle_filter_state_v2",
  "bgs_xgb_shoe_particle_filter_state_v1",
  "bgs_xgb_shoe_regime_state_v1",
  "bgs_xgb_particle_filter_state_v1",
  "bgs_xgb_transformer_window_v1"
];
const MAX_TRAINING_ROWS = 10000;

const PF_DEFAULTS = {
  n_particles: 1000,
  decks: 8,
  point_bins: 10,
  initial_point_counts: [128, 32, 32, 32, 32, 32, 32, 32, 32, 32],
  Q_early: 0.005,
  Q_late: 0.03,
  early_round_end: 15,
  late_round_start: 45,
  R: 0.25,
  resample_threshold: 500,
  resampling: "systematic",
  random_state: 42,
  likelihood_weights: {
    outcome: 0.55,
    total_cards: 0.12,
    points: 0.18,
    core_residual: 0.15
  },
  core_forecast_strength: 0.35,
  persistence_boost: 1.15,
  anomaly: {
    long_run_min: 3,
    long_run_break_floor: 0.95,
    short_run_break: 0.72,
    alternation_floor: 0.72,
    core_miss_base: 0.30,
    core_miss_confidence_gain: 0.35,
    stable_decay: 0.55,
    posterior_collapse_max_mix: 0.92,
    likelihood_precision_floor: 0.15
  }
};

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
  if (parsed < 40 || parsed > 90) {
    throw new Error("estimatedTotalHands must be between 40 and 90");
  }
  try { localStorage.setItem(CUT_KEY, String(parsed)); } catch (_) {}
  return parsed;
}

function buildFeatures(seq, corePrediction, signal = null) {
  const probabilities = corePrediction?.probabilities || {};
  const corePB = clip(+probabilities.B || 0.5, 0, 1);
  const roundIndex = Math.max(1, Math.min(70, seq.length + 1));
  const estimatedTotalHands = getEstimatedTotalHands();
  const remainingRatio = clip(
    (estimatedTotalHands - (roundIndex - 1))
    / Math.max(1, estimatedTotalHands)
  );
  const stage = Number.isFinite(+signal?.state?.length)
    ? +signal.state.length
    : currentStage(seq);
  const depth = Number.isFinite(+signal?.depth?.depth)
    ? +signal.depth.depth
    : currentDepth(seq);

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

function modelFeatureVector(features, pf4) {
  const vector = FEATURE_NAMES.map(name => {
    const value = +features[name];
    return Number.isFinite(value) ? value : 0;
  });
  const physical = Array.isArray(pf4)
    ? pf4
    : [4.8, 4.5, 4.5, 0.0];
  for (let i = 0; i < 4; i++) {
    const value = +physical[i];
    vector.push(Number.isFinite(value) ? value : 0);
  }
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
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
      return +node.leaf || 0;
    }
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

function predictXGBDelta(features, pf4) {
  const xgb = modelBundle?.xgb;
  if (!modelBundle?.trained || !xgb || !Array.isArray(xgb.trees)) return 0;
  const vector = modelFeatureVector(features, pf4);
  let result = +xgb.base_score || 0;
  for (const tree of xgb.trees) result += evaluateTree(tree, vector);
  return Number.isFinite(result) ? result : 0;
}

function anomalyBrakeFactor(anomalyScore) {
  const cfg = modelBundle?.anomaly_brake || {};
  const start = Number.isFinite(+cfg.start)
    ? +cfg.start
    : BRAKE_START_DEFAULT;
  const full = Number.isFinite(+cfg.full)
    ? +cfg.full
    : BRAKE_FULL_DEFAULT;
  const score = clip(anomalyScore, 0, 1);
  if (score <= start) return 1;
  if (score >= full) return 0;
  const t = clip((score - start) / Math.max(1e-12, full - start), 0, 1);
  const smooth = t * t * (3 - 2 * t);
  return clip(1 - smooth, 0, 1);
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
  const weights = cfg.likelihood_weights || {};
  const anomaly = cfg.anomaly || {};
  const initial = Array.isArray(cfg.initial_point_counts)
    && cfg.initial_point_counts.length === 10
    ? cfg.initial_point_counts.map(v => Math.max(0, Math.round(+v || 0)))
    : PF_DEFAULTS.initial_point_counts.slice();

  return {
    n_particles: Math.max(
      1,
      Math.round(+cfg.n_particles || PF_DEFAULTS.n_particles)
    ),
    decks: 8,
    point_bins: 10,
    initial_point_counts: initial,
    Q_early: Math.max(1e-12, +cfg.Q_early || PF_DEFAULTS.Q_early),
    Q_late: Math.max(1e-12, +cfg.Q_late || PF_DEFAULTS.Q_late),
    early_round_end: Math.round(
      +cfg.early_round_end || PF_DEFAULTS.early_round_end
    ),
    late_round_start: Math.round(
      +cfg.late_round_start || PF_DEFAULTS.late_round_start
    ),
    R: Math.max(1e-12, +cfg.R || PF_DEFAULTS.R),
    resample_threshold: Math.max(
      1,
      +cfg.resample_threshold || PF_DEFAULTS.resample_threshold
    ),
    random_state: Math.round(
      +cfg.random_state || PF_DEFAULTS.random_state
    ) >>> 0,
    likelihood_weights: {
      outcome: Number.isFinite(+weights.outcome)
        ? +weights.outcome
        : PF_DEFAULTS.likelihood_weights.outcome,
      total_cards: Number.isFinite(+weights.total_cards)
        ? +weights.total_cards
        : PF_DEFAULTS.likelihood_weights.total_cards,
      points: Number.isFinite(+weights.points)
        ? +weights.points
        : PF_DEFAULTS.likelihood_weights.points,
      core_residual: Number.isFinite(+weights.core_residual)
        ? +weights.core_residual
        : PF_DEFAULTS.likelihood_weights.core_residual
    },
    core_forecast_strength: Number.isFinite(+cfg.core_forecast_strength)
      ? +cfg.core_forecast_strength
      : PF_DEFAULTS.core_forecast_strength,
    persistence_boost: Number.isFinite(+cfg.persistence_boost)
      ? +cfg.persistence_boost
      : PF_DEFAULTS.persistence_boost,
    anomaly: {
      long_run_min: Math.max(
        2,
        Math.round(
          +anomaly.long_run_min
          || PF_DEFAULTS.anomaly.long_run_min
        )
      ),
      long_run_break_floor: Number.isFinite(+anomaly.long_run_break_floor)
        ? +anomaly.long_run_break_floor
        : PF_DEFAULTS.anomaly.long_run_break_floor,
      short_run_break: Number.isFinite(+anomaly.short_run_break)
        ? +anomaly.short_run_break
        : PF_DEFAULTS.anomaly.short_run_break,
      alternation_floor: Number.isFinite(+anomaly.alternation_floor)
        ? +anomaly.alternation_floor
        : PF_DEFAULTS.anomaly.alternation_floor,
      core_miss_base: Number.isFinite(+anomaly.core_miss_base)
        ? +anomaly.core_miss_base
        : PF_DEFAULTS.anomaly.core_miss_base,
      core_miss_confidence_gain: Number.isFinite(+anomaly.core_miss_confidence_gain)
        ? +anomaly.core_miss_confidence_gain
        : PF_DEFAULTS.anomaly.core_miss_confidence_gain,
      stable_decay: Number.isFinite(+anomaly.stable_decay)
        ? +anomaly.stable_decay
        : PF_DEFAULTS.anomaly.stable_decay,
      posterior_collapse_max_mix: Number.isFinite(+anomaly.posterior_collapse_max_mix)
        ? +anomaly.posterior_collapse_max_mix
        : PF_DEFAULTS.anomaly.posterior_collapse_max_mix,
      likelihood_precision_floor: Number.isFinite(+anomaly.likelihood_precision_floor)
        ? +anomaly.likelihood_precision_floor
        : PF_DEFAULTS.anomaly.likelihood_precision_floor
    }
  };
}

function nextUniform(state) {
  state.rng_state = (
    Math.imul(1664525, state.rng_state >>> 0)
    + 1013904223
  ) >>> 0;
  return (state.rng_state + 0.5) / 4294967296;
}

function newShoePFState(shoeId = getShoeId()) {
  const cfg = shoePFConfig();
  const particles = new Array(cfg.n_particles);
  for (let i = 0; i < cfg.n_particles; i++) {
    particles[i] = cfg.initial_point_counts.slice();
  }
  return {
    shoe_id: String(shoeId),
    updates: 0,
    rng_state: cfg.random_state >>> 0,
    particles,
    weights: Array(cfg.n_particles).fill(1 / cfg.n_particles),
    last_effective_q: cfg.Q_early,
    last_ess: cfg.n_particles,
    last_resampled: false,
    last_core_alignment: null,
    last_turbulence_break: false,
    recent_outcomes: [],
    anomaly_score: 0
  };
}

function writeShoePFState(state) {
  try {
    localStorage.setItem(
      SHOE_PF_STATE_KEY,
      JSON.stringify(state)
    );
  } catch (_) {}
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
    const state = JSON.parse(
      localStorage.getItem(SHOE_PF_STATE_KEY) || "null"
    );
    const valid = state
      && String(state.shoe_id || "") === String(shoeId)
      && Array.isArray(state.particles)
      && Array.isArray(state.weights)
      && Array.isArray(state.recent_outcomes)
      && state.particles.length === cfg.n_particles
      && state.weights.length === cfg.n_particles
      && state.particles.every(
        p => Array.isArray(p) && p.length === 10
      )
      && Number.isFinite(+state.rng_state)
      && Number.isFinite(+state.anomaly_score);
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
  const span = Math.max(
    1,
    cfg.late_round_start - cfg.early_round_end
  );
  const ratio = clip(
    (current - cfg.early_round_end) / span,
    0,
    1
  );
  return cfg.Q_early + (cfg.Q_late - cfg.Q_early) * ratio;
}

function runLength(values) {
  if (!values.length) return 0;
  const side = values[values.length - 1];
  let run = 1;
  for (let i = values.length - 2; i >= 0; i--) {
    if (+values[i] !== +side) break;
    run++;
  }
  return run;
}

function alternationStrength(values) {
  const tail = values.slice(-4);
  if (tail.length < 3) return 0;
  let switches = 0;
  for (let i = 1; i < tail.length; i++) {
    if (+tail[i] !== +tail[i - 1]) switches++;
  }
  return switches / Math.max(1, tail.length - 1);
}

function detectAnomaly(state, actualB, corePB) {
  const cfg = shoePFConfig();
  const a = cfg.anomaly;
  const actualSign = +actualB >= 0.5 ? 1 : -1;
  const history = Array.isArray(state.recent_outcomes)
    ? state.recent_outcomes
    : [];
  const priorRun = runLength(history);
  let structuralBreak = 0;

  if (history.length && actualSign !== +history[history.length - 1]) {
    if (priorRun >= a.long_run_min) {
      const runExcess = clip(
        (priorRun - a.long_run_min) / 4,
        0,
        1
      );
      structuralBreak = Math.min(
        1,
        a.long_run_break_floor + 0.05 * runExcess
      );
    } else if (priorRun >= 2) {
      structuralBreak = a.short_run_break;
    }
  }

  const candidateTail = [
    ...history.slice(-3),
    actualSign
  ];
  const alternation = alternationStrength(candidateTail);
  const alternationScore = alternation >= (2 / 3)
    ? a.alternation_floor * alternation
    : 0;

  const coreDirection = clip(corePB, 0, 1) > 0.5 ? 1 : -1;
  const coreConfidence = Math.abs(2 * clip(corePB, 0, 1) - 1);
  const coreMiss = coreDirection !== actualSign
    ? a.core_miss_base
      + a.core_miss_confidence_gain * coreConfidence
    : 0;

  const decayedPrevious = clip(
    state.anomaly_score || 0,
    0,
    1
  ) * a.stable_decay;

  return clip(
    Math.max(
      structuralBreak,
      alternationScore,
      coreMiss,
      decayedPrevious
    ),
    0,
    1
  );
}

function applyAnomalyCollapse(state, anomalyScore) {
  const cfg = shoePFConfig();
  const maxMix = cfg.anomaly.posterior_collapse_max_mix;
  const mix = Math.min(
    maxMix,
    maxMix * anomalyScore * anomalyScore
  );
  if (mix <= 0) return;

  const uniform = 1 / state.weights.length;
  let total = 0;
  for (let i = 0; i < state.weights.length; i++) {
    state.weights[i] =
      (1 - mix) * (+state.weights[i] || 0)
      + mix * uniform;
    total += state.weights[i];
  }
  if (!Number.isFinite(total) || total <= 0) {
    state.weights = Array(state.weights.length).fill(uniform);
  } else {
    for (let i = 0; i < state.weights.length; i++) {
      state.weights[i] /= total;
    }
  }
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
      counts[point] = Math.max(
        0,
        (+counts[point] || 0) - 1
      );
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
  if (bankerTotal === 4) {
    return playerThird >= 2 && playerThird <= 7;
  }
  if (bankerTotal === 5) {
    return playerThird >= 4 && playerThird <= 7;
  }
  if (bankerTotal === 6) {
    return playerThird >= 6 && playerThird <= 7;
  }
  return false;
}

function simulateVirtualRound(counts, state) {
  let totalRemaining = 0;
  for (const count of counts) totalRemaining += +count || 0;
  if (totalRemaining < 6) {
    return {
      sign: 0,
      playerTotal: 0,
      bankerTotal: 0,
      playerCards: 0,
      bankerCards: 0,
      totalCards: 0
    };
  }

  const player = [drawPoint(counts, state)];
  const banker = [drawPoint(counts, state)];
  player.push(drawPoint(counts, state));
  banker.push(drawPoint(counts, state));

  let playerTotal = (player[0] + player[1]) % 10;
  let bankerTotal = (banker[0] + banker[1]) % 10;
  const natural =
    playerTotal === 8
    || playerTotal === 9
    || bankerTotal === 8
    || bankerTotal === 9;

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

  const sign = bankerTotal > playerTotal
    ? 1
    : playerTotal > bankerTotal
      ? -1
      : 0;

  return {
    sign,
    playerTotal,
    bankerTotal,
    playerCards: player.length,
    bankerCards: banker.length,
    totalCards: player.length + banker.length
  };
}

function particleLogLikelihood(
  simulated,
  actualB,
  corePB,
  physicalObservation,
  persistenceMultiplier,
  anomalyScore
) {
  const cfg = shoePFConfig();
  const actual = +actualB >= 0.5 ? 1 : 0;
  const observedSign = actual >= 0.5 ? 1 : -1;

  const observedTotalCards = [4, 5, 6].includes(
    +physicalObservation?.totalCards
  )
    ? +physicalObservation.totalCards
    : null;

  const observedPlayerPoint =
    physicalObservation?.playerPoint !== null
    && physicalObservation?.playerPoint !== undefined
    && Number.isInteger(+physicalObservation.playerPoint)
    && +physicalObservation.playerPoint >= 0
    && +physicalObservation.playerPoint <= 9
      ? +physicalObservation.playerPoint
      : null;

  const observedBankerPoint =
    physicalObservation?.bankerPoint !== null
    && physicalObservation?.bankerPoint !== undefined
    && Number.isInteger(+physicalObservation.bankerPoint)
    && +physicalObservation.bankerPoint >= 0
    && +physicalObservation.bankerPoint <= 9
      ? +physicalObservation.bankerPoint
      : null;

  const w = cfg.likelihood_weights;
  let weightedError =
    w.outcome * Math.pow(
      (observedSign - simulated.sign) / 2,
      2
    );
  let activeWeight = w.outcome;

  if (observedTotalCards !== null) {
    const e = (
      observedTotalCards
      - simulated.totalCards
    ) / 2;
    weightedError += w.total_cards * e * e;
    activeWeight += w.total_cards;
  }

  if (
    observedPlayerPoint !== null
    && observedBankerPoint !== null
  ) {
    const pe = (
      observedPlayerPoint
      - simulated.playerTotal
    ) / 9;
    const be = (
      observedBankerPoint
      - simulated.bankerTotal
    ) / 9;
    weightedError +=
      w.points * 0.5 * (pe * pe + be * be);
    activeWeight += w.points;
  }

  const residual = actual - clip(corePB, 0, 1);
  const coreTarget = clip(2 * residual, -1, 1);
  const pointMargin = simulated.sign === 0
    ? 0
    : Math.abs(
        simulated.bankerTotal
        - simulated.playerTotal
      ) / 9;
  const simulatedSupport = simulated.sign === 0
    ? 0
    : simulated.sign * (0.5 + 0.5 * pointMargin);
  const coreError = coreTarget - simulatedSupport;
  weightedError +=
    w.core_residual * coreError * coreError;
  activeWeight += w.core_residual;

  const precision = Math.max(
    cfg.anomaly.likelihood_precision_floor,
    1 - 0.85 * anomalyScore * anomalyScore
  );

  return (
    -0.5
    * persistenceMultiplier
    * precision
    * (weightedError / Math.max(activeWeight, 1e-12))
    / cfg.R
  );
}

function rejuvenateParticles(state, qEff) {
  const cfg = shoePFConfig();
  const nMutations = Math.ceil(
    state.particles.length * clip(qEff, 0, 1)
  );

  for (let m = 0; m < nMutations; m++) {
    const idx = Math.min(
      state.particles.length - 1,
      Math.floor(
        nextUniform(state) * state.particles.length
      )
    );
    const counts = state.particles[idx];
    const sources = [];
    const destinations = [];

    for (let point = 0; point < 10; point++) {
      if ((+counts[point] || 0) > 0) sources.push(point);
      if (
        (+counts[point] || 0)
        < cfg.initial_point_counts[point]
      ) {
        destinations.push(point);
      }
    }
    if (!sources.length || !destinations.length) continue;

    const src = sources[
      Math.min(
        sources.length - 1,
        Math.floor(nextUniform(state) * sources.length)
      )
    ];
    const validDestinations = destinations.filter(
      point => point !== src
    );
    if (!validDestinations.length) continue;

    const dst = validDestinations[
      Math.min(
        validDestinations.length - 1,
        Math.floor(
          nextUniform(state) * validDestinations.length
        )
      )
    ];
    counts[src] -= 1;
    counts[dst] += 1;
  }
}

function forecastPhysicalFeatures(state, corePB, roundIndex) {
  void roundIndex;
  const cfg = shoePFConfig();
  const savedRngState = state.rng_state;
  const pB = clip(corePB, 0, 1);
  const expectedSign = 2 * pB - 1;
  const coreConfidence = Math.abs(expectedSign);

  let weightedCardCount = 0;
  let weightedBankerPoint = 0;
  let weightedPlayerPoint = 0;
  let totalWeight = 0;

  try {
    for (let i = 0; i < state.particles.length; i++) {
      const counts = state.particles[i].slice();
      const simulated = simulateVirtualRound(counts, state);
      const baseWeight = +state.weights[i] || 0;
      const alignmentError =
        simulated.sign - expectedSign;

      const coreFactor = Math.exp(
        -0.5
        * cfg.core_forecast_strength
        * coreConfidence
        * alignmentError
        * alignmentError
        / cfg.R
      );

      const weight = baseWeight * coreFactor;
      totalWeight += weight;
      weightedCardCount +=
        weight * simulated.totalCards;
      weightedBankerPoint +=
        weight * simulated.bankerTotal;
      weightedPlayerPoint +=
        weight * simulated.playerTotal;
    }
  } finally {
    state.rng_state = savedRngState;
  }

  const anomaly = clip(state.anomaly_score || 0, 0, 1);

  if (totalWeight <= 1e-12) {
    return [4.8, 4.5, 4.5, anomaly];
  }

  return [
    clip(weightedCardCount / totalWeight, 4, 6),
    clip(weightedBankerPoint / totalWeight, 0, 9),
    clip(weightedPlayerPoint / totalWeight, 0, 9),
    anomaly
  ];
}

function updateShoeParticleFilter(
  actualB,
  corePB,
  roundIndex,
  physicalObservation = null
) {
  const state = readShoePFState();
  const cfg = shoePFConfig();
  const actual = +actualB >= 0.5 ? 1 : 0;
  const coreDirectionB = clip(corePB, 0, 1) > 0.5;
  const alignment =
    coreDirectionB === (actual >= 0.5)
      ? 1
      : -1;

  const anomalyScore = detectAnomaly(
    state,
    actual,
    corePB
  );
  state.anomaly_score = anomalyScore;
  applyAnomalyCollapse(state, anomalyScore);

  const turbulenceBreak =
    state.last_core_alignment !== null
    && +state.last_core_alignment !== alignment;

  const persistenceMultiplier =
    +state.last_core_alignment === alignment
      ? 1
        + (cfg.persistence_boost - 1)
        * (1 - anomalyScore)
      : 1;

  const logs = new Array(state.particles.length);
  let maxLog = -Infinity;

  for (let i = 0; i < state.particles.length; i++) {
    const counts = state.particles[i].slice();
    const simulated = simulateVirtualRound(counts, state);
    state.particles[i] = counts;
    const logLike = particleLogLikelihood(
      simulated,
      actual,
      corePB,
      physicalObservation,
      persistenceMultiplier,
      anomalyScore
    );
    logs[i] = logLike;
    if (logLike > maxLog) maxLog = logLike;
  }

  let total = 0;
  for (let i = 0; i < state.weights.length; i++) {
    const updated =
      (+state.weights[i] || 0)
      * Math.exp(logs[i] - maxLog);
    state.weights[i] = updated;
    total += updated;
  }

  if (!Number.isFinite(total) || total <= 0) {
    state.weights = Array(
      state.weights.length
    ).fill(1 / state.weights.length);
  } else {
    for (let i = 0; i < state.weights.length; i++) {
      state.weights[i] /= total;
    }
  }

  const ess = effectiveSampleSize(state);
  state.last_ess = ess;
  state.last_resampled = false;

  if (ess < cfg.resample_threshold) {
    systematicResample(state);
    state.last_resampled = true;
  }

  const qEff = effectiveProcessNoise(roundIndex);
  rejuvenateParticles(state, qEff);

  state.last_effective_q = qEff;
  state.last_core_alignment = alignment;
  state.last_turbulence_break = turbulenceBreak;
  state.recent_outcomes.push(
    actual >= 0.5 ? 1 : -1
  );
  state.recent_outcomes = state.recent_outcomes.slice(-8);
  state.updates =
    Math.max(0, +state.updates || 0) + 1;

  writeShoePFState(state);
  return state;
}

function applyCorrection(seq, corePrediction) {
  const signal = corePrediction?.singleHazard || null;
  const features = buildFeatures(
    seq,
    corePrediction,
    signal
  );
  const state = readShoePFState();
  const pf4 = forecastPhysicalFeatures(
    state,
    features.core_p_b,
    features.round_index
  );

  const rawDelta = predictXGBDelta(features, pf4);
  const anomalyScore = clip(pf4[3], 0, 1);
  const brakeFactor = anomalyBrakeFactor(anomalyScore);
  const brakedDelta = rawDelta * brakeFactor;

  const maxDelta = clip(
    modelBundle?.max_delta ?? MAX_DELTA_DEFAULT,
    0,
    0.10
  );
  const delta = clip(
    brakedDelta,
    -maxDelta,
    maxDelta
  );

  const corePB = features.core_p_b;
  const finalPB = clip(corePB + delta, 0, 1);
  const direction = finalPB > 0.5 ? "B" : "P";
  const finalPP = 1 - finalPB;
  const confidence =
    direction === "B" ? finalPB : finalPP;
  const coreDirection = String(
    corePrediction?.direction
    || (corePB > 0.5 ? "B" : "P")
  );
  const active = Boolean(modelBundle?.trained);

  const physicalPrediction = {
    pred_card_count: pf4[0],
    pred_banker_point: pf4[1],
    pred_player_point: pf4[2],
    anomaly_score: anomalyScore
  };

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
        physicalPrediction,
        rawDelta: 0,
        brakeFactor,
        brakedDelta: 0,
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
    probabilities: {
      B: finalPB,
      P: finalPP
    },
    regime: flipped
      ? "11D Anomaly Brake 殘差修正換邊"
      : corePrediction.regime,
    residualBias: {
      version: VERSION,
      active: true,
      modelLoaded,
      modelLoadError,
      coreDirection,
      corePB,
      physicalPrediction,
      rawDelta,
      brakeFactor,
      brakedDelta,
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
      const raw = JSON.parse(
        localStorage.getItem(key) || "null"
      );
      if (raw && Array.isArray(raw.history)) {
        return raw.history
          .filter(x => ["B", "P", "T"].includes(x))
          .slice(-500);
      }
    } catch (_) {}
  }
  return [];
}

function rotateShoeId() {
  try {
    localStorage.removeItem(SHOE_KEY);
    localStorage.removeItem(PENDING_KEY);
    localStorage.removeItem(SHOE_PF_STATE_KEY);
    localStorage.removeItem(PHYSICAL_OBS_KEY);
    for (const key of LEGACY_SHOE_PF_STATE_KEYS) {
      localStorage.removeItem(key);
    }
  } catch (_) {}
}

function saveSelection(direction) {
  try {
    const old = JSON.parse(
      localStorage.getItem(STORAGE_KEY) || "null"
    ) || {};
    const streak =
      old.last_selected === direction
        ? Math.max(
            1,
            (+old.selection_streak || 0) + 1
          )
        : 1;

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        last_selected: direction,
        selection_streak: streak
      })
    );
  } catch (_) {}
}

function renderPrediction(p, historyLength) {
  const el = id => document.getElementById(id);
  const orb = el("directionOrb");
  if (!orb) return;

  const isB = p.direction === "B";
  el("directionText").textContent =
    isB ? "莊" : "閒";
  el("directionCode").textContent =
    isB ? "BANKER" : "PLAYER";
  el("confidence").textContent =
    (p.confidence * 100).toFixed(1) + "%";
  el("regime").textContent = p.regime;
  el("strength").textContent =
    p.strength >= .68
      ? "穩定"
      : p.strength >= .52
        ? "中等"
        : "保守";

  orb.className =
    "direction-orb "
    + (isB ? "banker" : "player");

  if (el("modePill")) {
    el("modePill").textContent =
      p.residualBias?.active
        ? "11D Anomaly Brake 修正完成"
        : "分析完成";
  }
  if (el("roundCount")) {
    el("roundCount").textContent =
      historyLength;
  }
  if (el("message")) {
    el("message").textContent =
      `第 ${historyLength + 1} 局分析完成`;
  }
}

function readTrainingRows() {
  try {
    const rows = JSON.parse(
      localStorage.getItem(TRAINING_KEY) || "[]"
    );
    return Array.isArray(rows) ? rows : [];
  } catch (_) {
    return [];
  }
}

function writeTrainingRows(rows) {
  try {
    localStorage.setItem(
      TRAINING_KEY,
      JSON.stringify(
        rows.slice(-MAX_TRAINING_ROWS)
      )
    );
  } catch (_) {}
}

function normalizePhysicalObservation(value) {
  if (!value || typeof value !== "object") return null;

  const totalCards = [4, 5, 6].includes(+value.totalCards)
    ? +value.totalCards
    : null;

  const playerPoint =
    value.playerPoint !== null
    && value.playerPoint !== undefined
    && Number.isInteger(+value.playerPoint)
    && +value.playerPoint >= 0
    && +value.playerPoint <= 9
      ? +value.playerPoint
      : null;

  const bankerPoint =
    value.bankerPoint !== null
    && value.bankerPoint !== undefined
    && Number.isInteger(+value.bankerPoint)
    && +value.bankerPoint >= 0
    && +value.bankerPoint <= 9
      ? +value.bankerPoint
      : null;

  if (
    totalCards === null
    && playerPoint === null
    && bankerPoint === null
  ) {
    return null;
  }

  return {
    totalCards,
    playerPoint,
    bankerPoint
  };
}

function setPhysicalObservation(value) {
  const normalized =
    normalizePhysicalObservation(value);
  if (!normalized) {
    throw new Error(
      "physical observation requires totalCards 4/5/6 and/or Player/Banker points 0..9"
    );
  }
  try {
    localStorage.setItem(
      PHYSICAL_OBS_KEY,
      JSON.stringify(normalized)
    );
  } catch (_) {}
  return normalized;
}

function consumePhysicalObservation(explicitValue = null) {
  const direct =
    normalizePhysicalObservation(explicitValue);
  if (direct) return direct;

  try {
    const fromWindow =
      normalizePhysicalObservation(
        window.__BGS_RESIDUAL_PHYSICAL_OBSERVATION__
      );
    if (fromWindow) {
      window.__BGS_RESIDUAL_PHYSICAL_OBSERVATION__ = null;
      return fromWindow;
    }
  } catch (_) {}

  try {
    const stored = normalizePhysicalObservation(
      JSON.parse(
        localStorage.getItem(PHYSICAL_OBS_KEY)
        || "null"
      )
    );
    localStorage.removeItem(PHYSICAL_OBS_KEY);
    return stored;
  } catch (_) {
    return null;
  }
}

function registerPrediction(seq, prediction) {
  const residual = prediction?.residualBias || {};
  const features =
    residual.features
    || buildFeatures(
      seq,
      prediction,
      prediction?.singleHazard || null
    );
  const physical =
    residual.physicalPrediction || {};

  const pending = {
    shoe_id: getShoeId(),
    created_at: Date.now(),
    history_fingerprint: seq.join(""),
    core_p_b: +features.core_p_b,
    core_direction:
      residual.coreDirection
      || String(prediction?.direction || ""),
    physical_prediction: {
      pred_card_count:
        Number.isFinite(+physical.pred_card_count)
          ? +physical.pred_card_count
          : 4.8,
      pred_banker_point:
        Number.isFinite(+physical.pred_banker_point)
          ? +physical.pred_banker_point
          : 4.5,
      pred_player_point:
        Number.isFinite(+physical.pred_player_point)
          ? +physical.pred_player_point
          : 4.5,
      anomaly_score:
        Number.isFinite(+physical.anomaly_score)
          ? +physical.anomaly_score
          : 0
    },
    features
  };

  try {
    localStorage.setItem(
      PENDING_KEY,
      JSON.stringify(pending)
    );
  } catch (_) {}
}

function settlePending(
  actualOutcome,
  physicalObservation = null
) {
  const actual =
    String(actualOutcome || "").toUpperCase();

  if (actual === "T") return;
  if (actual !== "B" && actual !== "P") return;

  let pending = null;
  try {
    pending = JSON.parse(
      localStorage.getItem(PENDING_KEY) || "null"
    );
  } catch (_) {}
  if (!pending?.features) return;

  const actualB = actual === "B" ? 1 : 0;
  const corePB = clip(
    +pending.features.core_p_b || 0.5
  );
  const residualTarget = actualB - corePB;
  const physical =
    consumePhysicalObservation(
      physicalObservation
    );
  const physicalPrediction =
    pending.physical_prediction || {};

  const row = {
    schema_version: 9,
    shoe_id: String(
      pending.shoe_id || getShoeId()
    ),
    created_at:
      +pending.created_at || Date.now(),
    history_fingerprint:
      String(pending.history_fingerprint || ""),
    actual_outcome: actual,
    actual_b: actualB,
    residual_target: residualTarget,
    pred_card_count:
      Number.isFinite(
        +physicalPrediction.pred_card_count
      )
        ? +physicalPrediction.pred_card_count
        : 4.8,
    pred_banker_point:
      Number.isFinite(
        +physicalPrediction.pred_banker_point
      )
        ? +physicalPrediction.pred_banker_point
        : 4.5,
    pred_player_point:
      Number.isFinite(
        +physicalPrediction.pred_player_point
      )
        ? +physicalPrediction.pred_player_point
        : 4.5,
    anomaly_score:
      Number.isFinite(
        +physicalPrediction.anomaly_score
      )
        ? +physicalPrediction.anomaly_score
        : 0,
    observed_total_cards:
      physical?.totalCards ?? null,
    observed_player_point:
      physical?.playerPoint ?? null,
    observed_banker_point:
      physical?.bankerPoint ?? null,
    ...pending.features
  };

  const rows = readTrainingRows();
  const duplicate =
    rows.length
    && rows.at(-1)?.shoe_id === row.shoe_id
    && rows.at(-1)?.history_fingerprint
      === row.history_fingerprint;

  if (!duplicate) {
    rows.push(row);
    updateShoeParticleFilter(
      actualB,
      corePB,
      +pending.features.round_index || 1,
      physical
    );
  }

  writeTrainingRows(rows);
  try {
    localStorage.removeItem(PENDING_KEY);
  } catch (_) {}
}

function rebuildShoeParticleFilter(
  rows = readTrainingRows()
) {
  const shoeId = getShoeId();
  resetShoeParticleFilter(shoeId);

  for (const row of rows) {
    if (
      String(row?.shoe_id || "")
      !== String(shoeId)
    ) {
      continue;
    }

    const actualB =
      Number.isFinite(+row?.actual_b)
        ? +row.actual_b
        : String(
            row?.actual_outcome || ""
          ).toUpperCase() === "B"
          ? 1
          : 0;

    const physical =
      normalizePhysicalObservation({
        totalCards: row?.observed_total_cards,
        playerPoint: row?.observed_player_point,
        bankerPoint: row?.observed_banker_point
      });

    updateShoeParticleFilter(
      actualB,
      clip(+row?.core_p_b || 0.5),
      +row?.round_index || 1,
      physical
    );
  }
}

function rollbackTrainingIfNeeded() {
  const history = readHistory();
  const rows = readTrainingRows();
  if (!rows.length) return;

  const last = rows.at(-1);
  if (
    last?.shoe_id === getShoeId()
    && +last.round_index > history.length
  ) {
    rows.pop();
    writeTrainingRows(rows);
    rebuildShoeParticleFilter(rows);
  }

  try {
    localStorage.removeItem(PENDING_KEY);
  } catch (_) {}
}

function exportTrainingData() {
  return JSON.stringify({
    schema_version: 9,
    feature_names: FEATURE_NAMES,
    physical_feature_names:
      PHYSICAL_FEATURE_NAMES,
    model_feature_names:
      MODEL_FEATURE_NAMES,
    feature_schema:
      "7D_PLUS_4D_ANOMALY_BRAKE",
    rows: readTrainingRows()
  }, null, 2);
}

function downloadTrainingData() {
  const blob = new Blob(
    [exportTrainingData()],
    {
      type: "application/json;charset=utf-8"
    }
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download =
    `bgs_xgb_anomaly_brake_11d_training_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function loadModel(url = MODEL_URL) {
  modelLoaded = false;
  modelLoadError = "";

  try {
    const response = await fetch(
      url,
      { cache: "no-store" }
    );
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status}`
      );
    }

    const bundle = await response.json();
    if (
      !bundle
      || bundle.model_type
        !== "xgb_anomaly_brake_11d_residual"
    ) {
      throw new Error("invalid_model_bundle");
    }

    const names =
      Array.isArray(bundle.feature_names)
        ? bundle.feature_names
        : [];
    if (
      names.join("|")
      !== FEATURE_NAMES.join("|")
    ) {
      throw new Error(
        "upstream_feature_schema_mismatch"
      );
    }

    const modelNames =
      Array.isArray(bundle.model_feature_names)
        ? bundle.model_feature_names
        : [];
    if (
      modelNames.join("|")
      !== MODEL_FEATURE_NAMES.join("|")
    ) {
      throw new Error(
        "model_feature_schema_mismatch"
      );
    }

    modelBundle = bundle;
    modelLoaded = true;
    readShoePFState();
    return bundle;
  } catch (error) {
    modelBundle = null;
    modelLoadError = String(
      error?.message
      || error
      || "model_load_failed"
    );
    readShoePFState();
    return null;
  }
}

function installUIOverride() {
  if (typeof document === "undefined") return;

  const oldBtn =
    document.getElementById("btnStart");
  if (!oldBtn) return;

  const btn = oldBtn.cloneNode(true);
  oldBtn.replaceWith(btn);

  btn.addEventListener("click", () => {
    const history = readHistory();
    if (!history.length) {
      const msg =
        document.getElementById("message");
      if (msg) {
        msg.textContent = "請先輸入牌局紀錄";
        msg.classList.add("warning");
      }
      return;
    }

    const corePrediction =
      CORE.hazardChoose(history);
    const prediction =
      applyCorrection(
        history,
        corePrediction
      );

    saveSelection(prediction.direction);
    registerPrediction(
      history,
      prediction
    );
    renderPrediction(
      prediction,
      history.length
    );
  });

  const b =
    document.getElementById("btnB");
  const p =
    document.getElementById("btnP");
  const t =
    document.getElementById("btnT");

  if (b) {
    b.addEventListener(
      "click",
      () => settlePending("B")
    );
  }
  if (p) {
    p.addEventListener(
      "click",
      () => settlePending("P")
    );
  }
  if (t) {
    t.addEventListener(
      "click",
      () => settlePending("T")
    );
  }

  const back =
    document.getElementById("btnBack");
  if (back) {
    back.addEventListener(
      "click",
      () => setTimeout(
        rollbackTrainingIfNeeded,
        0
      )
    );
  }

  const end =
    document.getElementById("btnEnd");
  if (end) {
    end.addEventListener(
      "click",
      rotateShoeId
    );
  }
}

if (typeof window !== "undefined") {
  window.__BGS_RESIDUAL_BIAS__ = {
    version: VERSION,
    featureNames: FEATURE_NAMES,
    physicalFeatureNames:
      PHYSICAL_FEATURE_NAMES,
    modelFeatureNames:
      MODEL_FEATURE_NAMES,
    buildFeatures,
    sxMarkovPSame,
    applyCorrection,
    loadModel,
    setEstimatedTotalHands,
    getEstimatedTotalHands,
    setPhysicalObservation,
    settlePending,
    exportTrainingData,
    downloadTrainingData,
    resetShoeParticleFilter,
    updateShoeParticleFilter,
    anomalyBrakeFactor,
    getBlindPhysicalPrediction: (
      corePB = 0.5,
      roundIndex = 1
    ) => {
      const values =
        forecastPhysicalFeatures(
          readShoePFState(),
          corePB,
          roundIndex
        );

      return {
        pred_card_count: values[0],
        pred_banker_point: values[1],
        pred_player_point: values[2],
        anomaly_score: values[3]
      };
    },
    getShoeParticleFilterStatus: () => {
      const state = readShoePFState();
      return {
        shoeId: state.shoe_id,
        updates: +state.updates || 0,
        anomalyScore:
          clip(state.anomaly_score || 0, 0, 1),
        recentOutcomes:
          Array.isArray(state.recent_outcomes)
            ? state.recent_outcomes.slice()
            : [],
        effectiveSampleSize:
          effectiveSampleSize(state),
        lastEffectiveQ:
          +state.last_effective_q
          || shoePFConfig().Q_early,
        lastCoreAlignment:
          Number.isFinite(
            +state.last_core_alignment
          )
            ? +state.last_core_alignment
            : null,
        lastTurbulenceBreak:
          Boolean(
            state.last_turbulence_break
          ),
        lastResampled:
          Boolean(state.last_resampled),
        config: shoePFConfig()
      };
    },
    resetParticleFilter:
      resetShoeParticleFilter,
    getTrainingCount: () =>
      readTrainingRows().length,
    getModelStatus: () => ({
      loaded: modelLoaded,
      trained:
        Boolean(modelBundle?.trained),
      error: modelLoadError,
      featureSchema:
        "7D_PLUS_4D_ANOMALY_BRAKE"
    })
  };
}

loadModel();
installUIOverride();
})();
