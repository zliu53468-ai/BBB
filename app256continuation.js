(() => {
"use strict";

const BASE = (typeof window !== "undefined") ? window.__BGS256_TEST__ : null;
if (!BASE) return;

const VERSION = "V24_1_TRANSITION_CLIFF";
const STORAGE_KEY = "bgs256d_adaptive_sx_hazard_v24_1";
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const GAP_SCALE = 0.30;

const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const bp = seq => seq.filter(x => x === "B" || x === "P");
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

function transitionSequence(seq) {
  const a = bp(seq), out = [];
  for (let i = 1; i < a.length; i++) out.push(a[i] === a[i - 1] ? "S" : "X");
  return out;
}

function transitionRunsFromTokens(tokens) {
  if (!tokens.length) return { completed: [], current: null };
  const runs = [];
  let token = tokens[0], length = 1;
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === token) length++;
    else { runs.push({ token, length }); token = tokens[i]; length = 1; }
  }
  return { completed: runs, current: { token, length } };
}

function transitionMotifOrder(tokens, order) {
  if (order < 1 || tokens.length <= order) return null;
  const context = tokens.slice(-order).join("");
  let same = 0, sw = 0, total = 0;
  const start = Math.max(0, tokens.length - 120);
  for (let i = start; i + order < tokens.length; i++) {
    if (tokens.slice(i, i + order).join("") !== context) continue;
    const age = tokens.length - 1 - (i + order);
    const w = Math.pow(0.965, age);
    total += w;
    if (tokens[i + order] === "S") same += w;
    else sw += w;
  }
  if (total <= 0) return null;
  const prior = 0.62, denom = total + 2 * prior;
  return {
    order,
    context,
    pSame: clip((same + prior) / denom),
    pSwitch: clip((sw + prior) / denom),
    support: clip(total / 3.2),
    weightedSamples: total
  };
}

function transitionMotifBackoff(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };
  const details = [];
  for (let order = Math.min(5, tokens.length); order >= 1; order--) {
    const item = transitionMotifOrder(tokens, order);
    if (item) details.push(item);
  }
  if (!details.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, order: 0, agreement: 0, details: [] };

  const orderWeights = { 5: 1.00, 4: 0.84, 3: 0.68, 2: 0.52, 1: 0.36 };
  let sameSum = 0, weightSum = 0, supportSum = 0;
  const dirs = [];
  for (const item of details) {
    const ow = orderWeights[item.order] || 0.30;
    const w = ow * (0.35 + 0.65 * item.support);
    sameSum += item.pSame * w;
    weightSum += w;
    supportSum += ow * item.support;
    if (Math.abs(item.pSame - 0.5) >= 0.04) dirs.push(Math.sign(item.pSame - 0.5));
  }
  const raw = weightSum ? sameSum / weightSum : 0.5;
  const support = clip(supportSum / 2.5);
  const agreement = dirs.length ? Math.abs(dirs.reduce((a, b) => a + b, 0)) / dirs.length : 0;
  const shrink = 0.48 + 0.52 * support;
  const pSame = clip(0.5 + (raw - 0.5) * shrink, 0.12, 0.88);
  return { pSame, pSwitch: 1 - pSame, support, order: details[0]?.order || 0, agreement, details };
}

function transitionPersistAtDepth(completedRuns, token, depth) {
  const items = completedRuns.filter(r => r.token === token).slice(-24);
  let reached = 0, persisted = 0, ended = 0;
  for (let i = 0; i < items.length; i++) {
    if (items[i].length < depth) continue;
    const w = Math.pow(0.95, items.length - 1 - i);
    reached += w;
    if (items[i].length > depth) persisted += w;
    else ended += w;
  }
  const prior = 0.70, denom = reached + 2 * prior;
  const raw = denom > 0 ? clip((persisted + prior) / denom) : 0.5;
  const support = clip(reached / 4);
  const pPersist = clip(0.5 + (raw - 0.5) * (0.42 + 0.58 * support), 0.12, 0.88);
  return { pPersist, support, reached, persisted, ended };
}

function transitionDepthForecast(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return { pSame: 0.5, pSwitch: 0.5, support: 0, token: "", depth: 0, pPersist: 0.5 };
  const runs = transitionRunsFromTokens(tokens);
  const token = runs.current.token;
  const depth = runs.current.length;
  const stat = transitionPersistAtDepth(runs.completed, token, depth);
  const pSame = token === "S" ? stat.pPersist : 1 - stat.pPersist;
  return { pSame, pSwitch: 1 - pSame, support: stat.support, token, depth, pPersist: stat.pPersist, ...stat };
}

function transitionMomentum(seq) {
  const tokens = transitionSequence(seq);
  if (tokens.length < 3) return {
    pSame: 0.5, pSwitch: 0.5, edge: 0, rawEdge: 0,
    support: clip(tokens.length / 6), rawSupport: clip(tokens.length / 6),
    recentSame: 0.5, previousSame: 0.5, delta: 0,
    token: tokens.at(-1) || "", depth: tokens.length ? 1 : 0,
    resetApplied: false, resetScale: 1
  };

  const runs = transitionRunsFromTokens(tokens);
  const token = runs.current.token;
  const depth = runs.current.length;
  const recent = tokens.slice(-4);
  const previous = tokens.slice(Math.max(0, tokens.length - 12), Math.max(0, tokens.length - 4));
  const sameRate = arr => arr.length ? arr.filter(x => x === "S").length / arr.length : 0.5;
  const recentSame = sameRate(recent);
  const previousSame = sameRate(previous);
  const delta = recentSame - previousSame;
  const rawSupport = clip(Math.min(recent.length, Math.max(1, previous.length)) / 4) * clip(tokens.length / 8);
  const rawEdge = signed(delta * 1.55);

  const resetScale = depth === 1 ? 0.30 : depth === 2 ? 0.72 : 1;
  const resetApplied = depth <= 2;
  const edge = signed(rawEdge * resetScale);
  const support = clip(rawSupport * resetScale);
  const pSame = clip(0.5 + 0.5 * edge * (0.45 + 0.55 * support), 0.16, 0.84);
  return {
    pSame, pSwitch: 1 - pSame, edge, rawEdge, support, rawSupport,
    recentSame, previousSame, delta, token, depth, resetApplied, resetScale
  };
}

function transitionSurvivalCliff(seq) {
  const tokens = transitionSequence(seq);
  if (!tokens.length) return {
    token: "", depth: 0, pPersist: 0.5, nextPPersist: 0.5, previousPPersist: 0.5,
    support: 0, nextSupport: 0, previousSupport: 0,
    cliff: 0, acceleration: 0, cliffScore: 0, cliffSupport: 0, evidence: 0,
    directionEdge: 0
  };
  const runs = transitionRunsFromTokens(tokens);
  const token = runs.current.token;
  const depth = runs.current.length;
  const now = transitionPersistAtDepth(runs.completed, token, depth);
  const next = transitionPersistAtDepth(runs.completed, token, depth + 1);
  const prev = depth > 1 ? transitionPersistAtDepth(runs.completed, token, depth - 1) : now;
  const cliff = clip(now.pPersist - next.pPersist, 0, 1);
  const localSlope = next.pPersist - now.pPersist;
  const previousSlope = depth > 1 ? now.pPersist - prev.pPersist : 0;
  const acceleration = clip(previousSlope - localSlope, 0, 1);
  const cliffScore = clip(0.68 * cliff + 0.32 * acceleration);
  const cliffSupport = clip(Math.min(
    now.support || 0,
    Math.max(0.20, next.support || 0),
    depth > 1 ? Math.max(0.20, prev.support || 0) : 1
  ));
  const depthGate = depth >= 4 ? 1 : depth === 3 ? 0.72 : depth === 2 ? 0.38 : 0;
  const evidence = cliffScore * cliffSupport * depthGate;
  const directionEdge = (token === "S" ? -1 : 1) * evidence;
  return {
    token, depth,
    pPersist: now.pPersist, nextPPersist: next.pPersist, previousPPersist: prev.pPersist,
    support: now.support, nextSupport: next.support, previousSupport: prev.support,
    cliff, acceleration, cliffScore, cliffSupport, evidence, directionEdge
  };
}

function currentRoadState(seq) {
  const road = BASE.buildBigRoad(seq);
  const current = road.currentStreak || null;
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];
  const side = current?.side || "";
  const length = current?.logicalLength || 0;
  const opposite = side === "B" ? "P" : "B";
  return { road, current, completed, side, length, opposite, sign: sideSign(side) };
}

function stageCurve(completed, side, maxStage = 6) {
  const curve = [];
  for (let stage = 1; stage <= maxStage; stage++) {
    const s = BASE.stageSurvival(completed, side, stage);
    curve.push({ stage, pContinue: s.cont, pTurn: s.turn, support: s.support, reached: s.reached || 0 });
  }
  return curve;
}

function conditionalStageEvidence(completed, side, length) {
  if (!side || !length) return {
    pSame: 0.5, support: 0, contextPSame: 0.5, contextSupport: 0,
    nextPSame: 0.5, nextSupport: 0, cliff: 0, cliffSupport: 0,
    previousPSame: 0.5, previousSupport: 0, localSlope: 0, previousSlope: 0,
    cliffAcceleration: 0, cliffScore: 0, curve: []
  };
  const now = BASE.stageSurvival(completed, side, length);
  const next = BASE.stageSurvival(completed, side, length + 1);
  const previous = length > 1 ? BASE.stageSurvival(completed, side, length - 1) : now;
  const context = BASE.contextualStageStats(completed, completed.length, side, length);
  const curve = stageCurve(completed, side, Math.max(6, Math.min(8, length + 2)));
  const cliff = clip(now.cont - next.cont, 0, 1);
  const localSlope = next.cont - now.cont;
  const previousSlope = length > 1 ? now.cont - previous.cont : 0;
  const cliffAcceleration = clip(previousSlope - localSlope, 0, 1);
  const cliffScore = clip(0.68 * cliff + 0.32 * cliffAcceleration);
  const cliffSupport = clip(Math.min(
    now.support || 0,
    Math.max(0.20, next.support || 0),
    length > 1 ? Math.max(0.20, previous.support || 0) : 1
  ));
  return {
    pSame: now.cont,
    pSwitch: now.turn,
    support: now.support || 0,
    contextPSame: context.cont,
    contextPSwitch: context.turn,
    contextSupport: context.support || 0,
    nextPSame: next.cont,
    nextSupport: next.support || 0,
    previousPSame: previous.cont,
    previousSupport: previous.support || 0,
    cliff,
    cliffSupport,
    localSlope,
    previousSlope,
    cliffAcceleration,
    cliffScore,
    curve
  };
}

function candidateEvidence(seq, state, basePrediction) {
  const cand = basePrediction?.candidates || BASE.bigRoadCandidates(seq);
  if (!state.side) return { pSame: 0.5, pSwitch: 0.5, support: 0, sameScore: 0.5, switchScore: 0.5 };
  const sameScore = clip(cand?.[state.side] ?? 0.5);
  const switchScore = clip(cand?.[state.opposite] ?? 0.5);
  const diff = sameScore - switchScore;
  const pSame = clip(1 / (1 + Math.exp(-diff / 0.12)), 0.18, 0.82);
  return { pSame, pSwitch: 1 - pSame, support: clip(cand?.support || 0), sameScore, switchScore, diff };
}

function baseBackgroundEvidence(basePrediction, state) {
  if (!state.side) return { pSame: 0.5, pSwitch: 0.5, support: 0 };
  const gap = Number(basePrediction?.gap || 0);
  const sideGap = state.sign * gap;
  const pSame = clip(1 / (1 + Math.exp(-sideGap / 0.42)), 0.32, 0.68);
  const support = clip(Math.abs(gap) / 0.24);
  return { pSame, pSwitch: 1 - pSame, support, gap, sideGap };
}

function singleHazardSignals(seq, basePrediction) {
  const state = currentRoadState(seq);
  if (!state.side) return {
    state, pSame: 0.5, pSwitch: 0.5, support: 0, sameEdge: 0,
    motif: transitionMotifBackoff(seq), depth: transitionDepthForecast(seq),
    momentum: transitionMomentum(seq), transitionCliff: transitionSurvivalCliff(seq)
  };

  const motif = transitionMotifBackoff(seq);
  const depth = transitionDepthForecast(seq);
  const momentum = transitionMomentum(seq);
  const transitionCliff = transitionSurvivalCliff(seq);
  const stage = conditionalStageEvidence(state.completed, state.side, state.length);
  const candidate = candidateEvidence(seq, state, basePrediction);
  const background = baseBackgroundEvidence(basePrediction, state);

  const shortPhase = state.length === 1;
  const formationPhase = state.length >= 2 && state.length <= 3;
  const maturePhase = state.length >= 4;
  const strongShortX = shortPhase && depth.token === "X" && depth.depth >= 2 && motif.pSwitch >= 0.54 && depth.pSwitch >= 0.54;
  const switchConsensus = strongShortX;
  const consensusStrength = switchConsensus
    ? clip(((motif.pSwitch - 0.50) + (depth.pSwitch - 0.50)) / 0.28)
    : 0;
  const staleScale = switchConsensus ? clip(0.62 - 0.18 * consensusStrength, 0.44, 0.62) : 1;

  let bases;
  if (shortPhase && strongShortX) {
    bases = { stage: 0.16, context: 0.11, motif: 0.28, depth: 0.22, candidate: 0.06, momentum: 0.12 };
  } else if (shortPhase) {
    bases = { stage: 0.24, context: 0.16, motif: 0.18, depth: 0.13, candidate: 0.08, momentum: 0.10 };
  } else if (formationPhase) {
    bases = { stage: 0.31, context: 0.23, motif: 0.13, depth: 0.09, candidate: 0.09, momentum: 0.08 };
  } else {
    bases = { stage: 0.24, context: 0.15, motif: 0.15, depth: 0.12, candidate: 0.08, momentum: 0.10 };
  }

  const stageWeight = bases.stage * (0.28 + 0.72 * stage.support) * staleScale;
  const contextWeight = bases.context * (0.28 + 0.72 * stage.contextSupport) * staleScale;
  const motifWeight = bases.motif * (0.30 + 0.70 * motif.support) * (0.78 + 0.22 * motif.agreement);
  const depthWeight = bases.depth * (0.30 + 0.70 * depth.support);
  const candidateWeight = bases.candidate * (0.30 + 0.70 * candidate.support);
  const momentumWeight = bases.momentum * (0.30 + 0.70 * momentum.support);

  let hazardSupport;
  if (shortPhase && strongShortX) {
    hazardSupport = clip(
      0.15 * stage.support + 0.10 * stage.contextSupport + 0.28 * motif.support +
      0.23 * depth.support + 0.11 * candidate.support + 0.13 * momentum.support
    );
  } else if (shortPhase) {
    hazardSupport = clip(
      0.24 * stage.support + 0.16 * stage.contextSupport + 0.18 * motif.support +
      0.13 * depth.support + 0.12 * candidate.support + 0.10 * momentum.support +
      0.07 * transitionCliff.cliffSupport
    );
  } else if (formationPhase) {
    hazardSupport = clip(
      0.30 * stage.support + 0.22 * stage.contextSupport + 0.12 * motif.support +
      0.08 * depth.support + 0.12 * candidate.support + 0.08 * momentum.support +
      0.08 * transitionCliff.cliffSupport
    );
  } else {
    hazardSupport = clip(
      0.21 * stage.support + 0.13 * stage.contextSupport + 0.14 * motif.support +
      0.11 * depth.support + 0.09 * candidate.support + 0.10 * momentum.support +
      0.11 * stage.cliffSupport + 0.11 * transitionCliff.cliffSupport
    );
  }

  const backgroundWeight = shortPhase
    ? 0.17 * (1 - 0.70 * hazardSupport) + 0.035
    : formationPhase
      ? 0.20 * (1 - 0.68 * hazardSupport) + 0.04
      : 0.17 * (1 - 0.70 * hazardSupport) + 0.035;
  let neutralWeight = shortPhase
    ? 0.27 * (1 - 0.56 * hazardSupport)
    : formationPhase
      ? 0.29 * (1 - 0.55 * hazardSupport)
      : 0.27 * (1 - 0.60 * hazardSupport);
  if (strongShortX) neutralWeight *= 0.74;

  let numerator = 0.5 * neutralWeight;
  let denominator = neutralWeight;
  const add = (p, w) => { numerator += clip(p) * w; denominator += w; };
  add(stage.pSame, stageWeight);
  add(stage.contextPSame, contextWeight);
  add(motif.pSame, motifWeight);
  add(depth.pSame, depthWeight);
  add(candidate.pSame, candidateWeight);
  add(momentum.pSame, momentumWeight);
  add(background.pSame, backgroundWeight);

  let pSame = denominator > 0 ? numerator / denominator : 0.5;

  let sideCliffGate = 0;
  if (state.length === 3 && stage.cliffScore >= 0.16 && stage.cliffSupport >= 0.55) sideCliffGate = 0.18;
  else if (state.length >= 4) sideCliffGate = clip(0.72 + 0.14 * (state.length - 4), 0.72, 1);
  const cliffEvidence = stage.cliffScore * stage.cliffSupport * sideCliffGate;
  const sideCliffAdjustment = -0.22 * cliffEvidence;
  pSame += sideCliffAdjustment;

  const transitionCliffAdjustment = 0.18 * transitionCliff.directionEdge;
  pSame += transitionCliffAdjustment;

  let shortSwitchBoost = 0;
  if (switchConsensus) {
    const evidence = clip(
      0.45 * clip((motif.pSwitch - 0.50) / 0.22) * (0.35 + 0.65 * motif.support) +
      0.40 * clip((depth.pSwitch - 0.50) / 0.22) * (0.35 + 0.65 * depth.support) +
      0.15 * clip(-momentum.edge) * momentum.support
    );
    shortSwitchBoost = 0.040 * evidence;
    pSame -= shortSwitchBoost;
  }

  let formationBoost = 0;
  let formationConsensus = 0;
  let formationConsensusCount = 0;
  if (formationPhase) {
    const votes = [
      { ok: stage.pSame >= 0.55, strength: clip((stage.pSame - 0.50) / 0.20), support: stage.support },
      { ok: stage.contextPSame >= 0.55, strength: clip((stage.contextPSame - 0.50) / 0.20), support: stage.contextSupport },
      { ok: candidate.pSame >= 0.54, strength: clip((candidate.pSame - 0.50) / 0.20), support: candidate.support }
    ];
    const positive = votes.filter(v => v.ok);
    formationConsensusCount = positive.length;
    if (positive.length >= 2) {
      formationConsensus = clip(positive.reduce((s, v) => s + v.strength * (0.35 + 0.65 * v.support), 0) / positive.length);
    }

    const agreement = Math.min(stage.pSame, stage.contextPSame);
    const support = Math.min(1, 0.55 * stage.support + 0.45 * stage.contextSupport);
    const stageBoost = clip((agreement - 0.5) / 0.30) * support * 0.030;
    const momentumBoost = clip(momentum.edge) * momentum.support * 0.010;
    const consensusBoost = formationConsensus * (state.length === 2 ? 0.040 : 0.032);
    formationBoost = stageBoost + momentumBoost + consensusBoost;
    pSame += formationBoost;
  }

  pSame = clip(pSame, 0.16, 0.84);
  const sameEdge = signed((pSame - 0.5) * 2);
  const support = clip(
    0.64 * hazardSupport + 0.13 * background.support + 0.09 * Math.abs(sameEdge) +
    0.06 * momentum.support + 0.04 * stage.cliffSupport + 0.04 * transitionCliff.cliffSupport
  );

  return {
    state,
    pSame,
    pSwitch: 1 - pSame,
    sameEdge,
    support,
    hazardSupport,
    motif,
    depth,
    momentum,
    transitionCliff,
    stage,
    candidate,
    background,
    cliffEvidence,
    sideCliffAdjustment,
    transitionCliffAdjustment,
    formationBoost,
    formationConsensus,
    formationConsensusCount,
    shortPhase,
    strongShortX,
    shortSwitchPhase: strongShortX,
    formationPhase,
    maturePhase,
    switchConsensus,
    consensusStrength,
    staleScale,
    shortSwitchBoost,
    weights: { stageWeight, contextWeight, motifWeight, depthWeight, candidateWeight, momentumWeight, backgroundWeight, neutralWeight }
  };
}

function hazardChoose(seq) {
  const base = BASE.choose(seq);
  const sig = singleHazardSignals(seq, base);
  const currentSide = sig.state.side;
  if (!currentSide) return base;

  let decisionPSame = sig.pSame;
  let tieBreakEdge = 0;
  let tieBreakApplied = false;
  if (Math.abs(decisionPSame - 0.5) <= 0.02) {
    const motifEdge = (sig.motif.pSame - 0.5) * 2 * sig.motif.support * (0.70 + 0.30 * sig.motif.agreement);
    const depthEdge = (sig.depth.pSame - 0.5) * 2 * sig.depth.support;
    const momentumEdge = sig.momentum.edge * sig.momentum.support;
    const sideCliffEdge = -sig.cliffEvidence;
    const transitionCliffEdge = sig.transitionCliff.directionEdge;
    const formationEdge = sig.formationConsensus * (sig.formationPhase ? 1 : 0);
    const backgroundEdge = (sig.background.pSame - 0.5) * 2 * sig.background.support;
    tieBreakEdge = signed(
      0.22 * motifEdge +
      0.18 * depthEdge +
      0.17 * momentumEdge +
      0.13 * sideCliffEdge +
      0.13 * transitionCliffEdge +
      0.10 * formationEdge +
      0.07 * backgroundEdge
    );
    if (Math.abs(tieBreakEdge) >= 0.025) {
      const nudge = Math.min(0.018, Math.max(0.003, Math.abs(tieBreakEdge) * 0.035));
      decisionPSame = clip(0.5 + Math.sign(tieBreakEdge) * nudge, 0.482, 0.518);
      tieBreakApplied = true;
    }
  }

  const direction = decisionPSame >= 0.5 ? currentSide : sig.state.opposite;
  const pBUnclipped = currentSide === "B" ? decisionPSame : 1 - decisionPSame;
  const pB = clip(pBUnclipped, PROB_MIN, PROB_MAX), pP = 1 - pB;
  const confidence = direction === "B" ? pB : pP;
  const decisionEdge = signed((decisionPSame - 0.5) * 2);
  const gap = sig.state.sign * decisionEdge * GAP_SCALE;

  let regime = "S/X平衡";
  if (tieBreakApplied) regime = "邊界前瞻";
  else if (sig.transitionCliff.evidence >= 0.08 && Math.sign(sig.transitionCliff.directionEdge) === Math.sign(decisionPSame - 0.5)) regime = "規律斷點";
  else if (sig.strongShortX && decisionPSame < 0.50) regime = "短交錯切換";
  else if (sig.state.length >= 4 && sig.cliffEvidence >= 0.08 && decisionPSame < 0.50) regime = "條件斷點";
  else if (sig.formationPhase && sig.formationConsensusCount >= 2 && decisionPSame >= 0.54) regime = "延續確認";
  else if (decisionPSame >= 0.56) regime = "條件延續";
  else if (decisionPSame <= 0.44) regime = "條件切換";

  const strength = clip(
    0.42 + 0.32 * sig.support + 0.15 * Math.abs(decisionEdge) +
    0.04 * Math.min(1, sig.state.length / 4) + 0.03 * sig.momentum.support +
    0.02 * sig.transitionCliff.cliffSupport + 0.02 * sig.formationConsensus
  );

  const diag = {
    version: VERSION,
    baseGap: base.gap,
    finalGap: gap,
    rawPSame: sig.pSame,
    pSame: decisionPSame,
    pSwitch: 1 - decisionPSame,
    currentSide,
    currentLength: sig.state.length,
    phase: sig.shortPhase ? "short" : sig.formationPhase ? "formation" : "mature",
    transitionToken: sig.depth.token,
    transitionDepth: sig.depth.depth,
    motifPSame: sig.motif.pSame,
    motifSupport: sig.motif.support,
    depthPSame: sig.depth.pSame,
    depthSupport: sig.depth.support,
    momentumPSame: sig.momentum.pSame,
    momentumEdge: sig.momentum.edge,
    momentumSupport: sig.momentum.support,
    momentumResetApplied: sig.momentum.resetApplied,
    momentumResetScale: sig.momentum.resetScale,
    stagePSame: sig.stage.pSame,
    previousStagePSame: sig.stage.previousPSame,
    nextStagePSame: sig.stage.nextPSame,
    stageSupport: sig.stage.support,
    contextPSame: sig.stage.contextPSame,
    contextSupport: sig.stage.contextSupport,
    survivalCliff: sig.stage.cliff,
    cliffAcceleration: sig.stage.cliffAcceleration,
    cliffScore: sig.stage.cliffScore,
    cliffEvidence: sig.cliffEvidence,
    transitionCliffToken: sig.transitionCliff.token,
    transitionCliffDepth: sig.transitionCliff.depth,
    transitionCliffScore: sig.transitionCliff.cliffScore,
    transitionCliffEvidence: sig.transitionCliff.evidence,
    transitionCliffDirectionEdge: sig.transitionCliff.directionEdge,
    formationBoost: sig.formationBoost,
    formationConsensus: sig.formationConsensus,
    formationConsensusCount: sig.formationConsensusCount,
    hazardSupport: sig.hazardSupport,
    backgroundWeight: sig.weights.backgroundWeight,
    strongShortX: sig.strongShortX,
    switchConsensus: sig.switchConsensus,
    staleScale: sig.staleScale,
    shortSwitchBoost: sig.shortSwitchBoost,
    tieBreakApplied,
    tieBreakEdge
  };

  return {
    ...base,
    direction,
    gap,
    confidence,
    probabilities: { B: pB, P: pP },
    regime,
    strength,
    singleHazard: sig,
    v22: diag,
    v23: diag,
    v24: diag,
    v241: diag
  };
}

function continuationSignals(seq, basePrediction) { return singleHazardSignals(seq, basePrediction || BASE.choose(seq)); }
function enhancedChoose(seq) { return hazardChoose(seq); }
function expectedBranchFutureQuality(seq, first) {
  const road = BASE.buildBigRoad([...seq, first]);
  const current = road.currentStreak;
  if (!current) return { quality: 0.5, pSame: 0.5, support: 0 };
  const completed = road.streaks.length > 1 ? road.streaks.slice(0, -1) : [];
  const stage = BASE.stageSurvival(completed, first, current.logicalLength);
  const context = BASE.contextualStageStats(completed, completed.length, first, current.logicalLength);
  const support = clip(0.56 * (stage.support || 0) + 0.44 * (context.support || 0));
  const pSame = clip((0.56 * stage.cont + 0.44 * context.cont), 0.12, 0.88);
  return { quality: pSame, pSame, support };
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
    const p = hazardChoose(history);
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
    version: VERSION,
    transitionSequence,
    transitionMotifBackoff,
    transitionDepthForecast,
    transitionMomentum,
    transitionSurvivalCliff,
    conditionalStageEvidence,
    singleHazardSignals,
    continuationSignals,
    hazardChoose,
    enhancedChoose,
    expectedBranchFutureQuality
  };
}

installUIOverride();
})();