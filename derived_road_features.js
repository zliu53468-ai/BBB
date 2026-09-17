(() => {
"use strict";

const FEATURE_NAMES = [
  "big_eye_continue_now",
  "big_eye_turn_now",
  "big_eye_p_bigroad_continue",
  "big_eye_p_bigroad_turn",
  "small_road_continue_now",
  "small_road_turn_now",
  "small_road_p_bigroad_continue",
  "small_road_p_bigroad_turn",
  "cockroach_continue_now",
  "cockroach_turn_now",
  "cockroach_p_bigroad_continue",
  "cockroach_p_bigroad_turn",
  "big_road_p_continue",
  "big_road_p_turn",
  "derived_p_bigroad_continue",
  "derived_p_bigroad_turn"
];

const PRIOR = 1.0;
const LOCAL_WINDOW = 12;
const CONDITIONAL_WINDOW = 24;
const MIN_EXACT_MATCHES = 3;

function normalizeBP(history) {
  const values = Array.isArray(history)
    ? history
    : String(history || "").toUpperCase().split("");
  return values
    .map(x => String(x || "").toUpperCase().trim())
    .filter(x => x === "B" || x === "P");
}

function derivedMarkers(history, offset) {
  offset = Math.trunc(+offset || 0);
  if (![1, 2, 3].includes(offset)) throw new Error("derived-road offset must be 1, 2, or 3");
  const seq = normalizeBP(history);
  if (!seq.length) return [];

  const runs = [];
  const markers = [];
  let previous = "";

  for (const side of seq) {
    if (side === previous && runs.length) {
      runs[runs.length - 1] += 1;
      const currentCol = runs.length - 1;
      const row = runs[runs.length - 1] - 1;
      if (currentCol >= offset) {
        const referenceDepth = runs[currentCol - offset];
        markers.push(referenceDepth === row ? -1 : 1);
      }
    } else {
      runs.push(1);
      const currentCol = runs.length - 1;
      if (currentCol >= offset + 1) {
        const leftDepth = runs[currentCol - 1];
        const compareDepth = runs[currentCol - 1 - offset];
        markers.push(leftDepth === compareDepth ? 1 : -1);
      }
      previous = side;
    }
  }
  return markers;
}

function runLengths(values) {
  if (!values.length) return [];
  const runs = [];
  let last = Symbol("none");
  for (const value of values) {
    if (runs.length && value === last) runs[runs.length - 1] += 1;
    else {
      runs.push(1);
      last = value;
    }
  }
  return runs;
}

function recentContinueRate(values, window = LOCAL_WINDOW, prior = PRIOR) {
  if (values.length < 2) return 0.5;
  const recent = values.slice(-Math.max(2, Math.trunc(+window || LOCAL_WINDOW) + 1));
  let cont = 0;
  for (let i = 1; i < recent.length; i++) {
    if (recent[i] === recent[i - 1]) cont += 1;
  }
  const turn = Math.max(0, recent.length - 1 - cont);
  return (cont + prior) / (cont + turn + 2 * prior);
}

function survivalContinueProb(values, prior = PRIOR) {
  if (values.length < 2) return 0.5;
  const runs = runLengths(values);
  const depth = runs.at(-1);
  const completed = runs.slice(0, -1);
  const eligible = completed.filter(length => length >= depth);
  if (eligible.length < 2) return recentContinueRate(values);
  const continued = eligible.filter(length => length > depth).length;
  const stopped = eligible.length - continued;
  return (continued + prior) / (continued + stopped + 2 * prior);
}

function markerContinueTurn(markers) {
  if (markers.length < 2) return { continue_now: 0, turn_now: 0 };
  const continueNow = +markers.at(-1) === +markers.at(-2) ? 1 : 0;
  return { continue_now: continueNow, turn_now: 1 - continueNow };
}

function smoothedBinaryProb(outcomes, prior = PRIOR) {
  if (!outcomes.length) return 0.5;
  const positives = outcomes.filter(value => +value === 1).length;
  const negatives = outcomes.length - positives;
  return (positives + prior) / (positives + negatives + 2 * prior);
}

function derivedToBigRoadContinueProb(history, offset, window = CONDITIONAL_WINDOW) {
  const seq = normalizeBP(history);
  if (seq.length < 2) return 0.5;

  const currentMarkers = derivedMarkers(seq, offset);
  if (!currentMarkers.length) return survivalContinueProb(seq);

  const currentSignal = +currentMarkers.at(-1);
  const currentState = markerContinueTurn(currentMarkers);
  const currentTransition = currentMarkers.length >= 2 ? currentState.continue_now : null;

  let exact = [];
  let signalOnly = [];

  for (let t = 1; t < seq.length; t++) {
    const prefix = seq.slice(0, t);
    const markers = derivedMarkers(prefix, offset);
    if (!markers.length || +markers.at(-1) !== currentSignal) continue;

    const bigRoadContinued = seq[t] === seq[t - 1] ? 1 : 0;
    signalOnly.push(bigRoadContinued);

    if (currentTransition !== null && markers.length >= 2) {
      const state = markerContinueTurn(markers);
      if (state.continue_now === currentTransition) exact.push(bigRoadContinued);
    }
  }

  exact = exact.slice(-Math.max(1, Math.trunc(+window || CONDITIONAL_WINDOW)));
  signalOnly = signalOnly.slice(-Math.max(1, Math.trunc(+window || CONDITIONAL_WINDOW)));

  if (exact.length >= MIN_EXACT_MATCHES) return smoothedBinaryProb(exact);
  if (signalOnly.length >= 2) return smoothedBinaryProb(signalOnly);
  return survivalContinueProb(seq);
}

function roadProbabilityState(history, offset) {
  const markers = derivedMarkers(history, offset);
  const now = markerContinueTurn(markers);
  const pBigRoadContinue = Math.max(0, Math.min(1, derivedToBigRoadContinueProb(history, offset)));
  return {
    continue_now: now.continue_now,
    turn_now: now.turn_now,
    p_bigroad_continue: pBigRoadContinue,
    p_bigroad_turn: 1 - pBigRoadContinue,
    available: markers.length ? 1 : 0
  };
}

function bigRoadContinuationState(history) {
  const seq = normalizeBP(history);
  const pContinue = Math.max(0, Math.min(1, survivalContinueProb(seq)));
  let continueNow = 0;
  let turnNow = 0;
  if (seq.length >= 2) {
    continueNow = seq.at(-1) === seq.at(-2) ? 1 : 0;
    turnNow = 1 - continueNow;
  }
  return {
    continue_now: continueNow,
    turn_now: turnNow,
    p_continue: pContinue,
    p_turn: 1 - pContinue,
    available: seq.length >= 2 ? 1 : 0
  };
}

function buildFeatures(history) {
  const bigEye = roadProbabilityState(history, 1);
  const small = roadProbabilityState(history, 2);
  const cockroach = roadProbabilityState(history, 3);
  const bigRoad = bigRoadContinuationState(history);

  const states = [bigEye, small, cockroach];
  const available = states.filter(state => state.available > 0);
  const derivedPContinue = available.length
    ? available.reduce((sum, state) => sum + state.p_bigroad_continue, 0) / available.length
    : bigRoad.p_continue;
  const derivedPTurn = 1 - derivedPContinue;

  return {
    big_eye_continue_now: bigEye.continue_now,
    big_eye_turn_now: bigEye.turn_now,
    big_eye_p_bigroad_continue: bigEye.p_bigroad_continue,
    big_eye_p_bigroad_turn: bigEye.p_bigroad_turn,
    small_road_continue_now: small.continue_now,
    small_road_turn_now: small.turn_now,
    small_road_p_bigroad_continue: small.p_bigroad_continue,
    small_road_p_bigroad_turn: small.p_bigroad_turn,
    cockroach_continue_now: cockroach.continue_now,
    cockroach_turn_now: cockroach.turn_now,
    cockroach_p_bigroad_continue: cockroach.p_bigroad_continue,
    cockroach_p_bigroad_turn: cockroach.p_bigroad_turn,
    big_road_p_continue: bigRoad.p_continue,
    big_road_p_turn: bigRoad.p_turn,
    derived_p_bigroad_continue: derivedPContinue,
    derived_p_bigroad_turn: derivedPTurn
  };
}

if (typeof window !== "undefined") {
  window.__BGS_DERIVED_ROADS__ = {
    version: "DERIVED_ROADS_BIGROAD_CONTINUE_TURN_23D_V4",
    featureNames: FEATURE_NAMES,
    normalizeBP,
    derivedMarkers,
    runLengths,
    recentContinueRate,
    survivalContinueProb,
    markerContinueTurn,
    smoothedBinaryProb,
    derivedToBigRoadContinueProb,
    roadProbabilityState,
    bigRoadContinuationState,
    buildFeatures
  };
}
})();
