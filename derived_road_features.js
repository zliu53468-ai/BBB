(() => {
"use strict";

const FEATURE_NAMES = [
  "big_eye_continue_now",
  "big_eye_turn_now",
  "big_eye_p_continue",
  "big_eye_p_turn",
  "small_road_continue_now",
  "small_road_turn_now",
  "small_road_p_continue",
  "small_road_p_turn",
  "cockroach_continue_now",
  "cockroach_turn_now",
  "cockroach_p_continue",
  "cockroach_p_turn",
  "big_road_p_continue",
  "big_road_p_turn",
  "derived_p_continue",
  "derived_p_turn"
];

const PRIOR = 1.0;
const LOCAL_WINDOW = 12;

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
  if (!runs.length) return 0.5;
  const depth = runs.at(-1);
  const completed = runs.slice(0, -1);
  const eligible = completed.filter(length => length >= depth);
  if (eligible.length < 2) return recentContinueRate(values);
  const continued = eligible.filter(length => length > depth).length;
  const stopped = eligible.length - continued;
  return (continued + prior) / (continued + stopped + 2 * prior);
}

function continuationState(values) {
  if (values.length < 2) {
    return {
      continue_now: 0,
      turn_now: 0,
      p_continue: 0.5,
      p_turn: 0.5,
      available: 0
    };
  }
  const continueNow = values.at(-1) === values.at(-2) ? 1 : 0;
  const turnNow = 1 - continueNow;
  const pContinue = Math.max(0, Math.min(1, survivalContinueProb(values)));
  return {
    continue_now: continueNow,
    turn_now: turnNow,
    p_continue: pContinue,
    p_turn: 1 - pContinue,
    available: 1
  };
}

function roadContinuationState(history, offset) {
  return continuationState(derivedMarkers(history, offset));
}

function bigRoadContinuationState(history) {
  return continuationState(normalizeBP(history));
}

function buildFeatures(history) {
  const bigEye = roadContinuationState(history, 1);
  const small = roadContinuationState(history, 2);
  const cockroach = roadContinuationState(history, 3);
  const bigRoad = bigRoadContinuationState(history);

  const states = [bigEye, small, cockroach];
  const available = states.filter(state => state.available > 0);
  const derivedPContinue = available.length
    ? available.reduce((sum, state) => sum + state.p_continue, 0) / available.length
    : 0.5;
  const derivedPTurn = 1 - derivedPContinue;

  return {
    big_eye_continue_now: bigEye.continue_now,
    big_eye_turn_now: bigEye.turn_now,
    big_eye_p_continue: bigEye.p_continue,
    big_eye_p_turn: bigEye.p_turn,
    small_road_continue_now: small.continue_now,
    small_road_turn_now: small.turn_now,
    small_road_p_continue: small.p_continue,
    small_road_p_turn: small.p_turn,
    cockroach_continue_now: cockroach.continue_now,
    cockroach_turn_now: cockroach.turn_now,
    cockroach_p_continue: cockroach.p_continue,
    cockroach_p_turn: cockroach.p_turn,
    big_road_p_continue: bigRoad.p_continue,
    big_road_p_turn: bigRoad.p_turn,
    derived_p_continue: derivedPContinue,
    derived_p_turn: derivedPTurn
  };
}

if (typeof window !== "undefined") {
  window.__BGS_DERIVED_ROADS__ = {
    version: "DERIVED_ROADS_CONTINUE_TURN_23D_V3",
    featureNames: FEATURE_NAMES,
    normalizeBP,
    derivedMarkers,
    runLengths,
    recentContinueRate,
    survivalContinueProb,
    continuationState,
    roadContinuationState,
    bigRoadContinuationState,
    buildFeatures
  };
}
})();
