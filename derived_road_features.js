(() => {
"use strict";

const FEATURE_NAMES = [
  "big_eye_turn_now",
  "big_eye_steps_since_turn",
  "big_eye_turn_rate_6",
  "small_road_turn_now",
  "small_road_steps_since_turn",
  "small_road_turn_rate_6",
  "cockroach_turn_now",
  "cockroach_steps_since_turn",
  "cockroach_turn_rate_6",
  "derived_turn_sync"
];

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

function turnNow(markers) {
  if (markers.length < 2) return 0;
  return +markers.at(-1) !== +markers.at(-2) ? 1 : 0;
}

function stepsSinceTurn(markers) {
  if (!markers.length) return 0;
  const current = +markers.at(-1);
  let steps = 1;
  for (let i = markers.length - 2; i >= 0; i--) {
    if (+markers[i] !== current) break;
    steps += 1;
  }
  return steps;
}

function turnRate(markers, window = 6) {
  if (markers.length < 2) return 0;
  const recent = markers.slice(-Math.max(2, Math.trunc(+window || 6)));
  let turns = 0;
  for (let i = 1; i < recent.length; i++) {
    if (+recent[i] !== +recent[i - 1]) turns += 1;
  }
  return turns / Math.max(1, recent.length - 1);
}

function roadTurnState(history, offset) {
  const markers = derivedMarkers(history, offset);
  return {
    turn_now: turnNow(markers),
    steps_since_turn: stepsSinceTurn(markers),
    turn_rate_6: turnRate(markers, 6),
    available: markers.length >= 2 ? 1 : 0
  };
}

function buildFeatures(history) {
  const bigEye = roadTurnState(history, 1);
  const small = roadTurnState(history, 2);
  const cockroach = roadTurnState(history, 3);
  const states = [bigEye, small, cockroach];
  const available = states.reduce((sum, state) => sum + state.available, 0);
  const turnSync = available
    ? states.reduce((sum, state) => sum + state.turn_now, 0) / available
    : 0;

  return {
    big_eye_turn_now: bigEye.turn_now,
    big_eye_steps_since_turn: bigEye.steps_since_turn,
    big_eye_turn_rate_6: bigEye.turn_rate_6,
    small_road_turn_now: small.turn_now,
    small_road_steps_since_turn: small.steps_since_turn,
    small_road_turn_rate_6: small.turn_rate_6,
    cockroach_turn_now: cockroach.turn_now,
    cockroach_steps_since_turn: cockroach.steps_since_turn,
    cockroach_turn_rate_6: cockroach.turn_rate_6,
    derived_turn_sync: turnSync
  };
}

if (typeof window !== "undefined") {
  window.__BGS_DERIVED_ROADS__ = {
    version: "DERIVED_ROADS_TURN_17D_V2",
    featureNames: FEATURE_NAMES,
    normalizeBP,
    derivedMarkers,
    turnNow,
    stepsSinceTurn,
    turnRate,
    roadTurnState,
    buildFeatures
  };
}
})();
