(() => {
"use strict";

const FEATURE_NAMES = [
  "big_eye_color",
  "big_eye_run",
  "big_eye_switch_rate_6",
  "small_road_color",
  "small_road_run",
  "small_road_switch_rate_6",
  "cockroach_color",
  "cockroach_run",
  "cockroach_switch_rate_6",
  "derived_road_agreement"
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

function currentMarkerRun(markers) {
  if (!markers.length) return 0;
  const color = +markers.at(-1);
  let run = 1;
  for (let i = markers.length - 2; i >= 0; i--) {
    if (+markers[i] !== color) break;
    run += 1;
  }
  return run;
}

function switchRate(markers, window = 6) {
  if (markers.length < 2) return 0;
  const recent = markers.slice(-Math.max(2, Math.trunc(+window || 6)));
  let changes = 0;
  for (let i = 1; i < recent.length; i++) {
    if (+recent[i] !== +recent[i - 1]) changes += 1;
  }
  return changes / Math.max(1, recent.length - 1);
}

function roadState(history, offset) {
  const markers = derivedMarkers(history, offset);
  return {
    color: markers.length ? +markers.at(-1) : 0,
    run: currentMarkerRun(markers),
    switch_rate_6: switchRate(markers, 6)
  };
}

function buildFeatures(history) {
  const bigEye = roadState(history, 1);
  const small = roadState(history, 2);
  const cockroach = roadState(history, 3);
  const agreement = (bigEye.color + small.color + cockroach.color) / 3;
  return {
    big_eye_color: bigEye.color,
    big_eye_run: bigEye.run,
    big_eye_switch_rate_6: bigEye.switch_rate_6,
    small_road_color: small.color,
    small_road_run: small.run,
    small_road_switch_rate_6: small.switch_rate_6,
    cockroach_color: cockroach.color,
    cockroach_run: cockroach.run,
    cockroach_switch_rate_6: cockroach.switch_rate_6,
    derived_road_agreement: agreement
  };
}

if (typeof window !== "undefined") {
  window.__BGS_DERIVED_ROADS__ = {
    version: "DERIVED_ROADS_17D_V1",
    featureNames: FEATURE_NAMES,
    normalizeBP,
    derivedMarkers,
    currentMarkerRun,
    switchRate,
    roadState,
    buildFeatures
  };
}
})();
