const assert = require("assert");

global.window = {};
require("./derived_road_features.js");

const roads = global.window.__BGS_DERIVED_ROADS__;
assert.ok(roads, "derived-road API should be exposed");

assert.deepStrictEqual(roads.derivedMarkers("BPB", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BPP", 1), [-1]);
assert.deepStrictEqual(roads.derivedMarkers("BBPP", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BTPB", 1), roads.derivedMarkers("BPB", 1));

const features = roads.buildFeatures("BBPBBBPPBBPBPBB");
for (const prefix of ["big_eye", "small_road", "cockroach"]) {
  assert.ok(
    Math.abs(
      features[`${prefix}_p_bigroad_continue`] +
      features[`${prefix}_p_bigroad_turn`] - 1
    ) < 1e-12
  );
}
assert.ok(Math.abs(features.big_road_p_continue + features.big_road_p_turn - 1) < 1e-12);
assert.ok(
  Math.abs(
    features.derived_p_bigroad_continue +
    features.derived_p_bigroad_turn - 1
  ) < 1e-12
);

const state = roads.roadProbabilityState("BBPBBBPPBBPBPBB", 1);
if (state.available) {
  assert.strictEqual(state.continue_now + state.turn_now, 1);
}

const sparse = roads.buildFeatures("B");
assert.strictEqual(sparse.big_road_p_continue, 0.5);
assert.strictEqual(sparse.derived_p_bigroad_continue, 0.5);

for (const legacy of [
  "big_eye_color",
  "small_road_color",
  "cockroach_color",
  "derived_turn_sync"
]) {
  assert.ok(!Object.prototype.hasOwnProperty.call(features, legacy));
}

for (const name of roads.featureNames) {
  assert.ok(Number.isFinite(features[name]), `${name} should be finite`);
}

const history = "BBPBBBPPBBPBPBBPBPBBPP";
assert.strictEqual(
  roads.derivedToBigRoadContinueProb(history, 1),
  roads.derivedToBigRoadContinueProb(history, 1)
);

console.log("derived-road to Big Road continuation/reversal JS tests passed");
