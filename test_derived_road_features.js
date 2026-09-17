const assert = require("assert");

global.window = {};
require("./derived_road_features.js");

const roads = global.window.__BGS_DERIVED_ROADS__;
assert.ok(roads, "derived-road API should be exposed");

assert.deepStrictEqual(roads.derivedMarkers("BPB", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BPP", 1), [-1]);
assert.deepStrictEqual(roads.derivedMarkers("BBPP", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BTPB", 1), roads.derivedMarkers("BPB", 1));

const features = roads.buildFeatures("BPBP");
assert.strictEqual(features.big_eye_color, 1);
assert.strictEqual(features.small_road_color, 1);
assert.strictEqual(features.cockroach_color, 0);
assert.strictEqual(
  features.derived_road_agreement,
  (features.big_eye_color + features.small_road_color + features.cockroach_color) / 3
);

console.log("derived-road JS tests passed");
