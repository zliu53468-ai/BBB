const assert = require("assert");

global.window = {};
require("./derived_road_features.js");

const roads = global.window.__BGS_DERIVED_ROADS__;
assert.ok(roads, "derived-road API should be exposed");

assert.deepStrictEqual(roads.derivedMarkers("BPB", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BPP", 1), [-1]);
assert.deepStrictEqual(roads.derivedMarkers("BBPP", 1), [1]);
assert.deepStrictEqual(roads.derivedMarkers("BTPB", 1), roads.derivedMarkers("BPB", 1));

assert.strictEqual(roads.turnNow([-1, -1, 1]), 1);
assert.strictEqual(roads.turnNow([-1, 1, 1]), 0);
assert.strictEqual(roads.stepsSinceTurn([-1, -1, 1]), 1);
assert.strictEqual(roads.stepsSinceTurn([-1, 1, 1]), 2);
assert.strictEqual(roads.turnRate([-1, 1, -1, -1], 6), 2 / 3);

const features = roads.buildFeatures("BBPBBB");
assert.ok(!Object.prototype.hasOwnProperty.call(features, "big_eye_color"));
assert.strictEqual(features.big_eye_turn_now, 1);
assert.strictEqual(features.small_road_turn_now, 1);
assert.strictEqual(features.derived_turn_sync, 1);

console.log("derived-road turn-state JS tests passed");
