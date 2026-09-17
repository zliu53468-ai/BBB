# BBB XGBoost Residual Bias Layer — 23D Big Road Continuation / Turn

BBB remains a static GitHub Pages application. The 256D/V23 R1 core is still the base predictor and XGBoost still learns only the core residual:

```text
residual = actual_B - core_p_B
```

Production correction is unchanged:

```text
delta = clip(xgb_residual, -0.10, +0.10)
final_p_B = core_p_B + delta
B if final_p_B > 0.50 else P
```

There is no PASS state, no LightGBM, and no second model.

## 23D feature schema

The original 7 features are preserved:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

Sixteen continuation/reversal features are appended:

8. `big_eye_continue_now`
9. `big_eye_turn_now`
10. `big_eye_p_bigroad_continue`
11. `big_eye_p_bigroad_turn`
12. `small_road_continue_now`
13. `small_road_turn_now`
14. `small_road_p_bigroad_continue`
15. `small_road_p_bigroad_turn`
16. `cockroach_continue_now`
17. `cockroach_turn_now`
18. `cockroach_p_bigroad_continue`
19. `cockroach_p_bigroad_turn`
20. `big_road_p_continue`
21. `big_road_p_turn`
22. `derived_p_bigroad_continue`
23. `derived_p_bigroad_turn`

Raw lower-road red/blue markers are **not** exported to XGBoost. They are used only internally as structural states so the system can estimate what those states historically implied for the next Big Road continuation or turn.

## What the new probabilities mean

The feature reader now answers two related questions.

First, the Big Road itself gets a causal run-survival estimate:

```text
P(Big Road continues one more B/P)
P(Big Road turns on the next B/P)
```

At the current streak depth, previously completed streaks that reached the same depth are compared. Runs that continued beyond the depth count as continuation; runs that stopped at that depth count as turns. A Beta(1,1) prior prevents very small samples from creating extreme probabilities. If there are too few comparable runs, the estimator falls back to a Bayesian-smoothed recent transition rate over the latest 12 transitions. With insufficient history it returns 50/50.

Second, each lower road estimates the historical conditional probability of the **next Big Road** result continuing or turning while that lower road is in a comparable current structural state:

```text
P(next Big Road continues | current Big Eye state)
P(next Big Road turns     | current Big Eye state)

P(next Big Road continues | current Small Road state)
P(next Big Road turns     | current Small Road state)

P(next Big Road continues | current Cockroach state)
P(next Big Road turns     | current Cockroach state)
```

The code first matches historical prefixes with the same internal lower-road signal and the same lower-road continue/turn state. If there are fewer than three exact matches, it backs off to matching the same internal signal only. If that is still too sparse, it falls back to the Big Road base continuation probability. Only past information is used, so the feature is causal and does not peek at future outcomes.

`derived_p_bigroad_continue` is the mean of the currently available Big Eye / Small Road / Cockroach conditional continuation probabilities. `derived_p_bigroad_turn` is its complement.

This allows XGBoost to learn interactions such as:

```text
Big Road currently has high continuation probability
+ Big Eye historically supports Big Road continuation
+ Small Road is neutral
+ Cockroach recently turned and historically supports reversal
=> residual correction can learn whether the V23 core is under/over-estimating B
```

The model is still trained on residual error, not on a hand-written rule such as red=Banker or blue=Player.

## Lower-road calculation

- Big Eye Boy uses lookback offset 1.
- Small Road uses lookback offset 2.
- Cockroach Pig uses lookback offset 3.
- Ties do not create a Big Road cell.
- Dragon tails are treated as unbounded streak depth so display geometry does not change the calculation.

Python and browser implementations live in:

```text
derived_road_features.py
derived_road_features.js
```

## Browser runtime

The page continues to load:

```text
derived_road_features.js
residual_bias_runtime_17d.js
```

The runtime builds its feature order dynamically from `derived_road_features.js`, so the legacy runtime filename can serve the new 23D schema. Existing older labeled rows remain usable when their `history_fingerprint` is present because the trainer reconstructs the 16 road probability features from historical B/P data.

Useful browser-console helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getDerivedRoadFeatures(["B","P","P","B"])
__BGS_RESIDUAL_BIAS__.getModelStatus()
```

## Train and export

The compatibility wrapper filename is still `xgb_residual_bias_17d.py`, but its schema is now 23D:

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_bias_17d.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

It reuses the original V1 XGBRegressor parameters, deterministic validation split, Brier/accuracy gates and portable-tree exporter. Only the feature schema/build step changes.

The checked-in `residual_bias_model.json` intentionally remains `trained:false`. The new 23D reader therefore does not silently change the current 256D/V23 prediction until real labeled data is trained and a validated 23D model bundle is committed.
