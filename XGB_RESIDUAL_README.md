# BBB XGBoost Residual Bias Layer — 23D Continuation / Reversal

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
10. `big_eye_p_continue`
11. `big_eye_p_turn`
12. `small_road_continue_now`
13. `small_road_turn_now`
14. `small_road_p_continue`
15. `small_road_p_turn`
16. `cockroach_continue_now`
17. `cockroach_turn_now`
18. `cockroach_p_continue`
19. `cockroach_p_turn`
20. `big_road_p_continue`
21. `big_road_p_turn`
22. `derived_p_continue`
23. `derived_p_turn`

The lower-road red/blue markers are **not** passed to XGBoost as color identity. They are generated internally only to determine whether each lower road is structurally continuing or turning.

## Continuation / reversal probability

For each lower road the code tracks the current same-structure run depth. It estimates the probability that the current run continues one more marker by comparing that depth with previously completed runs in the same road:

```text
P(continue | current depth)
```

Previous runs that reached the current depth are eligible. Runs longer than the current depth count as historical continuations; runs that stopped exactly at that depth count as historical turns. A Beta(1,1) prior is applied so small samples do not create extreme probabilities.

When there are not enough completed runs at the current depth, the estimator falls back to a Bayesian-smoothed recent transition rate over the latest 12 transitions. With insufficient history it returns 50/50.

```text
p_turn = 1 - p_continue
```

The Big Road uses the same causal run-survival logic on B/P streaks. Therefore XGBoost can compare:

```text
Big Road P(continue) / P(turn)
Big Eye P(continue) / P(turn)
Small Road P(continue) / P(turn)
Cockroach P(continue) / P(turn)
Mean lower-road P(continue) / P(turn)
```

This is designed to answer whether the current table structure is more consistent with continuation or reversal. It does not assume that a raw red marker means Banker or that a raw blue marker means Player.

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

The checked-in `residual_bias_model.json` intentionally remains `trained:false`. The 23D reader therefore does not silently change the current 256D/V23 prediction until real labeled data is trained and a validated 23D model bundle is committed.
