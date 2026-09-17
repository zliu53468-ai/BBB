# BBB XGBoost Residual Bias Layer — 17D Derived-Road Turns

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

There is no PASS state and no second model.

## 17D feature schema

The original 7 features are preserved:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

Ten lower-road **turn-state** features are appended:

8. `big_eye_turn_now` — newest Big Eye marker just changed color (0/1)
9. `big_eye_steps_since_turn` — length of the current Big Eye structural segment
10. `big_eye_turn_rate_6` — turn density over the most recent 6 Big Eye markers
11. `small_road_turn_now`
12. `small_road_steps_since_turn`
13. `small_road_turn_rate_6`
14. `cockroach_turn_now`
15. `cockroach_steps_since_turn`
16. `cockroach_turn_rate_6`
17. `derived_turn_sync` — share of currently available lower roads that turned on the newest marker

Raw red/blue color is **not** exported to XGBoost. The standard red/blue lower-road markers are generated only internally so the code can detect structural turning points. The model therefore learns whether the three lower roads are turning, how recently they turned, and whether turns are synchronizing; it does not learn a rule such as red=Banker or blue=Player.

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

For each road, the internal marker stream is converted to:

```text
turn_now          = newest marker != previous marker
steps_since_turn  = current same-color segment length
turn_rate_6       = switches / transitions across the latest up-to-6 markers
```

`derived_turn_sync` is computed only across lower roads that have enough markers to determine whether the latest marker is a turn.

## Browser runtime

The page loads:

```text
derived_road_features.js
residual_bias_runtime_17d.js
```

The runtime collects labeled rows using the 17D schema. Existing V1 7D rows remain usable when their `history_fingerprint` is present, because the trainer can reconstruct the ten turn-state features from that history.

Useful browser-console helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getDerivedRoadFeatures(["B","P","P","B"])
__BGS_RESIDUAL_BIAS__.getModelStatus()
```

## Train and export

Use the 17D wrapper, not the original 7D trainer:

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_bias_17d.py train \
  --input bgs_xgb_residual_17d_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

`xgb_residual_bias_17d.py` reuses the original V1 XGBRegressor parameters, validation split, Brier/accuracy gates and portable-tree exporter. It changes only the feature build/schema from 7D to 17D.

The checked-in `residual_bias_model.json` intentionally remains `trained:false`. Therefore changing the reader to turn-state features does not invent a win-rate increase or silently change the current 256D/V23 output. XGB becomes active only after real labeled data is trained and a validated 17D model bundle is committed.
