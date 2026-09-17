# BBB XGBoost Residual Bias Layer — 17D Derived Roads

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

Ten deterministic lower-road features are appended:

8. `big_eye_color` — Big Eye Boy current marker: red=+1, blue=-1, unavailable=0
9. `big_eye_run` — current same-color marker run length
10. `big_eye_switch_rate_6` — color switch rate over the most recent 6 markers
11. `small_road_color`
12. `small_road_run`
13. `small_road_switch_rate_6`
14. `cockroach_color`
15. `cockroach_run`
16. `cockroach_switch_rate_6`
17. `derived_road_agreement` — mean of the three current colors; all red=+1, all blue=-1, 2-vs-1=+/-0.333...

The three lower roads are deterministic transforms of Big Road history. Their red/blue markers represent structural repetition vs. break, not Banker vs. Player direction.

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

The page loads:

```text
derived_road_features.js
residual_bias_runtime_17d.js
```

The runtime collects labeled rows using the 17D schema. Existing local V1 rows remain usable because the trainer can rebuild the ten lower-road features from `history_fingerprint`.

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

The checked-in `residual_bias_model.json` intentionally remains `trained:false`. Therefore adding the 17D road reader does not invent a win-rate increase or silently change the current 256D/V23 output. XGB becomes active only after real labeled data is trained and a validated 17D model bundle is committed.
