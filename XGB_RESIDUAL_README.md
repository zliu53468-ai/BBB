# BBB XGBoost Twin Residual Ensemble

The upstream pipeline remains frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

Only the residual correction layer is expanded. No LightGBM is used.

Both XGBoost models train on the exact same 7D feature matrix and the exact same residual label:

```text
residual = actual_B - core_p_B
```

## XGBoost twin models

### xgb_sensitive

```text
n_estimators=60
learning_rate=0.04
max_depth=4
min_child_weight=1.0
alpha=0.0
lambda=0.01
random_state=42
```

Python uses the equivalent XGBoost sklearn names `reg_alpha=0.0` and `reg_lambda=0.01`.

### xgb_robust

```text
n_estimators=50
learning_rate=0.02
max_depth=3
min_child_weight=3.0
alpha=0.1
lambda=0.3
random_state=100
```

Python uses `reg_alpha=0.1` and `reg_lambda=0.3`.

## Fixed 7D feature order

The feature dimension and feature order are unchanged:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

## Fusion inference

```text
delta_sens = xgb_sensitive(features_7d)
delta_robu = xgb_robust(features_7d)
delta_final = (delta_sens + delta_robu) / 2.0
delta_clipped = clip(delta_final, -0.10, +0.10)
final_p_B = clip(core_p_B + delta_clipped, 0.0, 1.0)
B if final_p_B > 0.50 else P
```

There is no PASS state.

## Collect labeled production rows

The existing browser collector is preserved. Ties are not used as directional labels.

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.setEstimatedTotalHands(60)
```

## Train and export

```bash
python -m pip install -r requirements-xgb.txt
python xgb_twin_residual.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The trainer reuses the existing deterministic shoe-level validation split. Both models receive the same `X_train` and the same `y_train = actual_B - core_p_b`. The production validation gate is applied to the fused 50/50 result.

`xgb_twin_residual.py` exports both XGBoost tree sets into `residual_bias_model.json`. `residual_bias_runtime.js` evaluates both models in the browser, averages their residuals, clips the fused residual to +/-10%, and applies the correction to the unchanged Core P(B).

The checked-in model bundle remains `trained:false` until real labeled data is trained. While `trained:false`, delta remains zero and the frozen V23 core output is preserved.
