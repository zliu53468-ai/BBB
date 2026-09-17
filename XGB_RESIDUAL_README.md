# BBB Dual Residual Ensemble — XGBoost + LightGBM

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

Only the residual correction layer is upgraded.

## Residual target

Both models train on the exact same 7D feature matrix and the exact same label:

```text
residual = actual_B - core_p_B
```

XGBoost keeps the original V1 training logic and parameters in `xgb_residual_bias.py`.
LightGBM is added in `dual_residual_ensemble.py`.

## LightGBM short-shoe parameters

```text
n_estimators=50
learning_rate=0.01
max_depth=3
num_leaves=7
min_data_in_leaf=3
bagging_fraction=0.7
bagging_freq=1
feature_fraction=0.8
verbosity=-1
random_state=42
```

`bagging_freq=1` activates the requested `bagging_fraction=0.7` sampling.

## 7D feature order

The feature dimension is unchanged:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

No additional road feature is added.

## Fusion inference

```text
delta_xgb = XGBoost(features_7d)
delta_lgb = LightGBM(features_7d)
delta_final = (delta_xgb + delta_lgb) / 2.0
delta_clipped = clip(delta_final, -0.10, +0.10)
final_p_B = clip(core_p_B + delta_clipped, 0.0, 1.0)
B if final_p_B > 0.50 else P
```

There is no PASS state.

## Training

Install dependencies:

```bash
python -m pip install -r requirements-xgb.txt
```

Train both residual models from the same browser-exported rows:

```bash
python dual_residual_ensemble.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The existing deterministic shoe-level validation split is reused. Validation reports Core, XGBoost-only, LightGBM-only and fused accuracy/Brier. The production gate is applied to the fused prediction.

## Static browser deployment

`dual_residual_ensemble.py` exports both tree sets into `residual_bias_model.json`.
`residual_bias_runtime.js` evaluates XGBoost and LightGBM in parallel in the browser, takes the arithmetic mean, clips the fused residual to +/-10%, and applies it to the unchanged Core P(B).

The checked-in model bundle remains `trained:false` until real labeled data is trained. While `trained:false`, delta stays zero and the frozen V23 core output is preserved.
