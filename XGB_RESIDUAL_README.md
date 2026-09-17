# BBB Dual Residual Ensemble — XGBoost + LightGBM

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

Only the residual correction layer is tuned.

## Residual target

Both models train on the exact same 7D feature matrix and the exact same label:

```text
residual = actual_B - core_p_B
```

## High-sensitivity XGBoost parameters

```text
n_estimators=75
learning_rate=0.035
max_depth=4
min_child_weight=1.5
alpha=0.01
lambda=0.05
subsample=0.85
colsample_bytree=0.9
random_state=42
```

In the Python API, XGBoost regularization is passed as `reg_alpha=0.01` and `reg_lambda=0.05`.

## High-sensitivity LightGBM parameters

```text
n_estimators=75
learning_rate=0.035
max_depth=4
num_leaves=10
min_data_in_leaf=2
reg_alpha=0.01
reg_lambda=0.05
bagging_fraction=0.85
feature_fraction=0.9
bagging_freq=1
verbosity=-1
random_state=42
```

`bagging_freq=1` keeps the requested `bagging_fraction=0.85` sampling active.

## 7D feature order

The feature dimension is unchanged:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

No additional feature is added.

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

```bash
python -m pip install -r requirements-xgb.txt
python dual_residual_ensemble.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The same `X_train` and the same residual `y_train` are passed to XGBoost and LightGBM. The deterministic shoe-level validation split is reused. Validation reports Core, XGBoost-only, LightGBM-only and fused accuracy/Brier, and the production gate is applied to the fused result.

## Static browser deployment

`dual_residual_ensemble.py` exports both tree sets into `residual_bias_model.json`.
`residual_bias_runtime.js` evaluates XGBoost and LightGBM in parallel, takes the 50/50 arithmetic mean, clips the fused residual to +/-10%, and applies it to the unchanged Core P(B).

The checked-in model bundle remains `trained:false` until real labeled data is trained. While `trained:false`, delta stays zero and the frozen V23 core output is preserved.
