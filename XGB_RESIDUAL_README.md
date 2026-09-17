# BBB XGBoost + Particle Filter Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

Only the downstream residual correction layer is expanded. No LightGBM or LSTM is used.

## Residual target

The XGBoost model still learns:

```text
residual = actual_B - core_p_B
```

The seven features are unchanged and remain in the exact same order:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

## XGBoost configuration

```text
n_estimators=50
learning_rate=0.02
max_depth=3
min_child_weight=3.5
alpha=0.1
lambda=0.4
random_state=42
```

Python uses the sklearn aliases `reg_alpha=0.1` and `reg_lambda=0.4`.

## Particle Filter configuration

The particle filter tracks one latent state: the current shoe's hidden residual drift.

```text
n_particles=1000
state_dim=1
Q=0.005
R=0.25
resample_threshold=500
resampling=systematic
```

A new shoe resets the particle filter. After each actual B/P result, the browser computes `actual_B - core_p_b`, updates particle weights, performs systematic resampling when effective sample size is below 500, then applies one process transition to project the latent state for the next round. Ties do not update the directional residual filter.

## Fusion inference

```text
delta_xgb = XGBoost(features_7d)
delta_pf = current particle-filter latent residual estimate
delta_final = (delta_xgb + delta_pf) / 2.0
delta_clipped = clip(delta_final, -0.10, +0.10)
final_p_B = clip(core_p_B + delta_clipped, 0.0, 1.0)
B if final_p_B > 0.50 else P
```

There is no PASS state.

## Training and online update

XGBoost is trained offline on the existing browser-exported 7D rows:

```bash
python -m pip install -r requirements-xgb.txt
python xgb_particle_filter_residual.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The particle filter is not batch-fit. It updates online inside `residual_bias_runtime.js` as the current shoe advances.

The trainer keeps the existing deterministic shoe-level validation split and validates the fused XGB + PF correction. The checked-in model bundle remains `trained:false` until real labeled data is trained. While XGBoost is untrained, the runtime continues collecting/updating PF state but does not apply downstream correction to the displayed prediction, preserving the current V23 core output.

Browser console helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getParticleFilterStatus()
__BGS_RESIDUAL_BIAS__.getParticleFilterEstimate()
__BGS_RESIDUAL_BIAS__.resetParticleFilter()
```
