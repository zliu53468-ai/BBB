# BBB Particle Filter State -> 8D XGBoost Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

The Particle Filter is inserted only after the fixed 7D features. It does not replace or modify the 256D/V23 core and it does not alter the original seven feature definitions.

## Serial downstream architecture

```text
fixed 7D features
      +
current pf_state
      -> 8D model vector
      -> XGBoost residual prediction
      -> clip residual to +/-0.10
      -> Final P(B)
      -> B if > 0.50 else P
```

There is no LightGBM, no LSTM, no parallel 50/50 XGB/PF averaging, and no PASS state.

## Fixed upstream 7D

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

The downstream XGBoost model sees one additional feature:

8. `pf_state` - the current shoe's causal Particle Filter latent residual state.

## Particle Filter

```text
n_particles=1000
state_dim=1
Q=0.005
R=0.25
resample_threshold=500
resampling=systematic
random_state=42
```

A new shoe resets the Particle Filter to a zero-centered 1000-particle state. For round t, `pf_state_t` is read before the outcome of round t is known. After the actual B/P result arrives, the observation `actual_B - core_p_b` updates the PF, systematic resampling runs when ESS < 500, and one process transition projects the state used by round t+1. Ties do not update the directional residual filter.

## XGBoost

```text
n_estimators=65
learning_rate=0.03
max_depth=4
min_child_weight=2.0
alpha=0.05
lambda=0.2
random_state=42
```

Python uses `reg_alpha=0.05` and `reg_lambda=0.2`.

## Training

Training replays every shoe causally. Each row is built as:

```text
features_8d_t = [features_7d_t, pf_state_t]
y_t = actual_B_t - core_p_b_t
```

Only after `features_8d_t` is recorded is `y_t` fed back into the PF to prepare `pf_state_(t+1)`. This prevents the current label from leaking into its own PF feature.

```bash
python -m pip install -r requirements-xgb.txt
python xgb_particle_filter_residual.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The existing shoe-level validation split remains in place. The checked-in model bundle stays `trained:false` until real labeled rows are trained and exported.

## Browser runtime

At prediction time:

```text
pf_state = current Particle Filter estimate
features_8d = [features_7d, pf_state]
raw_delta = XGBoost(features_8d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0.0, 1.0)
```

Browser helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getParticleFilterStatus()
__BGS_RESIDUAL_BIAS__.getParticleFilterEstimate()
__BGS_RESIDUAL_BIAS__.resetParticleFilter()
```
