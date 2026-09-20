# BBB Shoe Regime PF -> 8D XGBoost Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 Core -> Core P(B) -> fixed 7D features
```

Only the downstream correction layer is modified. The Particle Filter does not perform card counting, card-composition estimation, or remaining-card inference.

## Architecture

```text
fixed 7D
  -> Shoe Regime Particle Filter
  -> regime_state
  -> [fixed 7D + regime_state] = 8D
  -> XGBoost residual model
  -> residual Delta
  -> clip Delta to +/-0.10
  -> Final P(B)
  -> B / P
```

The fixed upstream seven features remain:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

The only downstream feature added for XGBoost is:

8. `regime_state`

## Shoe Regime Particle Filter

```text
n_particles = 1000
state_dim = 1
R = 0.25
ESS threshold = 500
resampling = systematic
state_clip = [-1.0, +1.0]
random_state = 42
```

A new shoe calls `reset()`. All particles are initialized exactly at zero and `regime_state = 0`.

### Three-dimensional likelihood

After each settled B/P round:

```text
actual_residual = actual_B - core_p_b

directionality
= +1 if Core direction matched the result
= -1 otherwise

residual_alignment
= clip(1 - 2 * abs(actual_residual), -1, +1)

persistence
= current directionality if the correct/miss state repeats
= 0 otherwise
```

Likelihood weights:

```text
directionality     0.55
residual_alignment 0.30
persistence        0.15
```

Each particle is scored against all three measurements using Gaussian likelihood with `R=0.25`.

If the correct/miss state suddenly flips, the round is treated as a turbulence break:

```text
measurements = [0, 0, 0]
particle weights are reset to uniform
then the neutral R=0.25 likelihood is applied
```

This pulls the environment estimate back toward the neutral turbulence zone instead of immediately declaring a new persistent regime.

### Round-adaptive process noise

```text
current_round < 15:
    Q = 0.005

15 <= current_round <= 45:
    Q transitions linearly from 0.005 to 0.020

current_round > 45:
    Q = 0.020
```

The round number controls only PF responsiveness. It does not directly choose Banker or Player.

## 8D XGBoost residual model

```text
n_estimators = 65
learning_rate = 0.03
max_depth = 4
min_child_weight = 2.0
alpha = 0.05
lambda = 0.25
random_state = 42
```

Python uses `reg_alpha=0.05` and `reg_lambda=0.25`.

## Causal training

Historical training is replayed shoe by shoe.

For round `t`:

```text
regime_state_t = PF state before outcome t is known
features_8d_t = [features_7d_t, regime_state_t]
target_t = actual_B_t - core_p_b_t
```

Only after the 8D row is captured does outcome `t` update the PF for round `t+1`. This prevents current-label leakage.

Training:

```bash
python -m pip install -r requirements-xgb.txt
python xgb_particle_filter_residual.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

## Inference

```text
regime_state = current PF estimate
features_8d = [features_7d, regime_state]
raw_delta = XGBoost(features_8d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0.0, 1.0)
direction = B if final_p_B > 0.50 else P
```

## Module separation

```text
shoe_regime_filter.py
  -> Shoe Regime PF only
  -> outputs regime_state

xgb_particle_filter_residual.py
  -> causal 8D training
  -> XGBoost residual fit/predict/export
  -> Delta +/-0.10 safety clip

residual_bias_runtime.js
  -> browser runtime implementation matching the same PF rules
```

The checked-in `residual_bias_model.json` remains `trained:false` until real labeled B/P rows are trained and exported.
