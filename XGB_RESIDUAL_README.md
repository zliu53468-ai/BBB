# BBB Shoe Regime Filter -> 8D XGBoost Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史 -> 256D/V23 core -> Core P(B) -> fixed 7D features
```

The Particle Filter is now used only as a **shoe-environment state tracker** after the fixed 7D output. It is not a card-composition or remaining-card counter, and it does not modify the 256D/V23 core or the definitions of the original seven features.

## Downstream architecture

```text
fixed 7D features
      +
Shoe Regime Filter -> regime_state
      -> 8D model vector
      -> XGBoost residual prediction
      -> clip residual to +/-0.10
      -> Final P(B)
      -> B if > 0.50 else P
```

There is no LightGBM, no LSTM, no parallel XGB/PF averaging, and no PASS state.

## Fixed upstream 7D

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

XGBoost receives one downstream latent feature:

8. `regime_state`

`regime_state` is bounded to approximately `[-1,+1]`:

```text
+1  = sustained Core-aligned / structured regime
 0  = turbulence, mixed evidence, or insufficient evidence
-1  = sustained Core-opposed / degraded regime
```

The state is environmental context. It never directly chooses Banker or Player.

## Shoe Regime Particle Filter

```text
n_particles=1000
state_dim=1
Q=0.005
R=0.25
resample_threshold=500
resampling=systematic
random_state=42
state_clip=1.0
```

Each settled B/P round creates a regime observation from:

```text
direction_alignment  weight 0.55
confidence_alignment weight 0.30
persistence          weight 0.15
```

- `direction_alignment`: +1 when the frozen Core direction matches the result, -1 otherwise.
- `confidence_alignment`: rewards correctly aligned probability margin and penalizes confident misses.
- `persistence`: reinforces only repeated relationships; repeated correct alignment is positive, repeated misses are negative, and flips add zero persistence.

Mixed or alternating behavior therefore tends to cancel toward zero instead of becoming a fake trend.

## Cut-card / shoe-depth handling

Cut-card position is treated as an **environment lifecycle variable**, not a Banker/Player signal.

The existing upstream `round_index`, `estimated_total_hands`, and `remaining_ratio` are unchanged. Inside the Shoe Regime Filter, shoe progress only scales process noise:

```text
shoe start: Q x 0.75
shoe tail : Q x 1.50
```

The state is steadier early in a shoe and stale regime assumptions can decay faster near the cut-card tail. A new shoe or shuffle resets `regime_state` to zero.

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

## Causal training

For round `t`:

```text
regime_state_t = filter state before outcome t is known
features_8d_t = [features_7d_t, regime_state_t]
y_t = actual_B_t - core_p_b_t
```

Only after `features_8d_t` is recorded does outcome `t` update the Shoe Regime Filter for round `t+1`. This prevents current-result leakage.

```bash
python -m pip install -r requirements-xgb.txt
python xgb_particle_filter_residual.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The deterministic shoe-level validation split remains. The checked-in model bundle stays `trained:false` until real labeled rows are trained and exported.

## Browser runtime

```text
regime_state = current Shoe Regime Filter estimate
features_8d = [features_7d, regime_state]
raw_delta = XGBoost(features_8d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0.0, 1.0)
```

Ties are non-directional and do not update the regime observation.

Browser helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getShoeRegimeState()
__BGS_RESIDUAL_BIAS__.getShoeRegimeStatus()
__BGS_RESIDUAL_BIAS__.resetShoeRegimeFilter()
```
