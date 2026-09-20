# BBB Blind Physical PF -> 3D Forecast -> 10D XGBoost

Frozen upstream:

```text
牌路歷史
-> 256D / V23 Core
-> Core P(B)
-> fixed 7D features
```

Downstream only:

```text
ShoeParticleFilter
-> pred_card_count
-> pred_banker_point
-> pred_player_point
-> fixed 7D + physical 3D = 10D
-> XGBoost residual
-> clip Delta to +/-0.10
-> Final P(B)
```

## Blind PF

Each of 1000 particles is one plausible remaining eight-deck shoe using point
counts 0..9. Hidden card identities are never observed.

Posterior update uses settled B/P plus the frozen Core residual:

```text
core_residual = actual_B - core_p_b
```

Optional real total-card count and final Banker/Player points may be supplied
only when they genuinely exist in historical/runtime data. Blind mode does not
require them and never fabricates them.

Likelihood weights:

```text
outcome       0.55
total_cards   0.12 (optional evidence)
points        0.18 (optional evidence)
core_residual 0.15
R             0.25
```

Repeated Core-alignment states use a mild persistence multiplier 1.15. If the
Core correct/miss state flips, particle weights are mixed 35% toward uniform
before the next likelihood update, preventing stale posterior concentration.

Process-noise / rejuvenation:

```text
round < 15  : Q = 0.005
round 15-45 : linear 0.005 -> 0.020
round > 45  : Q = 0.020
```

## Next-round physical forecast

Before the real outcome is known, every particle performs one virtual baccarat
round on a copy of its remaining shoe.

The current Core P(B) softly conditions rollout weights:

```text
expected_sign = 2 * core_p_b - 1
core_confidence = abs(expected_sign)

core_factor_i =
exp(
  -0.5
  * core_forecast_strength
  * core_confidence
  * (simulated_sign_i - expected_sign)^2
  / R
)
```

with:

```text
core_forecast_strength = 0.35
```

This conditioning becomes weak when Core is near 0.50.

The three physical expectations are:

```text
pred_card_count   = E[next total cards]   in [4, 6]
pred_banker_point = E[next Banker total] in [0, 9]
pred_player_point = E[next Player total] in [0, 9]
```

The forecast is side-effect-free: RNG state and stored particles are unchanged.

## 10D XGBoost

```text
n_estimators = 65
learning_rate = 0.03
max_depth = 4
min_child_weight = 2.0
alpha = 0.05
lambda = 0.25
random_state = 42
```

Training target:

```text
actual_B - core_p_b
```

Causal training:

```text
physical_3d_t = PF forecast using posterior before outcome t
features_10d_t = [fixed_7d_t, physical_3d_t]
target_t = actual_B_t - core_p_b_t

then outcome t updates PF for t+1
```

Inference:

```text
raw_delta = XGBoost(features_10d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0, 1)
```

The checked-in model remains `trained:false` until real labeled historical
rows are available and pass the validation gate.
