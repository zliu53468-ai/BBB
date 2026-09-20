# BBB Enhanced Shoe PF -> 4D Physical Tensor -> 11D XGBoost

The upstream pipeline is frozen and unchanged:

```text
牌路歷史
-> 256D / V23 Core
-> Core P(B)
-> fixed 7D features
```

Only the downstream correction layer changes.

## Architecture

```text
fixed 7D
  -> EnhancedShoeParticleFilter
  -> [p_4cards, p_6cards, win_point, lose_point]
  -> fixed 7D + physical 4D = 11D
  -> XGBoost residual model
  -> residual Delta
  -> clip Delta to +/-0.10
  -> Final P(B)
  -> B / P
```

The original seven upstream features remain unchanged.

## EnhancedShoeParticleFilter

Each of 1000 particles is one plausible remaining eight-deck shoe represented by
baccarat point-value counts 0..9.

Fresh shoe:

```text
0-point cards = 128
1..9          = 32 each
total         = 416
```

Actual hidden card identities remain unobserved.

### Bayesian likelihood

For particle i and settled round t:

```text
log L_i =
  - 0.5 * information_multiplier
  * normalized_weighted_error_i
  / R
```

with:

```text
R = 0.25

weighted_error =
    0.40 * outcome_error^2
  + 0.22 * total_card_error^2      (when total-card observation exists)
  + 0.23 * point_error^2           (when Player/Banker final points exist)
  + 0.15 * core_residual_error^2
```

The frozen Core contributes only as evidence:

```text
core_residual = actual_B - core_p_b
core_target   = clip(2 * core_residual, -1, +1)
```

A particle's simulated B/P direction and simulated point margin are compared with
that signed Core residual target.

If the real round used five or six total cards:

```text
information_multiplier = 1.5
```

Otherwise:

```text
information_multiplier = 1.0
```

This makes 5/6-card rounds contract the posterior faster when physical
observations are supplied.

If total-card/point observations are unavailable, the PF remains operational
using B/P outcome + Core residual only; it does not invent unseen observations.

## Physical total-card constraint

After likelihood weighting and before resampling:

```text
expected_remaining =
    416 - 4.8 * current_round
```

Particle remaining-card deviation:

```text
z_i =
  abs(remaining_i - expected_remaining)
  / max(3.0, 0.85 * sqrt(current_round))
```

Hard constraint:

```text
if z_i > 3.5:
    weight_i = 0
```

If a hard cutoff would collapse every particle, the implementation falls back
to a soft Gaussian physical prior instead of producing an invalid posterior.

## Adaptive process noise

Q is implemented as latent particle rejuvenation strength:

```text
current_round < 15:
    Q = 0.005

15 <= current_round <= 45:
    Q transitions linearly from 0.005 to 0.025

current_round > 45:
    Q = 0.025
```

## Four-dimensional next-round tensor

Before the next real outcome is known, every weighted particle performs one
forward virtual baccarat rollout.

Let w_i be normalized particle weights.

```text
p_4cards =
  sum_i w_i * I(total_cards_i == 4)

p_6cards =
  sum_i w_i * I(total_cards_i == 6)
```

For decisive B/P virtual rounds:

```text
win_point =
  sum_i w_i * winner_point_i
  / sum_i w_i * I(decisive_i)

lose_point =
  sum_i w_i * loser_point_i
  / sum_i w_i * I(decisive_i)
```

The XGBoost physical feature block is exactly:

```text
[p_4cards, p_6cards, win_point, lose_point]
```

## 11D XGBoost residual model

```text
n_estimators = 75
learning_rate = 0.025
max_depth = 4
min_child_weight = 2.0
alpha = 0.05
lambda = 0.30
random_state = 42
```

Training target:

```text
target = actual_B - core_p_b
```

Causal row construction:

```text
tensor_t = PF tensor BEFORE outcome t is known
features_11d_t = [fixed_features_7d_t, tensor_t]
target_t = actual_B_t - core_p_b_t

only after row t is captured:
    outcome/physical observations t update the PF
    -> tensor_(t+1)
```

This prevents the current result from leaking into its own physical feature.

## Optional high-information observations

Historical or runtime rows may include:

```text
observed_total_cards: 4 | 5 | 6
observed_player_point: 0..9
observed_banker_point: 0..9
```

Browser runtime also exposes:

```js
__BGS_RESIDUAL_BIAS__.setPhysicalObservation({
  totalCards: 6,
  playerPoint: 4,
  bankerPoint: 7
});
```

The next settled B/P result consumes that physical observation.

## Inference

```text
tensor = current Enhanced PF 4D forecast
features_11d = [features_7d, tensor]
raw_delta = XGBoost(features_11d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0.0, 1.0)
direction = B if final_p_B > 0.50 else P
```

The checked-in model bundle stays `trained:false` until real labeled training
rows are fitted and exported.
