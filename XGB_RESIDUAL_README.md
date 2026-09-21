# BBB 11D Anomaly Brake XGBoost Residual Layer

The upstream pipeline remains frozen:

```text
牌路歷史
-> 256D / V23 Core
-> Core P(B)
-> fixed 7D
```

Only the downstream correction layer changes.

## Architecture

```text
fixed 7D
-> ShoeParticleFilter
-> pred_card_count
-> pred_banker_point
-> pred_player_point
-> anomaly_score
-> 7D + 4D = 11D
-> XGBoost residual
-> deterministic anomaly brake
-> clip +/-0.10
-> Final P(B)
-> B / P
```

There is no PASS/standby output.

## Change-point anomaly score

The PF keeps recent settled B/P directions internally.

A structural break after a run of at least three same-side outcomes raises
`anomaly_score` to at least 0.95. Two-run breaks produce a medium anomaly, and
strong alternation keeps the score elevated. A Core miss contributes a smaller
confidence-weighted anomaly component.

```text
anomaly_t =
max(
  structural_break,
  alternation_signal,
  core_miss_signal,
  0.55 * anomaly_(t-1)
)
```

Stable periods therefore decay toward zero quickly.

## PF posterior collapse

When anomaly is high, the previous concentrated particle posterior is partially
mixed back toward uniform:

```text
mix = 0.92 * anomaly_score^2

w_i <- (1-mix) * w_i + mix / N
```

The likelihood is also tempered:

```text
precision =
max(0.15, 1 - 0.85 * anomaly_score^2)
```

This reduces the chance that a long-run posterior continues chasing the old
regime after a sudden break.

## Adaptive process noise

```text
round < 15:
  Q = 0.005

15 <= round <= 45:
  Q linearly rises from 0.005 to 0.030

round > 45:
  Q = 0.030
```

## 11D features

The first seven features are unchanged:

1. core_p_b
2. round_index
3. estimated_total_hands
4. remaining_ratio
5. sx_markov_p_same
6. stage
7. depth

PF adds:

8. pred_card_count
9. pred_banker_point
10. pred_player_point
11. anomaly_score

## XGBoost

```text
n_estimators = 75
learning_rate = 0.025
max_depth = 4
min_child_weight = 2.0
alpha = 0.10
lambda = 0.30
random_state = 42
```

Regularization helps control overfitting, but it does not mathematically
guarantee that high anomaly produces zero residual. Therefore the runtime adds
an explicit smooth brake.

## Deterministic anomaly brake

```text
anomaly <= 0.35:
  brake_factor = 1

0.35 < anomaly < 0.90:
  brake_factor = 1 - smoothstep(0.35, 0.90, anomaly)

anomaly >= 0.90:
  brake_factor = 0
```

Then:

```text
raw_delta = XGBoost(features_11d)
braked_delta = raw_delta * brake_factor
delta = clip(braked_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0, 1)
```

Thus an extreme change-point returns the correction to the frozen Core without
creating a PASS state.

## Causal training

For round t:

```text
pf_4d_t = PF state before outcome t
features_11d_t = [fixed_7d_t, pf_4d_t]
target_t = actual_B_t - core_p_b_t
```

Only after row t is captured does outcome t update the PF and anomaly state for
round t+1.

## Validation

Held-out shoe validation reports Core vs corrected Accuracy and Brier, plus:

- mean anomaly score
- high-anomaly fraction
- mean absolute raw Delta
- mean absolute braked Delta
- high-anomaly mean absolute Delta

The checked-in model remains `trained:false` until real historical labeled
rows are trained and pass validation.
