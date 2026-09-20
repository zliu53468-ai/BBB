# BBB Blind Shoe Particle Filter -> 8D XGBoost Residual Layer

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
  -> Shoe Particle Filter
  -> pseudo_count
  -> [fixed 7D + pseudo_count] = 8D
  -> XGBoost residual model
  -> residual Delta
  -> clip Delta to +/-0.10
  -> Final P(B)
  -> B / P
```

The original seven upstream features remain unchanged:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

The only downstream feature added for XGBoost is:

8. `pseudo_count`

## Important limitation

The application does not see actual card identities or point totals.

Therefore `pseudo_count` is **not a true remaining-card count**. It is the
weighted expectation across 1000 plausible hidden eight-deck shoes that remain
consistent with the observed B/P sequence under the Monte Carlo model.

## Shoe Particle Filter

Each particle is one virtual remaining shoe represented by point-value counts
for baccarat values 0 through 9.

Fresh eight-deck shoe:

```text
0-point cards = 128
1-point cards = 32
2-point cards = 32
...
9-point cards = 32

total cards = 416
```

The 0-point bin contains 10/J/Q/K.

Configuration:

```text
n_particles = 1000
R = 0.25
ESS threshold = 500
resampling = systematic
random_state = 42
pseudo_count range = [-1.0, +1.0]
```

### Virtual round propagation

For each settled real B/P round, every particle:

1. samples cards without replacement from its own remaining point counts;
2. deals Player/Banker in normal baccarat order;
3. applies natural 8/9 handling;
4. applies Player third-card rules;
5. applies Banker third-card rules;
6. consumes the sampled cards from that virtual shoe;
7. produces a simulated B/P/T result and simulated final point margin.

The particle is then weighted against the real B/P result.

### Core residual likelihood

```text
actual_residual = actual_B - core_p_b
```

The observed B/P direction plus residual magnitude determine how strongly the
real result selects among virtual shoes.

Particles whose simulated outcome and simulated point-margin behavior are more
compatible with the observed result receive higher Gaussian likelihood under
`R=0.25`.

This uses the Core residual only as an observation-strength signal; it does not
alter the frozen Core or fixed 7D features.

### Adaptive particle rejuvenation

Q is used as the fraction of particles that receive one feasible hidden-card
point-bin swap after each observed round. The swap preserves the total number of
remaining cards and keeps each point bin inside the original eight-deck bounds.

```text
current_round < 15:
    Q = 0.005

15 <= current_round <= 45:
    Q transitions linearly from 0.005 to 0.020

current_round > 45:
    Q = 0.020
```

This increases latent-shoe diversity near the tail/cut-card region.

### pseudo_count

After filtering, all weighted particles simulate the next virtual round without
mutating their stored remaining shoe.

```text
banker_mass = weighted forecast mass for B
player_mass = weighted forecast mass for P

pseudo_count =
    (banker_mass - player_mass)
    / (banker_mass + player_mass)
```

Ties are excluded from the denominator.

The result is clipped to `[-1, +1]`.

A new shoe calls `reset()`, restores all 1000 particles to the fresh
eight-deck point counts, and sets `pseudo_count = 0`.

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

Training is replayed shoe by shoe.

For round `t`:

```text
pseudo_count_t = PF state before outcome t is known
features_8d_t = [features_7d_t, pseudo_count_t]
target_t = actual_B_t - core_p_b_t
```

Only after the 8D row is captured does the real result for round `t` update the
1000 virtual shoes and produce `pseudo_count_(t+1)`.

This prevents current-label leakage.

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
pseudo_count = current Shoe Particle Filter estimate
features_8d = [features_7d, pseudo_count]
raw_delta = XGBoost(features_8d)
delta = clip(raw_delta, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0.0, 1.0)
direction = B if final_p_B > 0.50 else P
```

## Modules

```text
shoe_particle_filter.py
  -> 1000 latent eight-deck virtual shoes
  -> sampling without replacement
  -> baccarat drawing rules
  -> B/P observation weighting
  -> pseudo_count

xgb_particle_filter_residual.py
  -> causal historical PF replay
  -> fixed 7D + pseudo_count = 8D
  -> XGBoost residual training/inference/export
  -> Delta +/-0.10 safety clip

residual_bias_runtime.js
  -> browser implementation of the same virtual-shoe PF
  -> persists particle state per shoe
```

The checked-in `residual_bias_model.json` remains `trained:false` until real
labeled B/P training rows are fitted and exported.
