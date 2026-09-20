# BBB PF Base-Margin -> 7D XGBoost Residual Layer

The upstream pipeline is frozen and unchanged:

```text
牌路歷史
-> 256D / V23 Core
-> Core P(B)
-> fixed 7D features
```

The Particle Filter no longer occupies an XGBoost feature dimension.

## Architecture

```text
fixed 7D
  + Shoe Particle Filter -> pf_delta
  -> xgb.DMatrix(fixed 7D)
  -> set_base_margin(pf_delta)
  -> XGBoost Booster trees learn residual correction above the PF prior
  -> Booster.predict(DMatrix) = total residual Delta
  -> clip total Delta to +/-0.10
  -> Final P(B)
  -> B / P
```

XGBoost still sees exactly these seven frozen features:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `sx_markov_p_same`
6. `stage`
7. `depth`

There is no eighth XGBoost feature.

## Shoe Particle Filter

The application does not observe actual card identities. Each of 1000 particles
is one plausible latent eight-deck remaining shoe represented by baccarat point
counts 0..9.

Fresh eight-deck point counts:

```text
0-point = 128 cards
1..9    = 32 cards each
total   = 416 cards
```

The PF:

- samples virtual rounds without replacement;
- follows baccarat Player/Banker drawing rules;
- reweights latent shoes from the observed B/P result and Core residual;
- resamples when ESS < 500;
- uses Q=0.005 before round 15 and Q=0.02 after round 45, with a linear transition between;
- resets to a fresh eight-deck latent shoe set on a new shoe.

The next-round weighted B/P physical bias is converted to:

```text
pf_delta = clip(0.10 * physical_bias, -0.10, +0.10)
```

`pf_delta` is a probabilistic physical prior, not knowledge of the true
remaining cards.

## Native XGBoost base_margin

Training target remains:

```text
target = actual_B - core_p_b
```

For every historical round t:

```text
pf_delta_t = PF state before outcome t is known
dtrain = xgb.DMatrix(features_7d_t, label=target_t)
dtrain.set_base_margin(pf_delta_t)
```

The current result is used to update the PF only after the row has been captured,
so the current label cannot leak into its own base margin.

Prediction uses the same contract:

```python
dtest = xgb.DMatrix(features_7d)
dtest.set_base_margin(np.asarray([current_pf_delta], dtype=np.float32))

total_delta = booster.predict(dtest)[0]
delta_clipped = np.clip(total_delta, -0.10, +0.10)
final_pb = np.clip(core_pb + delta_clipped, 0.0, 1.0)
```

For `reg:squarederror`, the supplied base margin is the initial prediction
margin, and the boosting trees learn corrections on top of that prior.

## XGBoost parameters

Native `xgb.train` is used rather than sklearn `XGBRegressor.fit`.

```text
num_boost_round = 50
eta / learning_rate = 0.02
max_depth = 3
alpha = 0.1
lambda = 0.3
seed = 42
objective = reg:squarederror
tree_method = hist
```

## Browser runtime

The browser does not run the native XGBoost C++ DMatrix API. The exported tree
runtime reproduces the same semantics:

```text
total_delta = current_pf_delta + sum(exported_tree_leaf_outputs)
```

The exported model is validated in Python before writing the bundle: native
`Booster.predict(DMatrix with base_margin)` must match
`pf_delta + portable_tree_sum` within tolerance.

## Modules

```text
shoe_particle_filter.py
  -> 1000 latent virtual shoes
  -> output pf_delta [-0.10,+0.10]

xgb_particle_filter_residual.py
  -> fixed 7D DMatrix
  -> historical/current pf_delta via set_base_margin()
  -> native xgb.train / Booster.predict
  -> final residual clip +/-0.10

residual_bias_runtime.js
  -> same PF simulation in browser
  -> exported-tree equivalent of PF base-margin prediction
```

The checked-in `residual_bias_model.json` remains `trained:false` until real
labeled B/P rows are trained and exported.
