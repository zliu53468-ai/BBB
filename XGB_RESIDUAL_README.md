# BBB XGBoost Residual Bias Layer

This repository is deployed as static GitHub Pages, so Python/XGBoost cannot run directly inside the browser. The integration is split into two deterministic parts:

1. `xgb_residual_bias.py` trains `XGBRegressor` offline on labeled B/P outcomes and exports the trees to `residual_bias_model.json`.
2. `residual_bias_runtime.js` evaluates that exported tree bundle in the browser and applies the bounded residual correction to the existing deterministic V23 R1 core.

The 256D/V23 R1 core remains the base predictor. XGBoost does **not** replace it and does **not** directly train on a B/P class target. The regression target is:

```text
residual = actual_B - core_p_B
```

Production correction:

```text
delta = clip(xgb_residual, -0.10, +0.10)
final_p_B = core_p_B + delta
B if final_p_B > 0.50 else P
```

There is no PASS state.

## Feature order

The browser and Python trainer use the exact same seven features:

1. `core_p_b` - current deterministic core B probability.
2. `round_index` - next round index, capped to 1..70.
3. `estimated_total_hands` - cut/shoe-length estimate (default 60; accepted 40..90).
4. `remaining_ratio` - derived from round index and estimated shoe length.
5. `sx_markov_p_same` - local first-order S/X Markov probability of next token being SAME.
6. `stage` - current B/P streak length.
7. `depth` - current repeated S/X token depth.

## Collect labeled production rows

The browser runtime stores local labeled rows after a prediction is followed by an actual B/P result. Ties are non-directional and are not used as labels.

From the browser console:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
```

To set the current cut/shoe-length estimate:

```js
__BGS_RESIDUAL_BIAS__.setEstimatedTotalHands(60)
```

The value is saved in local storage for subsequent predictions.

## Train and export

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_bias.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The trainer uses a deterministic shoe-level validation split, `n_jobs=1`, fixed random state, and a conservative regularized `XGBRegressor`. It refuses to export a production model when held-out Brier/accuracy gates regress unless `--force` is explicitly used for diagnostics.

After a validated `residual_bias_model.json` is committed, the static BBB page loads it automatically. Until then the checked-in placeholder has `trained:false`, so `delta=0` and the existing V23 R1 output is preserved exactly.
