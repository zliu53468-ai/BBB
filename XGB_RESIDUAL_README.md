# BBB XGBoost Residual Bias + Dynamic Damper

BBB is deployed as static GitHub Pages, so Python/XGBoost training runs offline while the browser evaluates the exported trees deterministically.

The production chain is now:

```text
256D/V23 R1 core
  -> XGBRegressor residual delta
  -> 8-token S/X entropy dynamic damper
  -> final B/P probability
  -> bet-weight label
```

The XGBoost model still does **not** learn raw B/P classes. The regression target remains:

```text
residual = actual_B - core_p_B
```

## Feature schema

The Python trainer and browser runtime use the same eight features:

1. `core_p_b` - current deterministic core B probability.
2. `round_index` - next round index, capped to 1..70.
3. `estimated_total_hands` - cut/shoe-length estimate, default 60.
4. `remaining_ratio` - remaining shoe proportion derived from round index.
5. `sx_markov_p_same` - local first-order S/X Markov probability of next token being SAME.
6. `stage` - current B/P streak length.
7. `depth` - current repeated S/X token depth.
8. `sx_entropy_8` - normalized Shannon entropy of the most recent eight S/X transition tokens.

For fewer than four available S/X tokens, entropy is forced to `0.0` so the opening hands are not aggressively damped by a tiny sample.

## Dynamic damper

Production probability is:

```text
delta = clip(xgb_residual, -0.10, +0.10)
final_p_B = 0.50 + ((core_p_B - 0.50) + delta) * damper
B if final_p_B > 0.50 else P
```

Damper mapping:

```text
entropy <= 0.45      -> damper = 1.0
0.45 < entropy < .85 -> linearly falls from 1.0 to 0.4
0.85 <= entropy <= 1 -> linearly falls from 0.4 to 0.2
```

There is no PASS state. The damper only shrinks the adjusted edge toward 50%; it does not create a third action.

## Bet-weight output

The browser and Python class also return a continuous `bet_weight` from `0.20` to `1.00`, based on the final distance from 50%:

```text
edge = abs(final_p_B - 0.50)
bet_weight = 0.20 + 0.80 * clip(edge / 0.18, 0, 1)
```

Tiers:

```text
LOW    < 0.40
MEDIUM < 0.70
HIGH   >= 0.70
```

The web UI shows the weight after each analysis. This is a sizing signal only; B/P is still determined solely by the final 50% threshold.

## Python API

`xgb_residual_bias.py` exposes `DynamicResidualBiasPredictor` with:

- feature assembly via `build_features()`
- XGB residual training/prediction
- `dynamic_damper()`
- final B/P correction
- continuous bet-weight output

`ResidualBiasPredictor` remains as a backwards-compatible alias.

## Existing V1 training rows

The browser keeps the original local-storage training key so previously collected rows are not discarded. Old rows do not contain `sx_entropy_8`, but the Python loader reconstructs it from `history_fingerprint` during training.

## Collect labeled rows

From the browser console:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
```

Set the current estimated shoe length when needed:

```js
__BGS_RESIDUAL_BIAS__.setEstimatedTotalHands(60)
```

## Train and export

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_bias.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500
```

The trainer uses deterministic shoe-level validation, `n_jobs=1`, a fixed random state, and rejects a production export when held-out corrected Brier/accuracy gates regress unless `--force` is explicitly supplied for diagnostics.

The checked-in placeholder remains `trained:false` until enough labeled data exists. While XGBoost is untrained, `delta=0`, but the entropy damper and bet-weight output are already active and deterministic.
