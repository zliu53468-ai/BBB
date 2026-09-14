# BBB XGBoost Residual Bias + Dual-Window Dynamic Damper

The deterministic `256D / V23 R1` core remains the base predictor. This layer is external and does not modify `app256forward.js` or `app256continuation.js`.

BBB is deployed through static GitHub Pages, so training happens offline with Python/XGBoost, while the browser evaluates the exported XGBoost trees deterministically.

## Production chain

```text
256D / V23 R1 core
  -> XGBRegressor residual delta
  -> 18-token S/X entropy damper
  -> final B/P probability
  -> relative bet-weight signal
```

XGBoost does **not** train on a raw B/P classifier target. The target is:

```text
residual = actual_B - core_p_B
```

The residual is clipped to `[-0.10, +0.10]`.

## V3 feature schema

Python and browser use the same eight features, in this exact order:

1. `core_p_b`
2. `round_index`
3. `estimated_total_hands`
4. `remaining_ratio`
5. `stage`
6. `depth`
7. `sx_entropy_18`
8. `sx_transition_change_6`

### Long window: `sx_entropy_18`

Normalized Shannon entropy of the most recent 18 S/X transition tokens.

- near `0`: one S/X state dominates, so the regime is comparatively structured
- near `1`: S and X are close to balanced, so the regime is treated as highly oscillatory

Before 6 S/X tokens exist, entropy is forced to `0.0` so an opening tiny sample cannot trigger aggressive damping.

### Short window: `sx_transition_change_6`

The latest six S/X tokens are split into:

```text
older 3 | newer 3
```

The feature is:

```text
newer X-frequency - older X-frequency
```

Range: `[-1, +1]`.

Positive values mean local switching frequency is accelerating. Negative values mean the local road is becoming more persistent. This short-window feature goes into XGBoost; it does not directly control the deterministic damper.

The old `sx_markov_p_same` and `sx_entropy_8` are not part of the V3 model schema.

## Dynamic damper

Production probability:

```text
delta = clip(xgb_residual, -0.10, +0.10)

final_p_B =
    0.50
    + ((core_p_B - 0.50) + delta)
    * damper
```

Default 18-token entropy mapping:

```text
entropy <= 0.55      -> damper = 1.0
0.55 < entropy < .85 -> linearly falls from 1.0 to 0.4
0.85 <= entropy <= 1 -> linearly falls from 0.4 to 0.2
```

The damper is always positive, so it only contracts the adjusted edge toward 50%. Direction changes can occur when the XGBoost residual itself moves the adjusted edge across 50%.

There is no PASS state:

```text
final_p_B > 0.50 -> B
final_p_B <= 0.50 -> P
```

## Bet-weight output

The output also contains a continuous relative sizing signal:

```text
edge = abs(final_p_B - 0.50)

bet_weight =
    0.20
    + 0.80 * clip(edge / 0.18, 0, 1)
```

Default tiers:

```text
LOW    < 0.40
MEDIUM < 0.70
HIGH   >= 0.70
```

`0.20` is the relative bottom-weight floor. This layer does not define a currency amount.

## Python API

`xgb_residual_bias.py` exposes:

```python
DualWindowResidualBiasPredictor
```

It includes:

- `assemble_features()`
- `fit()`
- `predict_delta()`
- `damper()`
- `correct()`
- `predict_from_context()`

For compatibility, these names still resolve to the same V3 class:

```python
DynamicResidualBiasPredictor
ResidualBiasPredictor
```

## Existing V1/V2 training rows

The browser intentionally retains the original local-storage training key.

Older rows that do not contain `sx_entropy_18` or `sx_transition_change_6` can still be used because the Python trainer rebuilds the new V3 features from `history_fingerprint`. Old `sx_markov_p_same` / `sx_entropy_8` values are ignored by the V3 feature vector.

## Collect labeled rows

Browser console:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
```

Optional estimated shoe length:

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

Training uses:

- deterministic shoe-level validation split
- fixed random state
- `n_jobs=1`
- conservative regularization
- held-out Brier and directional-accuracy gates

The checked-in model placeholder remains `trained:false` until enough real labeled B/P rows exist. Until then XGBoost Delta is `0`, while the deterministic 18-token damper and bet-weight output still operate.
