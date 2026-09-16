# BBB XGBoost Residual Bias V1.1

BBB remains a static GitHub Pages application. Python/XGBoost trains offline and the browser evaluates the exported trees deterministically.

The architecture stays intentionally narrow:

```text
Frozen 256D / V23 R1 core
        -> existing 7 residual features
        -> XGBRegressor residual
        -> validation-selected delta scale
        -> clip to +/-10%
        -> final P(B)
        -> B if > 50%, otherwise P
```

There is no PASS state. No LightGBM, ensemble, second classifier, or replacement core is introduced.

## Residual target

XGBoost does not directly predict a B/P class. It learns only the error of the frozen core:

```text
residual = actual_B - core_p_B
```

V1.1 production correction is:

```text
raw_delta = xgb(features)
delta = clip(raw_delta * delta_scale, -0.10, +0.10)
final_p_B = clip(core_p_B + delta, 0, 1)
B if final_p_B > 0.50 else P
```

`delta_scale` is selected from out-of-fold validation instead of being guessed manually.

## Exact core binding

The residual model is bound to:

```text
V23_SHORT_X_DYNAMIC_HAZARD_R1
```

The exported JSON contains `base_core_version`. The browser refuses to activate the residual model if the runtime core version and trained model core version do not match.

## Feature order

Python and browser runtime keep the same seven features:

1. `core_p_b` - frozen core Banker probability.
2. `round_index` - next round index, capped to 1..70.
3. `estimated_total_hands` - shoe/cut estimate, default 60 and accepted 40..90.
4. `remaining_ratio` - derived remaining-hand ratio.
5. `sx_markov_p_same` - local first-order S/X probability for SAME.
6. `stage` - current B/P streak length.
7. `depth` - current repeated S/X token depth.

No additional road model is added in V1.1.

## Training-data hygiene

The browser records labeled rows only after a prediction receives an actual B/P result. Tie clears the pending prediction and is not used as a directional training label.

Rows carry the exact `core_version`. Both browser storage and the Python trainer de-duplicate data by:

```text
shoe_id + history_fingerprint
```

Rows from a different core version are excluded by the trainer.

Browser console helpers:

```js
__BGS_RESIDUAL_BIAS__.getTrainingCount()
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
__BGS_RESIDUAL_BIAS__.getModelStatus()
```

Cut/shoe-length estimate:

```js
__BGS_RESIDUAL_BIAS__.setEstimatedTotalHands(60)
```

## V1.1 grouped out-of-fold validation

V1.1 replaces the old single 80/20 split with deterministic grouped cross-validation by `shoe_id`.

Default:

```text
5 folds
4 folds train
1 fold validation
repeat until every fold was validation once
```

All rows from the same shoe remain in the same fold. This prevents one part of a shoe from leaking into training while another part of that same shoe is used for validation.

The five validation outputs are joined into one out-of-fold prediction vector. Only this OOF vector is used to select the residual scale.

Default candidate scales:

```text
0.25, 0.40, 0.55, 0.70, 0.85, 1.00
```

Selection priority:

1. lowest corrected OOF Brier score;
2. highest corrected OOF direction accuracy;
3. smaller scale when tied.

## Flip diagnostics

Validation now reports both probability quality and direction-changing quality:

```text
core_accuracy
corrected_accuracy
accuracy_gain
core_brier
corrected_brier
brier_gain
mean_abs_delta
max_abs_delta
flip_count
flip_rate
flip_win_rate
rescue_count
damage_count
net_flip_gain
```

Definitions:

```text
rescue_count  = core wrong, corrected direction right
damage_count  = core right, corrected direction wrong
net_flip_gain = rescue_count - damage_count
```

The default production gate requires no Brier regression, no direction-accuracy regression, and non-negative `net_flip_gain`.

## Deterministic XGB settings

The model class is still `XGBRegressor`. V1.1 removes row and feature subsampling:

```text
subsample = 1.0
colsample_bytree = 1.0
n_jobs = 1
fixed random_state
```

The regularized tree settings otherwise remain conservative.

## Train and export

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_bias.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500 \
  --folds 5
```

Optional scale grid:

```bash
python xgb_residual_bias.py train \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --delta-scales 0.25,0.40,0.55,0.70,0.85,1.00
```

`--force` is diagnostic only. A model that fails the production validation gate should not be deployed merely because it can be exported.

## Current checked-in model

Until real labeled rows produce a validated model, `residual_bias_model.json` intentionally remains:

```text
trained: false
```

Therefore the effective residual delta remains zero and the current V23 R1 core output is preserved exactly. After a validated model is trained and committed, the runtime automatically applies its exported `delta_scale` and tree bundle.
