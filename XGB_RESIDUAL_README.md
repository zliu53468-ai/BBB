# BBB Core-aware Residual V2

Production architecture remains:

```text
牌路歷史
→ 256D / V23 Core
→ Core P(B)
→ 固定 7D
→ XGBoost residual correction
→ adaptive Delta clip (max ±10%)
→ Tie shrink
→ Platt calibration
→ Final P(B)
→ 莊 / 閒
```

There is no PASS branch. Every hand still receives a B/P prediction.

## What changed

### 1. Strict chronological shoe walk-forward

Training no longer uses hash/random validation.

Each fold is:

```text
past
Train shoes → Calibration shoes → Test shoes
                                      future
```

A shoe is atomic and can never be split across partitions.

The trainer rejects:

- missing `shoe_id`
- missing prediction timestamp
- duplicate `(shoe_id, history_fingerprint)`
- non-increasing `round_index` inside a shoe

Default walk-forward settings:

```text
min_train_shoes = 8
calibration_shoes = 2
test_shoes = 2
step_shoes = 2
```

These are CLI-configurable but the order is always chronological.

### 2. Core logit as XGBoost base_margin

The displayed V23 probability remains unchanged.

The Core now also exposes its pre-display-clip probability and logit:

```text
core_raw_p_b
core_logit
```

The fixed 7D feature vector remains unchanged and still includes `core_p_b`
as its first feature.

The XGBoost objective is now:

```text
binary:logistic
eval_metric = logloss
```

and each non-tie row is trained with:

```text
base_margin = core_logit
```

Thus XGBoost learns an additive correction in log-odds space while the browser
still exposes the correction as a probability-space residual:

```text
p_xgb = sigmoid(core_logit + xgb_margin_correction)

raw_delta = p_xgb - displayed_core_p_b
delta = clip(raw_delta, adaptive_limit)
residual_p_b = displayed_core_p_b + delta
```

### 3. Stronger regularization

Current XGBoost defaults:

```text
num_boost_round = 180
eta = 0.025
max_depth = 3
min_child_weight = 12
reg_alpha = 0.30
reg_lambda = 12.0
max_delta_step = 1.0
subsample = 0.85
colsample_bytree = 0.90
```

### 4. Adaptive residual clip

The hard safety ceiling remains ±10%, but the allowed correction becomes
smaller when Core is close to 0.50:

```text
limit =
    min_delta
    + (max_delta - min_delta)
      * clip(abs(core_p_b - 0.5) / confidence_span, 0, 1)
```

Defaults:

```text
min_delta = 0.025
max_delta = 0.10
confidence_span = 0.08
```

Therefore Core near 50% can only be corrected modestly, while a Core near
42%/58% may use the full ±10% safety range.

### 5. Tie handling

The main B/P model trains and evaluates only:

```text
P(B | non-tie)
```

Ties are still stored in the training data and are used by a separate small
regularized logistic model:

```text
P(Tie | fixed 7D)
```

When predicted Tie probability is unusually high relative to its training
baseline, the B/P probability is smoothly shrunk toward 0.50. It never creates
a PASS output.

### 6. Platt calibration

Calibration is fit only on the calibration shoes, never on test shoes.

```text
p_final =
sigmoid(
    slope * logit(p_after_tie_shrink)
    + intercept
)
```

The calibrated `Final P(B)` is then thresholded at 0.50 for B/P.

### 7. Metrics

Every walk-forward test window reports:

- non-tie accuracy
- LogLoss
- Brier score
- equal-frequency ECE
- probability-bin predicted vs observed Banker rate
- Tie-model LogLoss / Brier
- mean residual Delta
- mean adaptive Delta limit

The report always compares:

```text
Core only
vs
Final calibrated residual model
```

### 8. 256D audit

The browser now stores the Core 256D vector with each labeled prediction when
available.

The offline trainer audits only the chronological training shoes for:

- zero / near-zero variance features
- pairwise correlation |r| >= 0.98
- diagnostic XGBoost gain importance

No 256D feature is automatically removed. The 256D/V23 Core remains intact
until real walk-forward data proves a pruning change is beneficial.

Important physical limitation: BBB does not see hidden card identities. Current
Core shoe information is primarily depth / estimated depletion. The program
cannot truthfully derive actual remaining rank composition or high-card ratio
from B/P/T outcomes alone, so no synthetic "known remaining deck" feature is
fabricated.

## Training

Example:

```bash
python -m pip install -r requirements-xgb.txt

python xgb_residual_bias.py train \
  --input bgs_xgb_walkforward_training.json \
  --output residual_bias_model.json
```

At least 12 chronologically ordered shoes are required by the default
8/2/2 three-way protocol.

The checked-in bundle remains `trained:false` until real data passes the
walk-forward promotion gate.

## Optional ensemble work

LightGBM / Ridge and road-regime-specific residual models are intentionally not
enabled in this revision. They should only be added after the strict
walk-forward baseline proves the single XGBoost residual is stable; otherwise
they add model-selection degrees of freedom before the evaluation protocol is
trustworthy.


## Operational training workflow

### 1. Collect browser exports

Use the BBB residual runtime normally. Each settled hand stores one row with:

- shoe_id
- created_at
- history_fingerprint
- actual_outcome (B/P/T)
- core_p_b
- core_raw_p_b
- core_logit
- fixed 7D features
- core_x_256 when available

Do not manually edit shoe_id or round_index.

### 2. Merge exports safely

```bash
python prepare_walkforward_training.py \
  exports/bgs_xgb_walkforward_training_*.json \
  --output bgs_xgb_walkforward_training.json
```

The merger removes exact duplicates, rejects conflicting duplicates, orders rows
chronologically by shoe, and reports whether the default 8/2/2 walk-forward
minimum of 12 shoes is available.

### 3. Validate before training

```bash
python xgb_residual_bias.py validate \
  --input bgs_xgb_walkforward_training.json
```

Validation must pass before training. It rejects missing shoe IDs, duplicate
prediction fingerprints, non-increasing round indices, insufficient shoes, and
overlapping OOS test windows.

### 4. Train

```bash
python xgb_residual_bias.py train \
  --input bgs_xgb_walkforward_training.json \
  --output residual_bias_model.json
```

A report is also written next to the model as
`residual_bias_model.report.json`.

By default the promotion gate requires the final OOS model to be no worse than
Core on all four primary metrics:

- non-tie accuracy
- LogLoss
- Brier
- equal-frequency ECE

Only after the gate passes is a trained model bundle written.

### 5. Confirm deployment bundle

Check:

```json
{
  "trained": true,
  "model_type": "xgb_core_margin_residual_v2"
}
```

Then deploy the resulting `residual_bias_model.json` with the BBB web files.
The runtime will automatically activate the residual layer after the model loads.
