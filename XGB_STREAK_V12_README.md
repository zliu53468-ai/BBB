# BBB XGBoost Residual V1.2 — Streak Quality

V1.2 does not add a second model. The prediction path remains:

```text
256D / V23 Frozen Core
  -> existing 7 residual features
  -> XGBRegressor residual
  -> delta scale
  -> final P(B)
  -> switch-margin hysteresis
  -> B / P only
```

There is no PASS state and no LightGBM/ensemble.

## Why V1.2 exists

V1.1 validates single-hand accuracy, Brier score and XGB flip gain. V1.2 additionally validates the rhythm of a 50-hand table so a policy is not accepted merely because its aggregate accuracy looks acceptable while wins are fragmented by repeated near-50% direction changes.

The switch margin is selected by grouped out-of-shoe OOF validation. It is not hard-coded. Default candidates are:

```text
0.00%, 0.25%, 0.50%, 0.75%, 1.00%, 1.25%, 1.50%, 2.00%
```

Example: if the previous output was B and the selected margin is 1%, the policy does not switch to P until final P(B) falls below 49%. If the previous output was P, it does not switch to B until final P(B) rises above 51%.

## 50-hand validation metrics

Each OOF shoe is evaluated in 50-hand blocks. The trainer reports:

- mean and median table accuracy
- share of tables at or above 52%, 54% and 56%
- average win-run length
- average longest win streak
- 3-win and 4-win table rates
- average longest loss streak
- P95 longest loss streak
- 3-loss table rate
- global maximum win/loss streak

## Safety gates

A candidate policy is not accepted when it regresses any default hard gate:

- direction accuracy must not fall below the frozen core
- Brier score must not be worse than the frozen core
- net XGB flip gain must be non-negative
- P95 longest loss streak must not worsen
- mean longest loss streak must not worsen

After these gates pass, selection favors higher >=52% and >=54% table rates, longer useful win runs and lower loss-run risk.

## Train

```bash
python -m pip install -r requirements-xgb.txt
python xgb_residual_streak_v12.py \
  --input bgs_xgb_residual_training.json \
  --output residual_bias_model.json \
  --min-samples 500 \
  --min-shoes 20 \
  --min-tables 20 \
  --folds 5 \
  --table-size 50
```

`--force` is diagnostic only. A rejected model should not be deployed just because it can be exported.

## Current placeholder

The checked-in `residual_bias_model.json` intentionally remains `trained:false` until real labeled production rows pass V1.2 validation. Therefore adding V1.2 code alone does not invent a higher win rate or silently alter the frozen core before a validated XGB model exists.
