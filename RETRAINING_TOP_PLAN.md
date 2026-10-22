# 新增：算牌優先的激進重訓候選

新 CLI：兩個 trainer 均可用 `--training-profile aggressive_cardcount`，必須輸出到獨立候選檔。未帶參數仍使用原 production 設定。完整參數、命令、獨立驗證與回滾見 [CARDCOUNT_MODERATE_PROFILE.md](CARDCOUNT_MODERATE_PROFILE.md)，共享參數為 `cardcount_profile.json`。

Physics 建議 6,000 靴、validation 0.20、calibration 0.10、augment 0.25；Winner 3.50、點數 1.45、張數 1.65、Rank 0.55。57D 仍是直接二元分類及 chronological whole-shoe，新增正則化候選 depth 2～4、500～750 trees、learning rate 0.015～0.025。階段樣本比重 1.10／1.15／1.25、低 uncertainty 最多 +18%、物理 actionability 最多 +8%、規律 +0%，類別平衡及正規化後硬限 0.35～1.50。

排序以 Brier、bet hit、EV/bet、Skip 為先；現有 holdout gate 保留，且必須另與目前 production＋PF＋Guard 完整回放比較才可換權重。此方案目前只新增程式設定，尚未用完整訓練資料執行重訓，沒有準確率或 EV 提升結論。以下保留原 production 計畫。

# BBB Top Retraining Plan (57D architecture locked)

This plan intentionally keeps the production architecture unchanged:

- Frozen Core: `app256forward.js` + `app256continuation.js` untouched.
- Physics MLP input/output stays 213D -> ReLU MLP -> 48D.
- Final feature vector stays 57D: Core P(B) + progress^3 + Original 6D + Physics 48D + physics_noise_score.
- XGBoost remains direct binary classification: P=0, B=1.
- Dynamic Clip + three-stage EV thresholds + optional causal EMA/noise gating remain in place.

## 1. Data

### Recommended volume

- Physics simulation: **5,000 shoes** default. Practical production range: 4,000-8,000.
- Final 57D real-history retraining: recommended **1,500-3,000 chronological shoes**. The CLI hard floor stays at 1,000 shoes / 12,000 directional rows so an existing dataset is not blocked unnecessarily.
- Prefer at least 300-500 shoes in the strict final hold-out.

### Chronological split

The 57D trainer uses whole shoes only and rejects interleaved shoes or backwards round order.

1. Oldest block: model train.
2. Latest tail inside train: XGBoost hyperparameter tuning.
3. Following chronological block: probability calibration, then a later sub-tail for EV/EMA tuning.
4. Newest 20%: strict hold-out. Never used for fit/calibration/threshold tuning.

Do not randomly shuffle shoes. Do not split one shoe across sets.

### Sample weighting

Weights are bounded and normalized. They emphasize:

- 50-70 rounds: 1.20 stage factor.
- 41-50 rounds: 1.05 stage factor.
- Lower physics uncertainty: up to +15%.
- Likely actionable rows based only on pre-label Core/EV information: up to +10%.
- Class rebalance P/B.

No target outcome is used to identify an "actionable" row, avoiding label leakage.

## 2. Physics MLP

The model structure is unchanged. Retraining changes only supervision emphasis and calibration.

Task loss emphasis:

- Winner probabilities: 2.50.
- Player/Banker point distributions: 1.10.
- 4/5/6 card-count: 1.25.
- Rank consumption: 0.80.
- Suit ratio: 0.65.
- Remaining composition / expected point-diff group: 0.55.

After fitting:

1. Affine-calibrate non-probability outputs.
2. Temperature-calibrate probability blocks.
3. Build the existing entropy/composition uncertainty proxy.
4. Fit Isotonic calibration from that proxy to empirical weighted 48D residual magnitude on a separate calibration shoe block.
5. Export the uncertainty mapping with both Python and browser bundles.

Recommended command:

```bash
python physics_feature_extractor.py train \
  --shoes 5000 \
  --validation-fraction 0.20 \
  --calibration-fraction 0.10 \
  --augment-ratio 0.25 \
  --bootstrap-samples 2000 \
  --output physics_multitask_model.joblib \
  --browser-output physics_multitask_model.json
```

## 3. XGBoost 57D

The classifier is still binary logistic and direct probability, never residual.

The search objective is now driven primarily by:

1. Brier score.
2. Hit-rate on placed bets.
3. Realized EV per placed bet.
4. Realized EV per row.
5. Skip change.

Overall accuracy is still reported but is no longer a hard selection gate.

Current practical search region represented by the deterministic candidates:

- `max_depth`: 1-3.
- `n_estimators`: 320-700.
- `learning_rate`: 0.012-0.025.
- `min_child_weight`: 10-24.
- `subsample`: 0.80-0.90.
- `colsample_bytree`: 0.70-0.90.
- `reg_alpha`: 0.25-1.00.
- `reg_lambda`: 12-24.

Probability calibration chooses Identity / Platt / Isotonic on a chronological calibration tail. A calibration is kept only when its probe-tail Brier improvement is meaningful.

Recommended command:

```bash
python xgb_final_probability.py train \
  --input bgs_final57_training.json \
  --physics-model physics_multitask_model.joblib \
  --min-samples 12000 \
  --min-shoes 1000 \
  --validation-fraction 0.20 \
  --calibration-fraction 0.15 \
  --tuning-fraction 0.15 \
  --ev-tuning-fraction 0.30 \
  --bootstrap-samples 2000 \
  --report-output retrain_report.json \
  --joblib-output final_probability_model.joblib \
  --output final_probability_model.json
```

## 4. Decision layer

No architecture change.

- High noise: probability Clip remains the primary restraint; EV penalty is capped and small.
- Low-noise middle/late rounds: the existing soft transition can slightly increase action willingness.
- EMA is selected only if hold-out-like tuning shows no hit-rate/EV regression, no Skip increase, and Brier increase stays within the smoothing guardrail.
- Three EV stages remain <=40 / 41-50 / >50.

## 5. Validation and deployment success

Every strict hold-out report includes:

- Overall accuracy.
- Hit-rate on placed bets.
- Realized EV per placed bet.
- Skip rate.
- Brier score.
- 95% shoe-bootstrap intervals.
- Stages: <=40, 41-50, >50.
- Dedicated 50-70 report: `late_50_70`.
- Before/after smoothing and decision-policy comparison.

A retrain is deployable only when all hard gates pass:

- Hit-rate on bets does not regress versus the legacy baseline.
- EV per bet does not regress versus the legacy baseline.
- At least one of hit-rate or EV per bet improves.
- Brier does not regress by more than 0.0015.
- Skip increase is <= +5 percentage points.
- EMA and soft decision policy independently pass their no-regression guardrails.

If a hard gate fails, the trainer exits before writing a deployable model bundle.

## CI smoke vs production promotion

GitHub Actions uses `--ci-smoke` only for the synthetic end-to-end verification path. This does **not** change any reported success flag:

- `retraining_success=false` remains false when hit-rate/EV/Brier/Skip gates are not met.
- The smoke bundle may still be generated so serialization and browser inference can be tested.
- A failed smoke retrain is never promoted or committed as a production model.
- Manual/real-data production retraining must omit `--ci-smoke`; failed success gates stop before deployment.

