# Volume-Guard Selective Entry v1

## 1. Deployment objective

The deployment target is a **three-metric constrained objective**, evaluated only on a chronological shoe-level hold-out:

| Metric | Required change vs. hard-EV baseline |
| --- | --- |
| Skip Rate | increase by at least 2 percentage points for the strict profile, and never more than the existing 5-point ceiling |
| Hit-Rate on placed bets | non-decreasing; production acceptance requires an increase |
| Absolute Correct Bets | at least 95% of baseline |
| Realized EV per placed bet | non-decreasing |

Overall accuracy is report-only. It is not an optimization or deployment gate.

## 2. Why a fixed minEV increase is insufficient

A fixed `minEV` increase removes marginal and high-quality rows indiscriminately. It can improve percentage hit-rate merely by reducing denominator size, while lowering the absolute number of correct bets. It also ignores that early shoes and noisy physics estimates need a wider no-trade region than clean late shoes. This policy therefore requires both an uncertainty band and a rolling volume guard.

## 3. Browser-deployable decision rule

`final_p_b` remains the calibrated, clipped and causal-smoothed XGBoost probability. No 57D feature, Core module, Physics model, or model architecture changes.

```text
stage = early if round <= 40 else middle if round <= 50 else late
distance = abs(final_p_b - 0.50)

baseBand = {early: .028, middle: .020, late: .014}[stage]
band = clamp(baseBand + .018 * max(0, physics_noise_score - .50)
             - (.003 if stage == late and physics_noise_score <= .78 else 0),
             .010, .050)

minEV = existing strict stage minEV + existing noise penalty
activationEV = minEV

if volumeGuardActive:
    band = max(.010, band - .006)
    activationEV = max(0, activationEV - .004)

direction = B/P only if directionEV > activationEV AND distance >= band
otherwise direction = Skip

entryTier = strong if distance >= band + strongMargin[stage]
            weak if B/P but not strong
            skip otherwise
stakeMultiplier = strong: 1.0, weak: 0.5, skip: 0.0
```

The current page continues to show one B/P/Skip direction. `weak` is persisted as metadata for UI or stake handling; it does not alter the frozen Core or game-history buttons.

## 4. Volume Guard

Use only preceding prediction rows in the same shoe. Do not consume settled outcomes, so the guard cannot create label leakage.

```text
window = 16 previous same-shoe prediction rows
minHistory = 8
targetActionRate = {early: .30, middle: .36, late: .42}
expectedCorrectFloor = .95

expectedCorrect = sum(max(pB, 1-pB) for actual entries in window)
baselineExpectedCorrect = sum(max(pB, 1-pB) for entries that pass EV without confidence band)

activate guard when:
  actualActionRate < targetActionRate[stage]
  OR expectedCorrect < .95 * baselineExpectedCorrect
```

The guard relaxes only the next eligible decision by `band_relief=.006` and `ev_relief=.004`. It does not force a bet. Each next row re-evaluates the rolling window, which makes it a soft-to-hard transition rather than a fixed quota.

## 5. Evaluation and production gates

Every chronological shoe-level report must show: `skip_rate`, `hit_rate_on_bets`, `absolute_correct_bets`, and `realized_ev_per_bet`, plus Brier and `50–70` late-stage results. Include early (`<=40`), middle (`41–50`), late (`>50`), and `50–70` segmentation. The existing stage report remains the home for this data.

Block deployment unless all of these hold on hold-out:

1. strict-profile Skip increases by at least 2 pp and no more than 5 pp;
2. Absolute Correct Bets are at least 95% of legacy baseline;
3. hit-rate and EV per placed bet do not regress (and hit-rate improves for production acceptance);
4. Brier does not exceed the existing model-regression limit;
5. calibration, chronological split, bootstrap CI, and existing smoothing gates all pass.

## 6. Engineer change list

### `final_probability_runtime.js`

1. Read `decision_policy.confidence_band` and calculate the stage/noise band above.
2. Add `volumeGuardState()` using only `readRows()` from the current shoe and preceding `round_index` values.
3. Gate B/P by both EV and `abs(final_p_b - .5) >= effectiveConfidenceBand`.
4. Persist `confidence_band`, `effective_confidence_band`, `effective_activation_ev`, `volume_guard_active`, `entry_tier`, and `stake_multiplier`; snapshot schema is now 7.

### `xgb_final_probability.py`

1. Keep 57D construction, classifier, calibration, chronological splits, and physics model unchanged.
2. Mirror the confidence-band rule in `decision_returns()` and pass shoe IDs during tuning/evaluation.
3. Apply Volume Guard sequentially per shoe from prior predictions only.
4. Add `absolute_correct_bets` to metrics and make every tuning/deployment comparison require `>= 95%` of baseline.
5. Require `min_skip_increase=.02` for `strict_selective_entry`; retain the existing hard maximum Skip increase.

## 7. Initial configuration

```json
{
  "min_skip_increase": 0.02,
  "confidence_band": {
    "early": 0.028,
    "middle": 0.020,
    "late": 0.014,
    "noise_reference": 0.50,
    "noise_gain": 0.018,
    "clean_late_relief": 0.003,
    "minimum": 0.010,
    "maximum": 0.050,
    "strong_margin": {"early": 0.007, "middle": 0.005, "late": 0.003}
  },
  "volume_guard": {
    "enabled": true,
    "window": 16,
    "min_history": 8,
    "target_action_rate": {"early": 0.30, "middle": 0.36, "late": 0.42},
    "expected_correct_floor": 0.95,
    "band_relief": 0.006,
    "ev_relief": 0.004
  }
}
```
