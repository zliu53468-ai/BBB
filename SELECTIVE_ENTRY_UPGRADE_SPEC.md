# BBB Selective Entry Upgrade Spec

## 1. Diagnosis

1. `B/P/T` history contains no observed card ranks, so the Physics 48D block is a conditional eight-deck expectation rather than a reconstructed residual shoe state. Its signal must be treated as uncertainty-aware context, not proof of next-hand direction.
2. The old low late-round EV threshold and low-noise soft band permit actions whose calibrated probability remains close to 0.50. These are the main source of borderline entries.
3. Calibration can improve global Brier while still being weak in the selected tails. The deployment objective must therefore rank hold-out `hit_rate_on_bets` and `realized_ev_per_bet` before overall accuracy.
4. A runtime policy that differs from the policy used for training/tuning creates selection mismatch: offline quality gates do not represent the browser's actual B/P/Skip behavior.
5. Shoe-level non-stationarity makes random-row validation invalid. Every model, calibration method, EMA profile, and entry policy must be selected only from chronological whole-shoe splits.

## 2. Implemented Phase 1

- Keep Frozen Core, Physics `213D -> 48D`, Final 57D, direct `P=0/B=1` XGBoost, Dynamic Clip, and B/P/Skip unchanged.
- Add deployable policy profiles:
  - `hard_ev`: current hard three-stage gate, used as baseline.
  - `strict_selective_entry_v1`: no middle/late soft band, no low-noise relief, noise threshold `0.68`, and up to `0.006` extra EV requirement in high noise.
- Expand the EV tuning search to early `0.010..0.045`, middle `0.005..0.030`, late `0.000..0.025`.
- Tune and evaluate with the exact policy config that is exported to browser JSON. The browser and Python now calculate the same noise penalty, relief, and soft band.
- A strict policy may increase Skip by at most `+5pp`, and only if hit-rate and EV/bet do not decline and Brier remains within the existing limit.

## 3. Feature / Model Guidance

- Do not remove or alter Frozen 256D Core inputs. For Phase 2, use grouped SHAP/permutation reports to measure the marginal value of `core`, original-6D, Physics winner, density, rank/suit, and noise groups. Use stronger XGBoost regularization or lower `colsample_bytree` for noisy groups; preserve 57D layout.
- Do not hand-delete Physics dimensions before a chronological ablation proves it helps hit-rate on bets. Winner distribution, density gap, and uncertainty are expected to be decision-relevant; rank/suit expectation blocks are most likely to be weak without real card observations.
- Keep direct classification as the production target. A residual or ensemble model may be evaluated offline only as a calibrated candidate feature/model; it must not replace the direct classifier without the same strict hold-out gates.
- Compare identity, Platt, and Isotonic calibration on a separate chronological calibration split. Select by Brier first, then verify selected-tail hit-rate/EV on the later EV-tuning shoes.

## 4. Optional Conservative Online Adaptation

Do not update the Frozen Core. If enabled in a later phase, keep only a per-shoe final-probability bias correction with all of the following: settle the previous prediction before update, update only directional B/P outcomes, shrink toward zero, cap absolute bias, reset on shoe change, and log every update. It must start disabled and be selected by chronological replay, never by in-shoe random shuffle.

## 5. Production Success Gate

Promotion requires a strict chronological whole-shoe hold-out report with:

- `hit_rate_on_bets` and `realized_ev_per_bet` non-decreasing versus baseline, with at least one strictly improving.
- `bounded_brier` no worse than baseline by more than `0.0015`.
- Skip-rate increase `<= +0.05` absolute.
- Bootstrap shoe-level 95% CI for bet hit-rate, EV/bet, EV/row, Brier, and Skip.
- Stage reports for `<=40`, `41-50`, `>50`, and dedicated `50-70`.
- Pattern reports for long-run, single-chop, and double-chop segments. Add these labels from B/P history only; do not infer unavailable card data.
- Report wager count and maximum consecutive losing bets. Do not promote a candidate with too few bets for a stable bootstrap interval.

## 6. Execution Order

1. **Phase 1 — decision policy:** completed here. Validate strict policy versus hard EV baseline on real chronological shoes; deploy only when gates pass.
2. **Phase 2 — 57D retraining:** add grouped feature-importance/ablation report, chronological calibration selection, and pattern-segment reporting. Retain 57D and direct classification.
3. **Phase 3 — conservative adaptation:** test capped per-shoe final-probability bias correction in replay. Keep it off by default until it clears the same hold-out gates.

## Engineer Prompt / Spec

```
Modify only final_probability_runtime.js and xgb_final_probability.py decision/training paths. Do not edit app256forward.js, app256continuation.js, hazardChoose(), the 213D->48D Physics topology, 57D feature order, or direct binary XGBoost target.

Implement named decision-policy profiles. Each profile must contain enabled, noise_threshold, max_noise_ev_penalty, middle_relief, late_relief, middle_soft_band, late_soft_band, and min_confidence. Python tuning must evaluate the same profile values that browser runtime receives in final_probability_model.json.

Select models, calibration, EMA, EV thresholds, and policy only with chronological whole-shoe splits: train -> tuning -> probability calibration -> EV/policy tuning -> strict hold-out. Never random-shuffle rows.

Promotion gate: hold-out bet hit-rate and EV/bet must not decline and one must improve; Brier regression <= 0.0015; Skip increase <= 0.05; bootstrap shoe-level CI required; report <=40, 41-50, >50, 50-70, long-run, single-chop, double-chop, wager count, and max losing streak. If any gate fails, retain the current production artifact.
```
