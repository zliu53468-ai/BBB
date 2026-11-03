# BBB V2.6 — Quality-adaptive Early-Shoe Guard for independent final scores

Base `main`: `5d133bf0884d06c58d6eb6a6b95ec1276e124b87`.

## Issue

The V2.5 browser creates a **fresh** `PhysicsForecastFirstEngine` for each (Player final score, Banker final score) pair. Every model therefore has exactly one scored hand. The original Early-Shoe Guard shrinks every one-hand estimate toward fresh eight-deck `BASE` with the fixed `rounds/20=0.05` progress. This can leave ~86%–89% of the baseline prior in every output and make the slightly higher baseline banker EV dominate direction more often.

## Scoped change

- Add **optional** `independentFinalScoreMode: true` engine option. Only the authorized single-score web panel sets it. It activates **only** if exactly one scored hand was supplied and true observed card ranks are not used.
- Preserve old `earlyShoeGuard(..., {observed:false})` and default `PhysicsForecastFirstEngine()` behavior **exactly**, including legacy 1–19 round shrinkage, mature 20+ no-op and actual-card bypass.
- When opted in, reuse the same posterior and physical Monte Carlo, but determine a bounded shrinkage coefficient from:
  - PF update ESS, active mixture ESS, ancestor diversity and likelihood-relative error (`quality`, unchanged formula)
  - MC gap standard error (`mcNoise`) and posterior between-particle gap dispersion (`posteriorSpread`)
  - `mcReliability=1/(1+mcNoise/0.04)`, `posteriorReliability=1/(1+posteriorSpread/0.12)`
  - `reliability=.65*mcReliability+.35*posteriorReliability`
  - **`blend=clamp(.28+.60*quality*reliability, .28, .82)`** instead of the constant one-hand blend ~.11–.14.
  - `prob = BASE + blend*(rawParticleProbability-BASE)`; re-evaluate the same 5% banker-commission EV. Keep at least 18% baseline shrinkage, with more shrinkage when evidence quality is poor.
  - The research-gate supplemental uncertainty term is `(1-blend)*min(.01,.20*mcNoise)` only in this web mode; all original dynamic gates remain.
- No manual weighting toward Player, forced 50:50 direction frequencies, fake observed ranks, fixed output direction, or independent Markov evidence.

## Unchanged

PF256, 384 Rao–Blackwell conditional proposals per particle, 160 Monte Carlo EV deals per particle, ESS resampling cutoff 0.5N, legal drawing rules, physical EV and commissions, score-only inputs, per-hand fresh PF, desktop Enter/mobile Analyze, 5× physics/0.1× road, Markov diagnostic/shadow (production applied weight 0), LinUCB, observed-card BET eligibility, license/device binding, existing history keys, Render/BGS LINE.

## Validation and limitations

`test_v26_early_shoe_independent.js` compares default vs opted-in one-score behavior, synthetic high/low evidence quality, deterministic replay, B/P/T default and real-card bypass, and preserved BET rules.

`test_v26_100_scores.js` evaluates **all 100** combinations of final Player/Banker totals. It prints both versions' B/P direction counts using each scenario's *identical* simulated posterior and compares prior-blend/ESS diagnostics, without an artificial balanced-direction passing criterion. Legacy and license/mobile regression suites must also pass.

**This change improves internal consistency of independent-mode smoothing; it does not establish better real-world hit rate or positive expected value.** A single final-score pair does not reveal the actual card ranks or complete remaining shoe. Validate chronological unseen full-shoe data before making any betting performance claim.

Rollback: revert this PR to commit `5d133bf0884d06c58d6eb6a6b95ec1276e124b87`; no data migration.
