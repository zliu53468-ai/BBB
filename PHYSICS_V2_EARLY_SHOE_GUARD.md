# Early-Shoe Bayesian-style Particle Guard (V2 PF96 / 256 / 96)

## Purpose / limits

This production-candidate change targets **only early 1–19 round B/P/T-only virtual-shoe forecasts**. It reduces false numerical confidence in a guessed card composition when little identifying information has been observed. It is a **deterministic Bayesian-style prior shrinkage heuristic**, not a mathematically calibrated Bayesian posterior or a demonstrated increase in real-money baccarat accuracy.

It preserves:
- 96 weighted hypothetical 8-deck particles / 256 conditional proposals per observed result / 96 forward legal deals per particle.
- Existing ESS<0.5*N systematic resampling and unbiased conditional matching reservoir sampling.
- Physics-first EV formula, banker commission, baseline 0.005 / overdrive 0 / cushion 0.035 dynamic V2 state machine; LinUCB and road physics scaling 5× / 0.1×.
- Existing licensing, no-scroll UI, B/P/T history, bankroll control, export and removed side-bet card.
- Any **real card observation** path remains untouched. The guard does **not** turn B/P/T-only into a qualified "BET".

## New diagnostics (no change to particle transition probabilities)

`quant256_online_runtime.js` records the likelihood approximation's binomial Monte Carlo relative standard error from the same 256 proposals already being sampled per particle. It tracks per-particle ancestor IDs across resampling and exposes the fraction of distinct ancestors. No additional dealing and no different random-number draws are added to the virtual particle update, so the latent particle positions and original post-round-20 physical distribution remain identical to the previous V2 for the same B/P/T sequence.

## Early virtual EV confidence shrinkage

`physics_forecast_first_runtime.js` adds a pure `earlyShoeGuard` function. Given n=1...19:

- Compute progress = n/20 and reliability from pre-resampling posterior ESS, active ESS, unique-ancestor diversity, and finite-proposal likelihood relative SE.
- `baseBlend = 0.12 + 0.88 * progress^1.25`
- `blend = baseBlend * (1 - 0.25 * (1-progress) * (1-quality))`
- For each result `k∈{B,P,T}`, shrink the unobserved MC mixture toward the exact fresh-shoe base: `P_guard(k) = P_base(k) + blend * (P_MC(k)-P_base(k))`.
- Recompute banker / player EV from the coherent shrunk B/P/T mixture. Raw MC standard errors are **not** shrunk: they remain conservative diagnostics.
- Apply **early-only** supplementary research gate `(1-progress)*min(0.01,0.20*mc_se_gap)`. Existing V2 phase state and gate `0.005 / 0 / 0.035` remain the same; this supplement controls early signal eligibility without claiming a new calibrated edge.
- Return diagnostic metadata `early_shoe_guard` with `blend`, `prior_weight`, ESS, ancestry, likelihood relative error and additional gate; no UI design change.

**From round 20 onward**, direct shrinkage and the supplementary gate are disabled entirely; raw PF96 simulated EV is returned unchanged. Caveat: V2's normal overdrive/cushion phase depends on prior **research signals**. Early signal changes can legitimately affect phase state at round 20; the original V2 phase transition code itself has not changed. Do **not** falsely promise every future signal will be identical to the no-guard historical run.

## Preflight and regression

- `test_v2_early_shoe_guard.js` covers early prior shrinkage and EV consistency, 20+ physical EV no-op, full observed-card bypass, valid normalized particles, unchanged 96/256/96 and V2 gate parameters, deterministic replay, and browser cache-loader references.
- Existing tests `test_v2_pf96_weighted.js`, `test_v2_pf96_l256_m96.js`, `test_physics_forecast_first.js`, `test_quant256_online.js`, `test_fixed_view_ui.js` and `test_v3_bankroll_ui.js` remain in the GitHub Action suite.
- Revert baseline: commit `56e5d7eb972a7093dd40f718fd1578c329bcb340`.

## Scientific limitation

B/P/T outcomes do not reveal real card ranks, and even with 96 particles / 256 proposals the hypothetical shoe remains uncertain. **No verified complete real-shoe holdout dataset is present**, so neither higher hit rate nor profitability can honestly be claimed. Early signals will generally be *less frequent*, which is intended to filter uncertain numerical deviations and does not by itself prove higher accuracy.
