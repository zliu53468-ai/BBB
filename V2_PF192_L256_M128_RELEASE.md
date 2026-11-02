# BBB V2 PF192 / L256 / M128 — User-requested production test

## Baselines and rollback
- Immediate prior live main: `956d49af73e0f61be7aa078a63eaacbc8dd2b8cf` (V2 dynamic shoe distribution monitor, PF96/256/96)
- Older known-stable fallback: `c020a8e519140e8aaa7b9e69ebf8cd4fbc03bb71` (PF96/256/96 without the subsequent distribution monitor)
- Restore by reverting this merge commit in a **new PR**, not rewriting public main's history, or use a rollback branch based on the chosen SHA.

## Changes: only sampling precision and corresponding defaults/UI cache and tests

| Sampling or safety setting | Prior | Trial |
| --- | --- | --- |
| Virtual 8-deck particle hypotheses | 96 | **192** |
| Particle count hard cap | 96 | **192** |
| Outcome-conditional legal hand proposals per particle | 256 | **256** |
| Hypothetical physical EV Monte Carlo deals per particle | 96 | **128** |
| All-particle likelihood proposals for one observation | 24,576 | **49,152** |
| Forward physical EV MC deals after each observed result | 9,216 | **24,576** |
| Nominal combined deal samples for the next round | 33,792 | **73,728 (~2.18×)** |
| ESS systematic resampling cutoff | 0.5N | **unchanged (96 with N=192)** |
| V2 base/overdrive/cushion research gate | 0.005 / 0 / 0.035 | unchanged |
| LinUCB lambda/alpha | 0.82 / 0.15 | unchanged |
| Physics vs road scaling | 5× / 0.1× | unchanged |
| Early-Shoe Guard and 8/16/24 window distribution monitor | enabled | unchanged |
| Observed-card-only real EV gate & synthetic B/P/T-only no-BET rule | enabled | unchanged |

- `quant256_online_runtime.js` updates **both** defaults (BlindPhysicsEstimator and OnlineLinUCB256) and the hard particle cap; no change to the legal sampleHand rules, 256 match attempts, weight updates, ESS, stochastic seeds, historical features, latent-fidelity disclaimers or feature calculations.
- `physics_forecast_first_runtime.js` updates the engine default and direct-summary helper default of 96 to 128 EV samples; no other physical EV formulas or gates touched.
- `quant256_online_panel.js` explicitly uses `simulationsPerParticle:128` and `quantOptions.particles:192` to avoid leaving the browser at 96.
- `auth-client.js` and `index.html` only bump cache keys of the changed modules and dynamic panel; the unchanged shoe-distribution monitor and mobile CSS/viewport versions are retained.
- Regression test expectations are updated to reflect the new chosen release baseline without rewriting any math, and `test_v2_pf192_l256_m128.js` adds strict actual N/cap/MC/check tests.
- The existing old test *filename* `test_v2_pf96_l256_m96.js` is kept for compatibility with existing jobs, but assertions now check the new production trial.
- GitHub Action Quant256 CI timeout increased to 12 minutes to account for the doubled compute load.

## Cautions and validation
192/256/128 improves the numerical approximation of *hypothesized* latent decks, not knowledge of actual cards. It **does not** establish higher real-world hit rates. While observed-card EV is more reliable if actual ranks are supplied, B/P/T-only virtual simulations still cannot identify the true residual shoe. The user is explicitly trialing this higher-cost configuration with the ability to roll back.

The nominal draw count is **not** a latency benchmark. The existing browser computes this in JavaScript on the member's phone; the +118% nominal draw cost could make mobile performance substantially worse. Need end-user performance tests on iPhone/Android (especially a full 50–70-hand replay) before widening rollout.

No changes to login, storage, bankroll, mobile responsive layout, or side-bet removal.
