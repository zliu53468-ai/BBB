# V2 Physics Monte Carlo Seed Isolation — 2026-11-01

Baseline commit: `1312554d45c65577878e3285b8a190ba3b2d49ac`. The prediction architecture is still **Physics Forecast First V2 Dynamic Gate**, based on `e323a243`.

## Problem and exact change

- **Before**: `physics_forecast_first_runtime.js` hashed the previous B/P/T outcomes with `hashHistory(history)` to choose the Monte Carlo random seed for `summarizeHypotheses`. The observed-card path also seeded simulation from `hashHistory(this.history)`. Thus the *same* remaining point-count shoe could produce a different EV estimate when merely given a different historical label string.
- **After**: `hashComposition(shoes)` chooses the seed from the current remaining 0–9 point counts (and stable particle order), not the historical label text. The virtual-particle update and all other prediction logic are **unchanged**. `summarizeHypotheses(physics, history, options)` retains its public signature; `history` is explicitly unused in the Monte Carlo computation.
- `quant256_online_runtime.js` remains byte-for-byte as in e323a243, including its **outcome-conditioned** particle depletion. This update **does not** eliminate the separate result → virtual-depletion pathway. It only removes an additional, statistically arbitrary result → PRNG-seed pathway.
- No changes to physical 5×, road 0.1×, `BASE_EV_GATE=.005`, `OVERDRIVE_GATE=0`, `ONE_HAND_CUSHION=.035`, `simulationsPerParticle=64`, `lambda=.82`, or `alpha=.15`. No login, UI controls, bankroll behavior, or historical storage changes.

## Synthetic prequential evaluation

The evaluator generated sequential legal baccarat hands from **fresh eight-deck shoes**, recording the generated outcome and removing *actual generated cards* from the synthetic true shoe before the next round. Before each result, each engine predicted using only past B/P/T outcomes, with the same history and original gate state machine.

A fixed-seed candidate comparison tested 64 vs 256 simulations and SE-ratio filters; these **did not consistently outperform baseline**. Those candidates were discarded rather than automatically deployed.

For the composition-seed-only change, a larger second comparison tested **92 independent synthetic shoes, 38 hands per shoe (3,496 predictions)**, partitioned into two different shoe groups:

| Group | V2 label-based seed: signal hit rate | Composition-based seed: signal hit rate |
|---|---:|---:|
| First 46 shoes | 51.36% | 51.98% |
| Other 46 shoes (separate) | 49.50% | 51.62% |

Hit rate excludes ties; only the model's pre-outcome `research_signal.available` rounds count. The composition-seeded change improved both *these* synthetic groups. The differences are relatively small, not a statistically established real-money improvement. Data Requirements states that **zero complete real shoes** have been collected; none of the synthetic figures is a verified real-world win rate. Do not claim an actual profitable edge based on these numbers.

## Regression

```bash
node test_physics_forecast_first.js
node test_quant256_online.js
node test_v2_mc_seed_isolation.js
node test_v3_bankroll_ui.js
```

New `test_v2_mc_seed_isolation.js` verifies that identical virtual particles yield identical `physical.ev_banker`, `physical.ev_player`, and the rest of the MC report for completely different history strings. It also verifies all original V2 gate constants and original research-vs-actual card distinction. Existing tests continue to run through the Quant256 GitHub Actions workflow.

The deployed loader gets a distinct cache key on the physical runtime and top-level authorization loader so updated clients do not keep using the old code.

## Remaining limitations

- Outcome-conditioned virtual particle depletion remains in the V2 baseline. Since only B/P/T is observed, virtual ranks are **not actual known ranks**, and virtual EV is not validated positive betting EV.
- Stable seeds reduce incidental Monte Carlo variation. They cannot create additional information or assure future win-rate improvement.
- Retain the original V2 baseline commit for reverting if independent real full-shoe forward testing contradicts the synthetic comparison.
