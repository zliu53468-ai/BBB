# PF256 / L384 / M160 sampling-only production trial

Prior live commit: `9c0b4772a2e6789c2c598932f3389b2edb267638` (PF192/L256/M128). Earlier rollback: `956d49af73e0f61be7aa078a63eaacbc8dd2b8cf`.

## Exact sampling changes

| Parameter | Before | After |
| --- | ---: | ---: |
| BlindPhysicsEstimator default particles | 192 | **256** |
| BlindPhysicsEstimator hard particle cap | 192 | **256** |
| OnlineLinUCB256 particles default argument (not its formula) | 192 | **256** |
| Conditional legal-hand likelihood proposals per observed result / particle | 256 | **384** |
| PhysicsForecastFirstEngine default EV samples per particle | 128 | **160** |
| summarizeHypotheses default EV samples per particle | 128 | **160** |
| Browser `quantOptions.particles` | 192 | **256** |
| Browser `simulationsPerParticle` | 128 | **160** |
| Conditional likelihood proposals / outcome | 49,152 | **98,304** |
| Per-round physical EV Monte Carlo deals | 24,576 | **40,960** |
| Nominal combined proposals/deals | 73,728 | **139,264 (1.889× prior)** |

## Explicit non-changes
- 8-deck legal `sampleHand` and banker third-card rules; 10-point count space; RNG/seeding mechanics.
- ESS cutoff `lastESS < size*0.5`, systematic resampling, weights, posterior likelihood calculation, conditional reservoir sample.
- All V2 physical EV formulas, `decisionFromEV` and observed-card-only BET eligibility.
- Research gate values base `0.005`, overdrive `0`, cushion `0.035`; phase transition logic.
- LinUCB `lambda=.82`, `alpha=.15`, learning features, weights, direction.
- Early-Shoe Guard; causal 8/16/24 distribution monitor, gate and no forced direction; physics 5× and road .1×.
- Authorization, storage key, bankroll, responsive iPhone viewport and no-side-bet UI.

## Verification
CI includes `test_v2_pf256_l384_m160.js`, updates all older PF default assertions, and preserves integration/physical rules, seed isolation, early-guard, distribution, storage and iPhone regressions. The older test filenames remain for workflow compatibility, but their expected values reflect this production sampling version.

## User test, latency, rollback
This is a **user-requested high-compute trial**. Simulating additional hypothetical 8-deck states can improve numerical stability but does not reveal true card values from B/P/T-only history or prove any increase in predictive success. Actual mobile responsiveness must be verified. To roll back, revert the release merge commit on a fresh branch + PR; prefer a history-preserving revert, not force resetting `main`. Use `9c0b4772` for immediate prior 192/256/128, or `c020a8e5` for the earlier known baseline.
