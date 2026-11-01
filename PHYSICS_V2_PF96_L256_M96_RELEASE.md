# BBB V2 PF96 physical precision candidate — 96 / 256 / 96

## Release scope

Starting from `main` commit `44a0eafba3072926ddfc6da90a0727ff8c3d2d27`, update **only** the existing V2 virtual-card / physical-EV sampling budgets:

| Component | Prior production | This release |
| --- | ---: | ---: |
| Virtual posterior particles | 96 | **96 (unchanged)** |
| Likelihood legal-hand proposals per particle / observed B/P/T | 128 | **256** |
| Physical EV simulations per particle / prediction | 64 | **96** |
| Total conditional proposals per new B/P/T | 12,288 | **24,576** |
| Total physical EV simulation draws (after a supported result) | 6,144 | **9,216** |
| Combined approximate draws per newly processed round | 18,432 | **33,792** |
| ESS systematic-resampling threshold | 0.5×N | **unchanged** |
| BASE / OVERDRIVE / CUSHION | 0.005 / 0 / 0.035 | **unchanged** |
| Physics / road feature scale | 5× / 0.1× | **unchanged** |
| LinUCB lambda / alpha | 0.82 / 0.15 | **unchanged** |

The runtime upper bound still clips particle count to **96**. 128 particles are **not** enabled by this change.

## Code, rollout and safety

- `quant256_online_runtime.js`: only the conditional proposal attempt count changes from 128 to **256**. Existing likelihood weighting, reservoir sampling, ESS diagnostics and resampling remain identical.
- `physics_forecast_first_runtime.js`: both the engine default and direct `summarizeHypotheses()` helper default change to **96** Monte Carlo draws per particle. Actual EV equations, direction, composition-based deterministic seeds, observed-card calculations, gate logic remain identical.
- `quant256_online_panel.js`: explicit browser configuration changes to **96** to avoid overriding the runtime's new default with the old value. No visual or storage changes.
- `auth-client.js` and `index.html`: script cache versions updated to `20261101-v2-pf96-l256-m96` for the online runtime, physics runtime, panel, and top-level loader. Authentication and card/bankroll UI unchanged.
- `test_v2_pf96_weighted.js`, `test_v2_mc_seed_isolation.js`: update runtime-default assertions only; explicit 64-sample helper test stays as an independent compatibility case.
- Add `test_v2_pf96_l256_m96.js`: end-to-end configuration and 96×96 sample-count guard, deterministic history replay, unchanged V2 controls/EV gate and cache integration. GitHub Actions `Quant256 Online Decay & Guard Checks` runs all relevant tests.

## Important limitations

256 likelihood proposals are still a **finite-sample approximation** to each particle's conditional result likelihood. Increasing the number reduces sampler noise under normal assumptions, but not model misspecification.

More conditional proposals and Monte Carlo simulations make an assumed latent virtual shoe distribution more precisely approximated; **B/P/T-only input does not reveal real cards consumed**, and neither improvement in real-money hit rate nor positive expected returns is established by this deployment. The repository's DATA_REQUIREMENTS.md records **zero verified complete real shoes**. Deploying this is a user-selected numerical-precision candidate, not an empirically validated best-hit-rate setting.

When 96 particles are supported, the nominal per-round sampling cost increases from 18,432 to 33,792 dealt hands (about +83%), without counting LinUCB overhead or full history replay. Browser/mobile responsiveness is device-dependent; especially long full-shoe replays may be slower.

Rollback reference: `44a0eafba3072926ddfc6da90a0727ff8c3d2d27`.
