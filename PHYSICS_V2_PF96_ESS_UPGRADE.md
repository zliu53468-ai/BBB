# V2 Physics Forecast First: 96-particle likelihood-weighted ESS upgrade

**Base commit** `c18d0a925101a305b52e2fd209646bd6c2c6339b`. The active model remains **BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE**. This enhancement ONLY modifies the virtual card-shoe particle filter and physical EV posterior mixture, plus regression tests, cache keys and CI.

## What changed

### `quant256_online_runtime.js` — `BlindPhysicsEstimator`
- Default particle count **24 → 96**, retaining supported range 8–96; `OnlineLinUCB256` passes 96 by default.
- Keep baccarat third-card draw rules and B/P/T-conditional updates.
- For each observed B/P/T, try **128 valid hands per particle** and count matches. Sample a legal matching transition uniformly by reservoir sampling.
- Multiply each particle's prior weight by its estimated `P(observed B/P/T | particle)` = matches / 128. Normalize the posterior weights. This corrects the previous unconditional **equal-weight** treatment of particles that could have very different outcome likelihoods.
- Compute effective sample size (ESS) = `1 / Σ weight²`. If **ESS < 0.5 × particle count**, perform deterministic systematic resampling of **whole legal card-count vectors**, then give children uniform weights. No arbitrary additions/removals of A–K ranks.
- If no particle can explain the outcome, preserve the shoe and flag `posteriorUnavailable`; downstream physical forecast returns `simulation_exhausted` instead of inventing a supported EV.
- Weighted posterior means/variance in `snapshot()`. Existing field names and browser/public APIs remain usable; added `particle_ess`, `particle_active_ess`, `resampled`, `posterior_unavailable`.

### `physics_forecast_first_runtime.js` — `summarizeHypotheses`
- For each surviving hypothetical shoe, simulate the unchanged **64 hands per particle**. Compute `P(B/P/T)`, banker/player EV and remaining-card estimates by **posterior particle weights**, rather than treating each particle as equally likely.
- Expose within-particle Monte Carlo standard error (`mc_se_gap`) and between-hypothesis deviation (`posterior_sd_gap`) separately. `se_gap` becomes their conservative quadrature combination. These are simulated uncertainties, **not calibrated confidence levels or proof of edge**.
- Preserve composition-derived RNG seeding from `c18d0a92`; keep V2 direction selection, `BASE_EV_GATE=.005`, `OVERDRIVE_GATE=0`, `ONE_HAND_CUSHION=.035`, physics 5× and road 0.1× exactly.

### Explicitly not changed

- No UI elements, bankroll inputs, authorized member login, Cloudflare device binding, local history keys, stake percentages, LinUCB `lambda=.82`/`alpha=.15`, road features, XGB or other models.
- `quant256_online_panel.js` unchanged. `auth-client.js` and `index.html` only update cache-busting strings, to force new versions of the two runtime scripts without stale JS.
- The conditional latent-card update is **hypothetical**: B/P/T-only inputs do not disclose which physical ranks were used.

## Regression and verification

```bash
node test_v2_pf96_weighted.js
node test_quant256_online.js
node test_physics_forecast_first.js
node test_v2_mc_seed_isolation.js
node test_v3_bankroll_ui.js
```

The new regression covers:
1. **96 × 64 = 6,144** forward simulated baccarat hands when the 96 particles are viable.
2. Identical output on a fixed-seed full replay; posterior weights remain normalized and each particle conserves legal cards.
3. Weighted-EV posterior differs from equal-weight EV for deliberately different shoe compositions (test makes sure weights actually affect EV).
4. Forced ESS collapse triggers resampling, new arrays do not alias, counts stay nonnegative.
5. No-support condition forces simulation exhaustion; V2 threshold/physics/road/direction invariants unchanged.

Additional early **synthetic** sequential replay (independent fresh 8-deck true shoes; pre-outcome forecasts; 16 shoes × 30 rounds = 480 decisions):
- V2 original 24-particle signals: **363**, non-tie hit rate **47.46%**.
- PF96 weighted signals: **310**, non-tie hit rate **49.65%**.
- This is a small **synthetic smoke test**, NOT significant evidence of real-world improvement, and the two versions made a different number of signals. Neither figure should be advertised as a real hit rate. The repository currently has **zero verified complete real shoes** available for holdout testing.

## Caveats

- Particle weights are based on finite-sample approximations `matches/128`, not analytic baccarat likelihoods. Fewer than 128 matches can introduce likelihood estimation error, especially for rare tie outcomes.
- Systematic resampling can duplicate latent ancestors and reduce true genealogical diversity; new legal draws on later hands provide diversity, but measured ESS does not fully capture ancestry collapse. Consider separate diversity diagnostics *before claiming further precision*.
- With only B/P/T, no actual A–K ranks or real card-count posterior can be verified. More particles improve approximation of a model assumption; they do **not** guarantee any improvement to real baccarat hit rate, profit, or avoidance of loss streaks.
- The original V2 binary/ternary result-conditioned virtual sampling remains by user intent; it is not actual observed card depletion.
