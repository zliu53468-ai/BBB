# V2 Dynamic Shoe Distribution Guard — test release

Base/rollback commit: `c020a8e519140e8aaa7b9e69ebf8cd4fbc03bb71`.

## Scope
The previous live predictor had good user-reported results and its physics model remains the **primary B/P/T direction source**. This release adds a **causal descriptive change-point monitor**, not a new predictive law. Input B/P/T-only cannot reveal exact depleted card ranks, nor guarantee that historical streaks continue or reverse.

- **Unchanged** 96 particles, 256 legal likelihood proposals per particle, and 96 EV MC deals per particle; 9,216 shadow EV deals per supported round.
- **Unchanged** ESS at 0.5N, particle weight update, conditioned legal dealing, Early-Shoe Guard, base/overdrive/cushion gates `0.005/0/0.035`, physics 5×, road 0.1×, genuine observed-card EV safety.
- **Unchanged** mobile stable viewport, login, history keys, bankroll and stake calculations, removed static side-bet panel.
- **New** `shoe_distribution_guard.js`, loaded before `physics_forecast_first_runtime.js`. An ordered Bernoulli Beta(1,1) marginal Bayes factor compares the most recent and immediately preceding windows of **8/16/24 non-tie outcomes**. Ties do not count as fake Banker/Player evidence. A conservative 10% prior change probability and an 80% posterior threshold (plus at least 20 percentage points observed difference) detect only strong historical shifts, not any predicted forthcoming change.
- When the historical shift is very strong and **only for B/P/T virtual-shoe research**, the module adds a modest **0 to 0.004 absolute EV-gap** penalty to the effective *research signal* gate. It **never** modifies raw PF particle distributions, physical EV, physical direction, actual observed-card BET safety, or the V2 gate phase state machine itself.
- Monitoring a change could alter which research signals are accepted and thereby indirectly influence subsequent overdrive/cushion phase transitions. This is an intentionally conservative experimental gate and could lower the number of signals. It is **not** validated to improve winning rate.
- **No expensive adaptive 128-draw simulation** in this release. We intentionally retain 96/256/96 to prevent more phone lag and isolate the change-point component.

## Verification
Run `node test_v2_dynamic_distribution_guard.js` plus the existing physics, PF96, early-shoe, bankroll, mobile and authorization regressions. The new test covers balanced vs changed early/mid/late sequences, tie neutrality, bypass of real card observations, no forced direction flip, unchanged EV and replay, and cache loader ordering.

New UI line is a descriptive monitor flag ("分布監測：近期比例明顯改變" / "資料不足" / "未見強烈變化"), not a forecast for a regime reversal.

## Rollback
To restore the exact prior production tree, create a revert PR for the merge commit of this release or use `c020a8e519140e8aaa7b9e69ebf8cd4fbc03bb71` as the original baseline. Prefer a revert commit instead of force-resetting public `main`.

## Research caveat
These parameters are engineering heuristics, not empirically optimized against independent, held-out real shoes. Historical variation can be ordinary randomness. **Increased real baccarat hit rate has not been proven**; test on unseen shoe-level data with equal signal coverage and mobile latency checks before interpreting results.
