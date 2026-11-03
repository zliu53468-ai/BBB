# BBB V2.1 — conditional likelihood + resample-move (prediction quality)

Base: `BBB/main@7f49802f6f514eb7bf44aa3a61bdf4b1ffcc9920`.

## Scope

**Unchanged:** 256 particle default/hard cap, 384 proposals per particle, 160 Monte Carlo EV samples per particle, ESS resampling trigger 0.5N, observed-card EV and payout rules, physics weight 5.0×, road weight 0.1×, dynamic gate (0.005 / 0 / 0.035), Early-Shoe Guard, distribution change-point guard, LinUCB policy, original web markup, user history keys, login/licensing, bankroll panel and model JSONs.

**Upgrade 1: Rao–Blackwell conditional likelihood.** Sample exactly 384 independent first-four-card draws per latent particle. Given those four cards, enumerate all legally possible third-card draws (including banker third-card rules) to compute a conditional P(B/P/T). The weighted average of those 384 values is the likelihood estimate for that particle. A weighted reservoir draws one legal conditional transition. This eliminates *third-card Monte Carlo noise conditional on the first four*, but does not solve the full eight-deck likelihood exactly or reveal true residual ranks.

**Upgrade 2: conditional resample-move.** At ESS below 0.5N, keep systematic ancestor selection, but for duplicate ancestor selections propose a fresh legal conditional hand from the **pre-observation** parent. Accept each proposal with probability P(observed outcome | first four), preserving the correct conditional distribution; cap at 32 attempts and safely retain a legal original descendant if no move is accepted. The count vectors and lineage remain physically valid. Rejuvenation uses additional conditional proposals only on ESS-triggered duplicates; the baseline 384/160 parameters stay unchanged.

**Upgrade 3: uncertainty diagnostics.** Add MC gap interval conditional on the current simulated particle mixture, between-hypothesis EV dispersion, particle diversity, likelihood relative SE and rejuvenation counters to the report. These are diagnostic only, NOT confidence of future win or validated positive expected return. **No change to displayed gates or BET eligibility.**

## Verification

- `node test_v2_1_rb_rejuvenation.js`: compare likelihood estimates to exact independent six-card combinatorial enumeration; test likelihood variance reduction vs naive binary sampling; ensure lawful card draws, resample-move restoration, and deterministic B/P/T history replay.
- Existing V2/PF256, Early-Shoe, physics, window/browser cache, bankroll, and license tests unchanged or updated only for expected source formulas/cache keys.
- GitHub Pages must complete **after** main merge; source tests do not prove that any long-run hit-rate/ROI improved.
- No change to BGS LINE repository. If BGS is to match BBB V2.1, its separate pinned source must later be intentionally updated and tested.

## Release risk and rollback

The Rao–Blackwell third-card integration performs additional arithmetic per proposal, so performance on both mobile browsers and 20–60-round initial history must be checked before merging. If performance or regression results fail, leave V2.1 in the feature branch; do not deploy to main.

Rollback via GitHub revert of the feature PR merge commit, without touching user history, licenses, persistent state or authentication.
