# BBB V2.4 — Web-only independent final-score mode

## Production web workflow

- On authorized web access, present **only two point selects**, left = Player final total (0–9), right = Banker final total (0–9). No B/P/T history bootstrap and no Start click. Existing B/P/T/Start control IDs remain hidden in the DOM solely for compatibility with older scripts.
- Once both final totals are selected, derive the *completed hand's* B/P/T result. Construct a **new** `PhysicsForecastFirstEngine` (PF256, 384 legal hand proposals per particle, 160 MC physical EV samples per particle), then call `fresh.syncHistory([outcome],[{player,banker}])`.
- Each such model sees exactly **one hand**, not previous history, prior score rounds, nor a previous PF particle posterior. The resulting research signal describes the **next**, not the already settled, hand.
- An optional licensed audit `bgs_v24_independent_point_audit` records final-score pairs only for UI undo/reload and displays a count of independent analyses. This log is NEVER included in model history. Undo restores the preceding single pair using a new fresh engine.
- Do not use or alter old `bgs256d_frozen_6x15_forward_v18`, `bgs_v23_point_score_log` or `bgs_v23_point_entry_mode` keys. Prior data remains available for rollback; current model ignores it. End Analysis still invokes the existing scoped license purge.

## Strict model invariants

No changes to `quant256_online_runtime.js`, `physics_forecast_first_runtime.js`, `markov_residual_runtime.js`, Early-Shoe Guard, dynamic gates `0.005/0/0.035`, LinUCB, banker payout/commission, true-observed-card EV/BET qualification, PF N=256, legal likelihood L=384, physical MC M=160, ESS resampling `0.5N`, physics/road weights 5×/0.1×, login and license/device-bound authorization.

Markov first-order shadow is still present in the engine, but with only one B/P/T result there are **zero observed directional transitions**. No valid basis exists for a trained transition prediction or inferred win-rate advantage.

## Validation

- `test_v24_independent_points.js`: real engine, new particle object per scored pair, same score deterministic, distinct score never reuses previous state, 1 score / 1 history, actual EV and betting invariants.
- `test_v23_point_ui.js` now checks V2.4 web behavior: immediate scores, no Start, two fields required, undo, reload, tie, licensed audit, old history untouched.
- Existing full V2/V2.3 physics, numerical exact-score likelihood, unit, road, legacy regressions and authorization checks remain in CI. Old stage-based *UI* assertions are updated to the requested V2.4 web-only semantics.

## Limitations and rollback

Final total points do **not** reveal exact dealt ranks, and one settled score pair contains limited information about the remaining real shoe. Resampling from a fresh eight-deck prior each time discards real historical shoe depletion, so this mode **cannot be claimed more accurate** than a history-conditioned PF without independent chronological validation.

GitHub Pages deployment from `main`; Render and BGS LINE must not be changed. Roll back by reverting this PR merge to `f68da260d061b107de2bd4f8fa7e9765f475b8b0`. No storage migration, new key is separate and license-scoped.
