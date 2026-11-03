# BBB V2.5 — Web-only manual final-point input and always-show-EV-direction

## Scope
Preserve the current **independent per-hand** PF256, L384 legal-card proposals, M160 physical-EV simulation, Markov shadow, LinUCB, 5× physics / 0.1× road settings, Early-Shoe Guard, existing 0.005/0/0.035 research gates, observed-card BET rules, authorization/device binding, and original external services. This PR changes **only browser input and presentation logic**, not any prediction engine runtime.

## User input
- Compact **typed** final totals, left = Player, right = Banker, integer 0–9 inclusive.
- A visible **開始分析** button explicitly submits after BOTH values are valid; selecting/typing/tabbing/change events never start calculations.
- Desktop: Enter in either field is an alternative submission; mobile touch/coarse-pointer input requires the visible button. Empty, spaces, decimals, signs, exponent notation or values outside 0–9 are rejected without clearing the fields.
- After successful submission, both input fields clear so the next completed hand can be entered.
- No historical bootstrap, no accumulating particles, no extra vertically expanding page and **no analysis count** in the header or prediction metrics.
- The V2.4 licensed score audit is kept only for undo and navigation. It NEVER enters the model. Initial login/reload does not auto-run a stale score forecast.

## No-SKIP display semantics
- The prominent **next-hand direction** is **B** if `report.ev_banker >= report.ev_player`, else **P**, using the existing simulated physical EV output. Exact equality resolves to B, consistent with the existing physical direction comparator.
- This displays a comparison even if `report.research_signal.available === false`, i.e., the *visual research direction no longer says 觀望*. It is **NOT** a change to `report.research_signal`, `report.action`, `BET` eligibility, EV payout formulas, PF particle posterior, Markov mixing, or guard thresholds.
- For invalid or exhausted physics results, show an error rather than inventing a direction.
- The UI explicitly notes the old research gate status and that higher *relative* EV can still be negative (not a statistically verified win edge or higher real-world hit rate).

## Regression
- Updated `test_v23_point_ui.js` verifies click-only mobile, desktop Enter, no auto prediction on typing/reload, correct 0–9 validation, both EV directions displayed despite underlying research SKIP, unchanged BET decision, fresh model every hand, historical state isolation, undo and scoped reset.
- Existing `test_v24_independent_points.js`, PF256/L384/M160 numerical regression, physical-EV rules, Markov, mobile viewport, authenticated loader, PR56 hash and road checks remain enforced.
- Rollback: revert this UI PR to previous main commit `90141c4a0653329c39841bdb54cd3291415d236a`. No database or history-schema migration.
