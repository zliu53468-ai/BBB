# BBB V2.3 · History first → Final-score conditioned particle filter

## Workflow

1. Enter initial historical Banker/Player/Tie results using the **existing** B/P/T buttons (original licensed state key unchanged).
2. Tap **開始分析** once. The existing predictions run, then the same compact controls card swaps the B/P/T buttons for two final-score selects: **left = Player, right = Banker**, integers 0–9.
3. After the two scores have been selected, the known outcome (B/P/T) is calculated once. It is committed through the **existing** B/P/T button event and its original history persistence. The separate scored-round record is stored under the license-managed `bgs_v23_point_score_log` key. No second Start click is necessary.
4. The model receives the exact final-score pair for this *settled* hand and produces a prediction for the **next**, still-unseen hand.
5. Undo drops the latest history and its corresponding score evidence; a corrected score replay rebuilds the particle posterior deterministically. Refresh restores the stage and scored observations through license-scoped local storage. End Analysis clears all scoped history and stage keys as before.

## Inference

The original `N=256` particle posterior and `L=384` first-four-card proposals are preserved. V2.1's Rao–Blackwell enumerator now optionally conditions its legal baccarat third-card proposals on **both final totals** `(Player, Banker)`, in addition to the observed outcome. The proposal's mean branch mass is the conditional likelihood for weighting each parent particle. A legal conditional hand is sampled by weighted reservoir without replacement. ESS resampling/rejuvenation is preserved; for rarer score pairs duplicate-parent rejuvenation tries up to 192 valid proposals (32 for the original B/P/T-only path). No unknown ranks are treated as observed.

Earlier history without point scores continues to use the **exact** original B/P/T-only path. Newly entered point scores improve the *constraint information* for possible virtual residual decks, but still do not reveal the actual four-to-six cards or guarantee more accurate real-world predictions.

## Unchanged

- Physical EV equations and B/P/T outcome rules, payout/commission formulas, physical direction authority and BET eligibility.
- V2.2 Markov research-only shadow, physics weight 5× / road 0.1×, LinUCB, early-shoe/distribution guards, dynamic gate 0.005/0/0.035, physical Monte Carlo M=160.
- Existing login, Cloudflare license/device binding, mobile buttons, old B/P/T storage format, end-analysis cross-device purge.
- No Render, no BGS LINE modifications.

## Compact UI

One controls card, with two mutually exclusive stages; no appended long second panel. All original controls retain their IDs. The status row shows how many rounds have additional final-score information. Mobile select fields use 16px text to avoid Safari auto-zoom.

## Validation & rollback

`test_v23_point_conditioned.js`: compares conditional score likelihood against independent exhaustive enumeration of all 720 permutations of a six-card toy shoe, legal residual-count constraints, B/P/T equivalence for unscored history, deterministic score replay/undo, ESS diagnostics and absence of a fake BET edge.

`test_v23_point_ui.js`: verifies history → Start → selects → one B/P/T settlement → aligned score persistence → undo → corrected score → reload → scoped reset, using a lightweight browser fixture.

Run the entire existing Quant256/physics/road/license/mobile CI before merging. Rollback by reverting the V2.3 PR merge. No schema migration is required; scored-round keys are additive and license scoped.

**Caveat:** precise score likelihood may itself be a rare event and reduce PF effective sample size. This update is a measurement and workflow improvement, not proof of an advantage or a higher hit rate. Assess speed and chronological out-of-sample accuracy separately.
