# BBB Soft Shoe — derived road auxiliary research layer

## Contract, scope and evidence

- **Baseline:** BBB Soft Shoe V5.1, `BBB_SOFT_SHOE_BPT_EV_V5_1`.
- **Inputs:** this shoe's final outcomes `B/P/T` only. Neither card ranks nor actual deck composition are observed or invented.
- **Isolation:** `derived_road_aux.js` is a *pure* B/P/T road-shape diagnostic. It never enters Soft Shoe PF particles, likelihoods, physical EV, ESS, standard errors, candidate selection, wager gate, fractional Kelly or bankroll fraction. The separate `bpt_big_road.js` is unchanged.
- **Research-only outputs:** `report.road_aux`, `research_view.road_confirmed`, `research_view.road_conflict`, `research_view.road_confidence_label`, `research_view.research_aux_edge_score`, `future_path.research_aux_edge_score`, and, when a future score exists, adjusted `future_path.future_edge_score` plus original `future_path.unadjusted_future_edge_score`.
- **Safety:** all road signals carry `estimated_only:true`, `validated_betting_edge:false`; with untrained models formal action stays `Skip`, `validated_edge:false`, `bankroll_fraction:0`.
- **No claim of uplift:** derived roads contain **no new information beyond B/P/T**, and may not improve real-shoe prediction. Any display confidence adjustment is a *heuristic*, not a calibrated probability or validated betting edge. Actual out-of-sample accuracy must be established prospectively before promoting anything.

## Big Road and derived-road algorithm

`buildBigRoad(history)` removes all ties (ties are annotated on prior cells in conventional displays, never separate columns), then groups consecutive equal B/P outcomes into **logical columns** with unbounded depths. The UI may wrap dragon tails at six rows; the comparisons use logical, unwrapped column lengths.

For a newly appended Big Road bead at zero-indexed column `c`, row `r`, compute each derived road at offsets `k=1,2,3` (Big Eye / Small / Cockroach):

- If `r=0`, begin only at `c>=k+1`; red if completed lengths `column[c-1]` and `column[c-k-1]` match, else blue.
- If `r>0`, begin only at `c>=k`; inspect the comparison column at `c-k` and row `r` and row `r-1`. Both occupied / both empty = red. Exactly one occupied = blue.
- Until there is enough prior data, mark `N`. `probes` are the most recent applicable red/blue mark of each derived road, not predictions of actual winners.

**Important:** derived-road red `R` means a regular *shape*, and blue `B` means an irregular *shape* — **not** banker and player. The additional, unvalidated research mapping applies a majority vote across non-`N` probes: red majority → the last non-tie B/P side is a *continuation hypothesis*; blue majority → the opposite side is a *switch hypothesis*; a tie or fewer than two available probes → `NEUTRAL`. No mapping can flip the primary `candidate`.

## Parameters (DEFAULT)

| Parameter | Value | Meaning |
|---|---:|---|
| `ROAD_AUX_ENABLED` | `true` | Toggle research road layer |
| `ROAD_AUX_MIN_HISTORY` | `12` | Minimum non-tie observations |
| `ROAD_AUX_AGREEMENT_MIN` | `0.67` | Minimum derived-road vote agreement |
| `ROAD_AUX_CONFIDENCE_MIN` | `0.45` | Minimum heuristic confidence for confirmation |
| `ROAD_AUX_MAX_BOOST` | `0.08` | Max additive research score adjustment; config cap 0.10 |
| `ROAD_AUX_CONFLICT_PENALTY` | `0.06` | Deduction from the research score; no candidate reversal |
| `ROAD_AUX_REQUIRE_SOFT_ELIGIBLE` | `true` | Road confirmation requires original research `eligible` regardless of setting; no path promotes an ineligible Soft signal |

The road confidence is a bounded function of observed non-tie depth, usable derived roads, and vote agreement; **it is not the chance of winning**. Agreement and confidence thresholds only affect optional research display annotations.

## Fusion, presentation and rollback

After Soft Shoe finishes the physical forecast, original `moderateDirectionGate`, and formal `wagerGate`, `fuseRoadResearch` may adjust the **research-only** short-horizon score. If the future path supplies `future_edge_score`, this is adjusted and the original retained as `unadjusted_future_edge_score`; otherwise a bounded SNR proxy is used. Both are heuristic visual scores. Only an *originally eligible* Soft candidate can be marked `road_confirmed`; contradictory roads mark `road_conflict` but never change the candidate, `research_view.eligible` or formal `action`.

To restore pure Soft Shoe V5.1 immediately, set `ROAD_AUX_ENABLED:false` in a `run(history, { ROAD_AUX_ENABLED:false })` option or set the single DEFAULT flag to false and redeploy. When disabled, the original research object and future-path diagnostics are preserved (except the extra disabled diagnostic `road_aux` and the config flag). No restoration of particle state or model weights is needed.

Complete repository rollback: branch `backup/pre-road-aux-20261112` points to the original BBB/main `4599789a9df519de3976d1a2f38cf459a175caf0`. Use `git revert <merge-commit>` on main or restore the commit via PR; do **not** force reset a live shared main.

## Validation and limitations

Run `node test_soft_shoe.js` and `node test_soft_shoe_frozen_lstm.js`. Check parity with `ROAD_AUX_ENABLED=false`, T-only and short-history contracts, pure derived colors, artificial confirmation/conflict bounded score changes, unchanged candidate, and identical PF particles/weights and formal gate across enabled/disabled. Any match-rate study requires **chronological holdout with shoe-grouped splits**, no leakage of the target hand into the input history, identical selection criteria and confidence intervals. A "Soft + road-confirmed only" selective subset may increase the observed accuracy simply by betting less; compare coverage and uncertainty as well.

The BBB upgrade is **not** an automatic sync to BGS. BGS remains pinned to its previously locked BBB V5.1 Git blob until a separately verified, explicit BGS change.
