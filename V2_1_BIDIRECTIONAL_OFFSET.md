# BBB V2.1 雙向偏移：B/P baseline-relative research guard

## What changed

This change supplements the V2.1 PF256/L384/M160 **simulated B/P/T research signal** with a two-sided probability offset relative to the standard eight-deck baseline, without changing physical EV probabilities, EV formulas, physical direction or the observed-card BET eligibility.

Because ties are not directional wins, compare **conditional-on-non-tie** B/P probabilities:

```
B0 = BASE.B / (BASE.B + BASE.P)  ≈ 50.68%
P0 = BASE.P / (BASE.B + BASE.P)
offset_B = simulated.B / (simulated.B + simulated.P) - B0
offset_P = simulated.P / (simulated.B + simulated.P) - P0
```

**Important:** `offset_P = -offset_B` exactly (up to floating point). These are **two directions of one derived quantity**, not two statistically independent measurements or any extra information from the real shoe. No new historical trend predictor is added.

Physical direction remains exactly:
```js
const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";
```

For **B/P/T-only** unobserved-shoe research signals, an additive guard makes weak or contrary baseline-relative lifts less likely to appear as a strong signal:
- Conditional lift reference: `minConditionalLift=0.002` (0.20 percentage points).
- Maximum added research EV gate: `maxSupplementalGate=0.0025` (0.25 EV percentage points).
- `lift = offset_B` when physical direction is B, else `offset_P`.
- `extraGate = 0.0025 * clamp((0.002 - lift) / 0.002, 0, 1)`.
- Effective **research** gate is original dynamic gate + original Early-Shoe addition + original shoe-distribution addition + extraGate.

Both values are **conservative provisional starting parameters**, not discovered optimal settings; no benchmark establishes higher long-term accuracy. The guard only **filters** signals, never flips their direction or invents a positive return.

For complete observed card values, the guard is **read-only** (no extra gate) so it cannot modify verified physical EV/BET eligibility. For an empty history, the guard is likewise inactive. The result and existing page message display both diagnostic offsets with an explicit note they are correlated. There are no additional Monte Carlo draws or new UI controls.

## Fixed contracts

- Unchanged `N=256, L=384, M=160`, ESS threshold and Rao–Blackwell PF updates.
- Unchanged baseline odds, real-card physics EV and 5% banker commission.
- Unchanged physical direction, dynamic gate `0.005 / 0 / 0.035`, LINUCB, Early-Shoe and distribution guard.
- Unchanged physics weight `5×`, road weight `0.1×`.
- Unchanged authorization, licensing, user history storage keys, bankroll calculation, device binding, all existing buttons and layout.
- NO Render deployment or BGS LINE change.

## Release & test

The new `test_bidirectional_offset.js` verifies two-sided symmetry, baseline consistency, capped supplemental gate, selected-direction check, true observed-card bypass, B/P/T-only no-BET guarantee, original EV formula, model replay and undo. Existing full Quant256/physics/road/license/mobile regression checks remain.

Deploy through GitHub Pages after merging the PR into `main`. Script cache keys changed only for modified physics runtime, panel runtime and parent auth loader. The original HTML and authentication contents are protected by pinned Git blob regressions once the cache keys are reverted.

Rollback: revert the PR merge commit; use the previous main `e0a0697776a4ffba724b2a5a093ec07e7ebec8af` as the baseline.

A cleaner numerical estimate or more selective gate does **not** establish a greater practical win rate or an advantage against a casino. Independently test on untouched chronological shoes before claiming such an effect.
