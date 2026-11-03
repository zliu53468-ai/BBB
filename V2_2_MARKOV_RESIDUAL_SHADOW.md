# BBB V2.2 — Markov-Residual Fusion (shadow-first)

Base production: `df49e8d5e3e58e899906c3b959fd5ac8572cdf00`.

## Shipping scope

- Remove V2.1 bidirectional-offset research guard, UI texts, and obsolete test/doc. This restores the prior `dynamicGate + earlyShoeGuard + distributionGuard` research gate, without a hidden offset term.
- Add `markov_residual_runtime.js` (separate from the legacy `bayes_markov_runtime.js` used by other interfaces). It estimates the first-order B/P transitions `B→B`, `B→P`, `P→B`, `P→P` causally within the current shoe. Ties are retained in totals but skipped for directional transition counts.
- Markov posterior `P(B|last side) = (n(last→B) + 64×P0(B|non-tie)) / (n(last→B)+n(last→P)+64)`. Use only when `support ≥ 12` for informational shadow calibration; shorter samples display diagnostics with `eligible=false`.
- A what-if research candidate can shift simulated conditional banker probability at most **±0.003** (0.30 percentage points), with reliability `support/(support+64)` and max hypothetical Markov mixing weight **0.10**. It does **not** alter the model's actual simulated outcome probabilities or official EV outputs.

## Guardrail / limitations

**Production applied Markov weight is 0.** Markov has no active influence on real-card EV, particle weights, physical direction, EV gate, or BET qualification until an independent chronological validation demonstrates a gain. This is essential because both Markov and particles use the same B/P/T history: interpreting both outputs as independent evidence risks double-counting.

`N256 / L384 / M160`, ESS trigger, Rao–Blackwell conditional deals, legal baccarat drawing rules, the 0.005/0/0.035 dynamic gate, LinUCB, Early-Shoe Guard, shoe distribution guard, 5× physics/0.1× road, stored histories, device-bound Cloudflare licensing, controls, and original bankroll UI remain unchanged. Browser script cache keys are updated. **No BGS LINE or Render changes.**

## Validation

New `test_markov_residual_v22.js`: causal transition counting, tie handling, correct smoothing, evidence gating, optional shadow caps, no mutation, no BET/PF/real-EV overwrite, deterministic undo/replay, module load order. Existing Quant256, physical EV and regression tests remain in GitHub Actions. **No validated improvement in long-run hit rate or payout is claimed.**

Recommended *future* promotion criterion: untouched, chronological shoe-level out-of-sample comparison against current model (prediction Brier / log loss, true/non-tie accuracy, confidence intervals, signal coverage, net EV including 5% banker commission), with a fixed model selection and no lookahead. Only after passing that gate should a separate PR enable applied Markov fusion or add an importance-corrected, posterior-valid particle proposal.

## Rollback

Revert the V2.2 PR merge, restoring the exact previous production commit `df49e8d5e3e58e899906c3b959fd5ac8572cdf00`. No user storage format changes or database migration.
