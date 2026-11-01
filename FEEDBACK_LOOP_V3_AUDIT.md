# V3 Virtual Shoe Feedback Loop / Physical EV Label Leakage Audit

Base commit: `b23db9e29f603fe4124b41d2815ec696f66f98ec`  
Branch: `fix/feedback-loop-virtual-shoe`  
Scope: virtual shoe depletion, physical EV Monte Carlo seeding, regression tests. No change to UI, existing 5× physics / 0.1× road feature scaling, the V3 `BASE_EV_GATE=.010`, overdrive 0, cushion .035, or `physicsDirection` ternary expression.

## A. Code paths and diff

### 1. `quant256_online_runtime.js`: `sampleHand` and `BlindPhysicsEstimator.observe`

- `sampleHand(counts,rng)` (original lines 37–63) already draws cards without replacement from the current 0–9-point distribution, applies baccarat third-card rules, **then** computes `result`. This function was correct and its interface is unchanged.
- **Pollution:** Original `BlindPhysicsEstimator.observe(outcome,options)` lines 85–113 tried as many as **128** hands per particle and accepted a simulated hand *only when* `trial.result===outcome` (and optionally `trial.draws===observedDraws`). The accepted simulated hand then overwrote `this.particles[i]`. This is outcome-conditioned virtual depletion, not observed card removal.
- **Fix:** Exactly one `sampleHand(this.particles[i],this.rng)` per particle per round; copy `chosen.counts` directly without checking or rejecting by `outcome`. Keep validating B/P/T and optional observedDraws; neither selects a simulated hand. Existing `observe`, `particles`, `snapshot`, `pairwise_depletion` and output field names remain usable.
- `pairwise_depletion.source` is now `unconditional_shoe_sampling` (formerly `conditional_joint_sampling`). `matched_particles` retains its legacy name for compatibility but now counts successful unconstrained simulations. `observed_draw_count` only notes that a draw-count hint was provided; **the hint does not constrain latent depletion**. `tie_or_long_hands` tracks simulated length >=5 without checking reported `T`.
- `OnlineLinUCB256.observe` (original lines 317–355) still keeps the real B/P/T training labels and calls `this.physics.observe(outcome)` for the virtual shoe; the function signature is preserved but the label cannot affect its cards.

### 2. `physics_forecast_first_runtime.js`: `summarizeHypotheses`, Monte Carlo seeds

- **Pollution:** Original `hashHistory(seq)` lines 34–38 hashed every B/P/T label; `summarizeHypotheses(physics,history,...)` original lines 131–143 used it as the Monte Carlo seed. Thus even with identical virtual composition, changing the last outcome changed simulated draws and therefore `physical.ev_banker` / `physical.ev_player`.
- **Fix:** `hashComposition(shoes)` hashes current point-count arrays only. `summarizeHypotheses(physics,_ignoredHistory,...)` keeps the original three-argument API shape but does **not** read the second argument. The caller passes `null`. The per-particle simulated hand sampling is seeded solely from the virtual card composition and particle index.
- In the **real-card** path `PhysicsForecastFirstEngine.predict`, `simulate(counts,{seed:...})` previously used `hashHistory(this.history)^0xb15f13`; now it uses `hashComposition(counts)^0xb15f13`. The observed `counts` are validated from actual supplied cards, not inferred from B/P/T.
- `simulate(counts)` and `deal(counts,rng)` (original lines 48–70, 108–130) are unchanged: EV(B) = `.95*P(B)-P(P)`, EV(P) = `P(P)-P(B)`.
- The following expression is **unchanged**, character for character: `const physicsDirection=physical.ev_banker>=physical.ev_player?"B":"P";`.

## B. Pollution path diagram

Before (virtual shoe):

```text
Entered previous result B/P/T
  -> OnlineLinUCB256.observe(outcome), old line 355
  -> BlindPhysicsEstimator.observe(outcome), old lines 85-113
  -> sampleHand(original particle) repeated up to 128 times
  -> accept only trial.result===outcome (+ optional drawn-card count)
  -> particles[i]=chosen.counts (result-conditioned card removal)
  -> PhysicsForecastFirstEngine.predict, old lines 420-421
  -> summarizeHypotheses -> deal -> simulated P(B),P(P)
  -> physical.ev_banker, physical.ev_player -> physicsDirection
```

Before (independent random-number leakage):

```text
History including last B/P/T -> hashHistory(history), old lines 34-38
  -> summarizeHypotheses stableSeed, old line 138
  -> Monte Carlo RNG, old line 140
  -> deals / P(B), P(P) -> physical EV
Real-card branch: hashHistory(this.history), old line 419 -> simulate RNG
```

After:

```text
Current residual point-count composition + seeded PRNG
  -> sampleHand() once per particle (deal cards first)
  -> simulated outcome derived after drawing (not compared to entered B/P/T)
  -> particles[i] = sampled remaining counts
  -> hashComposition(current particles), not outcome history
  -> Monte Carlo legal-hand samples -> physical EV -> unchanged physicsDirection

Entered B/P/T -> history / road / online label settlement only
             -> NOT a latent-card selection condition, NOT EV RNG seed
```

## C. Why this severs result-to-EV influence

1. A fixed initial virtual shoe and RNG seed now generate the same consumed cards for any result-label sequence of the same length. No rejection sampling based on B/P/T or user-supplied draw-count metadata.
2. Monte Carlo RNG seeds come from the point-count composition. When composition is identical, the simulated EV is *exactly* reproducible regardless of different B/P/T label strings.
3. B/P/T still updates the legitimate result history and online road/learning diagnostics. This is intentionally **not** removed. `roadPosterior` continues to use history for the 0.1× auxiliary signal/guard but has no authority to reverse the primary EV direction.
4. Observed-card inputs continue to verify that each card sequence matches the recorded outcome; the EV depletion itself removes only supplied cards, not inferred B/P/T cards.

## D. Verification

Tests:

```bash
node test_quant256_online.js
node test_physics_forecast_first.js
node test_v3_bankroll_ui.js
```

`test_quant256_online.js` now verifies B-only, P-only, T-only, and alternating result sequences (24 rounds, same initial shoe and seed) yield **identical per-particle virtual compositions**; `observedDraws: 4/6` never alters them. It retains legacy UI, 256D, V3 gate and history checks.

`test_physics_forecast_first.js` now verifies physical EV(B), EV(P), EV gap and Monte Carlo sample counts match **exactly** after each of 22 rounds for B-only, P-only and mixed B/P/T sequences. It also checks `summarizeHypotheses` produces equal results for different B/P/T history arrays when given the same shoe, while keeping 5×/0.1×/0.010 and original direction expression.

Observed during in-memory execution against original baseline vs fixed sources, using B-only and P-only with a fixed initial shoe and 12 sequential hands:

| Metric | Original baseline | Fixed |
|---|---:|---:|
| Hands in which B-only and P-only physical EV gap deviations disagreed | 12 / 12 | 0 / 12 |
| First hand B-only deviation | +0.02167 | +0.05503 |
| First hand P-only deviation | +0.08739 | +0.05503 |
| Twelfth hand B-only deviation | -0.02807 | -0.02820 |
| Twelfth hand P-only deviation | +0.03371 | -0.02820 |

These are **Monte Carlo deviation estimates**, not true advantage or betting results. Importantly, the original results did not show a consistent *monotonic same-side bias* at every hand. The verified defect is **label dependence**; the stronger monotonic-drift hypothesis is **not established**.

## E. PR

See the PR opened from `fix/feedback-loop-virtual-shoe` into `main` and the branch's Actions checks. The PR URL is recorded in the final user-facing report.

## F. Remaining risks / to verify

- **待確認:** This change removes fabricated result-conditioned card depletion, but the unconditional virtual cards still are *hypothetical*, not the real shoe. B/P/T alone cannot identify A–K rank counts and cannot prove predictive advantage.
- **待確認:** Unconditional latent depletion is no longer a Bayesian posterior conditioned on B/P/T; it intentionally reflects an unconditional prior simulation, so physical EV may become less responsive and win-rate improvement cannot be claimed.
- **待確認:** Monte Carlo finite-sample noise and seeded streams can create nonmonotonic deviations in an individual simulated shoe. Statistical monotonicity and live hit rate require out-of-sample, independently recorded full-shoe evidence.
- **待確認:** Other source files may use the recorded B/P/T for road/online training or separate nonphysical diagnostics. They are intentionally unchanged; only the physical EV computation chain was isolated.
- **Compatibility:** The `matched_particles` field name was retained for existing UI/tests but now means successfully simulated particles, not particles matching the user's result. The `observedDraws` argument is accepted/validated but not used for latent depletion.

**No guarantee of four consecutive wins, no guarantee of at most two losses, and no claim that a research-direction signal is validated positive EV.**