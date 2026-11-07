# BBB Soft Shoe BPT-only EV architecture (2026-11-06)

## Production wiring
- Source baseline \`main\` commit: \`74100946f9eed88fdc08843cbe6ba33293142177\`.
- Auth/license gateway, reset scope, device binding, visuals, styling and unrelated legacy engines remain unchanged.
- \`auth-client.js\` now loads \`soft_shoe_decision_runtime.js\` and \`soft_shoe_panel.js\` after verification.
- \`soft_shoe_worker.js\` loads the runtime and \`soft_shoe_model_bundle.js\` inside a worker. Existing HTML B/P/T buttons are restored; 0–9 input is hidden without deleting the old implementation.
- B/P/T history is stored under license-isolated \`bgs_soft_shoe_bpt_v1\`; history is *not* imported from the old independent-score audit, as it may span unrelated shoes. Clicking End Analysis still purges the entire verified-license scope.
- A new shoe requires ending the analysis and reopening. The model stops at 72 observed hands.

## Hard-coded, configurable defaults
\`PARTICLE_COUNT=128\`, \`LIKELIHOOD_DRAWS=14\`, \`NEXT_HAND_DRAWS=14\`,
\`MIN_EV_THRESHOLD=0.0015\`, \`MIN_CONFIDENCE=0.65\`,
\`FRACTIONAL_KELLY=0.30\`, \`MAX_BET_FRACTION=0.025\`,
\`AVERAGE_CARDS_PER_HAND=4.95\`, \`DEPTH_RELIABILITY=0.5\`.

## Model semantics
1. For each observed B/P/T outcome, every virtual shoe samples 14 legal complete hands with baccarat's third-card table. Its matching-outcome frequency estimates the likelihood, multiplied by a Gaussian *soft* shoe-depth constraint. Weight-normalized systematic resampling with legal conditional transition proposals produces a new set of 10-component remaining-card count vectors. This is *latent* estimation only; no true cards or observed draw count are claimed.
2. The posterior yields point composition means, posterior spreads, 0–9 EOR sensitivities and 14 next-hand simulations per particle. The 48-element vector is 10 normalized means + 10 spreads + 10 EOR + 3 B/P/T probabilities + 3 B/P/T EVs + 4 depth/quality features + 8 weak sequence statistics.
3. The frozen attention-style sequence diagnostic supplies soft probability features only. It is **not** a pretrained LSTM/Transformer, and it has not demonstrated out-of-sample predictive uplift. Sequence weight in the live decision is zero until validated.
4. The decision layer accepts a 48D **three-class XGBoost** JSON tree bundle, with real-shoe out-of-shoe holdout and calibration metadata. The repository's existing 57D XGB file is schema-incompatible. The supplied \`soft_shoe_model_bundle.js\` intentionally marks the slot untrained.
5. Real-money advice fails closed: Banker EV = 0.95 P(B) – P(P); Player EV = P(P) – P(B); Ties push. An action requires positive estimated EV, a conservative EV lower bound, confidence, particle quality, and a validated real-shoe 48D decision model. Fractional Kelly takes tie pushes into account and is capped at 2.5%. Without a validated new model the interface **always outputs Skip**.

## Limitations and acceptance
- The current data do not identify actual unseen cards; PF uncertainty is not a calibrated confidence interval. The Gaussian depth constraint is prior information, not independent observed evidence.
- Real-world positive EV and hit-rate improvement have not been established. Do not infer them from synthetic baccarat simulations, arbitrary model metadata or isolated numerical CI checks.
- A truly trained/frozen LSTM or tiny Transformer and real-shoe trained 48D XGB, with walk-forward/out-of-shoe evaluation, are **still required** for the complete requested architecture. There is no responsible substitute for such training data.
- Dedicated tests: \`node test_soft_shoe.js\`. The legacy V4 modules and their tests remain in the repository for rollback/comparison; they are no longer loaded by the production license manifest.
- Roll back by restoring \`auth-client.js\` and \`index.html\` from baseline \`74100946\`. The new isolated history key will then remain unused.

## Deployment retry
The initial Pages upload succeeded but the deployment failed on an intermittent OIDC token timeout. A failed-job retry then encountered duplicate `github-pages` artifacts. This commit triggers a fresh workflow run with a single upload artifact; consult the new run status for publication confirmation.


## Moderate observation display (PR #158)
The previous always-display B/P research direction is replaced by a separate **research-only moderate gate**. The model still computes candidate B/P with the greater estimated EV on each successfully sampled hand. The website shows that side only if all of the following pass; otherwise it shows **觀望** and a specific reason.

| Parameter | Recommended baseline | Purpose |
| --- | ---: | --- |
| \`MIN_DIRECTION_ROUNDS\` | 3 | Avoid directional claims on a near-empty shoe |
| \`MIN_DIRECTION_EV_GAP\` | 0.020 (2.0 pp) | Minimum modeled Banker-vs-Player EV separation |
| \`MIN_DIRECTION_GAP_SNR\` | 0.70 | Modeled EV difference / estimated MC gap standard error |
| \`MIN_DIRECTION_ESS_RATIO\` | 0.35 | Ensure enough particle weight support |
| \`MIN_DIRECTION_DIVERSITY\` | 0.20 | Guard against repeated particle ancestry |
| \`MAX_DIRECTION_GAP_SE\` | 0.080 (8 pp) | Reject exceptionally noisy gap estimates |

The gap estimator combines outcome Monte Carlo variance with a nonnegative, debiased between-particle mean uncertainty contribution. It is **not** a calibrated confidence interval. These are heuristic defaults tuned to avoid unconditional forced directions and exhaustive skipping; they do not demonstrate improved hit rate or a real positive EV. They do **not** change the independent wagering gate, which continues to require a real-shoe validated 48D model. Consequently authorized bet sizes remain zero until that separate requirement is satisfied.

The original \`PARTICLE_COUNT=128\`, \`LIKELIHOOD_DRAWS=14\` and \`NEXT_HAND_DRAWS=14\`, licensing, input buttons and history storage are unchanged. A clean real-shoe holdout must be run before representing this as a predictive or staking advantage.


## Layered prediction V2 (2026-11-06, source baseline ededd462)

**Scope**: Upgrade ONLY the prediction runtime and prediction worker's frozen-model loader. The unchanged website layout, 6x15 Big Road, 3 B/P/T result buttons, CSS, training data exports, license/device/UID gate, license storage, and session persistence remain intact. Cache-buster strings in the frontend loader/panel/HTML change solely to force the new production modules to load.

**Physical layer** is unchanged and follows the requested defaults: Particle Filter N=128, 14 legal complete-hand likelihood draws per particle per observed B/P/T result, 14 legal-hand next-round draws per particle, soft Gaussian depth constraint (4.95 cards/hand, reliability=0.5), 10-point latent shoe vectors, physical B/P/T probabilities, ROI EV, EOR sensitivities, ESS/diversity and MC uncertainty. 8 decks = 416 cards. No actual cards/ranks are observed or claimed.

**Frozen sequence inference** now includes a real bounded 8-input, 8-hidden LSTM forward-pass runtime (\`soft_shoe_frozen_lstm_runtime.js\`) with a maximum context window of 24 B/P/T rounds. Its per-round inputs contain B/P/T one-hot outcomes, run length, continuation indicator and trailing-8 B/P/T rates (weak road features). The optional \`soft_shoe_frozen_model_bundle.js\` is intentionally marked \`trained:false\` because independent real-shoe LSTM training data and weights were not supplied. Therefore the production system uses the explicitly named **untrained weak sequence diagnostic** (0 direct direction weight) until frozen/real-shoe trained, held-out and calibrated LSTM weights are available. No synthetic/fake pretrained weights are supplied.

**Physical + Frozen Core decision feature schema** has been versioned from \`soft_shoe_bpt_48d_v1\` to **\`soft_shoe_bpt_48d_v2\`** to prevent using an old incompatible XGBoost. The 48 features are:
- 10 posterior mean remaining card counts and 10 posterior between-particle spreads, each scaled by 416,
- 10 diagnostic EOR sensitivities, 3 B/P/T physical probabilities, 3 B/P/T physical EV values,
- 4 shoe depth/ESS/diversity/simulation quality features,
- 5 weak recent-sequence features (last 4/16 Banker and Tie rates and final-streak length),
- 3 Frozen Core B/P/T soft probabilities.

Optional \`SEQUENCE_WEIGHT\` is 0 by default. If supplied in configuration, nonzero blending (maximum 0.05) occurs **only if** the frozen LSTM model has real-shoe holdout validation and passes strict shape/finite checks. The live untrained fallback cannot influence either physical EV or real-money advice. A proper trained, calibrated 3-class XGBoost bundle with **matching v2 48D schema** and real-shoe out-of-shoe validation is still required for action \`Bet Banker\` or \`Bet Player\`. The existing untrained XGBoost bundle remains intentionally untrained, just with its schema ID updated to v2.

**Formal gate** unchanged: MIN_EV_THRESHOLD=.0015, MIN_CONFIDENCE=.65, LOWER_BOUND_Z=1.645, quality >= .65, 30% fractional Kelly, 2.5% bankroll fraction cap, 0.5 resampling/depth reliability. Even a displayed moderate B/P research direction is NOT a qualified wager. Live formal action remains \`Skip\` and suggested fraction \`0\` without a calibrated, verified XGBoost.

**Moderate research display gate** unchanged: 3 rounds, 2.0pp B/P EV difference, gap/estimated SE >=.70, ESS >=.35, ancestor diversity >=.20 and estimated gap SE <=.08. This produces visible B/P/SKIP decisions for research only; hit rate and real-money positive EV are unvalidated.

**Verification**: \`node test_soft_shoe.js && node test_soft_shoe_frozen_lstm.js && node test_bpt_big_road.js && node test_license_storage.mjs\`. Dedicated CI checks model schema compatibility, deterministic physical equivalence when the LSTM is untrained, valid LSTM inference with synthetic test-only weights, observed-signal updates, betting eligibility fail-closed behavior, and the preserved UI/license contracts. A true LSTM training run, genuine held-out XGBoost training and real-shoe profit validation cannot be completed without labeled real shoe histories; **do not represent the architecture as proven profitable**.

**Cross-repo note**: \`BGS0.3V.1\` LINE OCR repo previously vendors an immutable copy of BBB \`ededd462\` V1. This BBB-only change intentionally does NOT modify BGS; a separately requested verified port would be needed to align the LINE bot with this v2 model.


## Formal wager-only dynamic gate parameters (2026-11-07)

Only the **formal wager eligibility and fractional Kelly stake-sizing policy** was changed. The existing latent Particle Filter, physics simulation, EOR, 48D feature schema, Frozen LSTM, XGBoost adapter, B/P/SKIP research-display gate, license/session data, buttons, 6×15 Big Road and page design remain unchanged. Asset version strings are refreshed solely to avoid stale browser JS.

| Config key | Default | Description |
| --- | ---: | --- |
| \`MIN_EV_THRESHOLD\` | 0.00300 | Remaining fraction > 0.60: 0.30% modeled EV |
| \`MID_SHOE_MIN_EV_THRESHOLD\` | 0.00235 | 0.30 ≤ remaining fraction ≤ 0.60: 0.235% |
| \`LATE_SHOE_MIN_EV_THRESHOLD\` | 0.00190 | Remaining fraction < 0.30: 0.19% |
| \`EARLY_SHOE_REMAINING_RATIO\` | 0.60 | Top EV tier boundary |
| \`LATE_SHOE_REMAINING_RATIO\` | 0.30 | Bottom EV tier boundary |
| \`MIN_CONFIDENCE\` | 0.75 | Standard modeled confidence threshold |
| \`LATE_CONFIDENCE_REMAINING_RATIO\` | 0.35 | Confidence becomes stricter below this fraction |
| \`LATE_MIN_CONFIDENCE\` | 0.80 | Late-shoe modeled confidence threshold |
| \`MIN_BET_EV_GAP\` | 0.0015 | Minimum absolute Banker–Player EV difference |
| \`MIN_BET_ESS_RATIO\` | 0.45 | Require posterior ESS ≥ 45% of particle count |
| \`MIN_BET_QUALITY\` | 0.65 | Preserve earlier composite ESS × diversity quality gate |
| \`MAX_BET_EV_SE\` | 0.060 | Reject selected-side EV estimator SE > 6.0 percentage points |
| \`FRACTIONAL_KELLY\` | 0.225 | Use 22.5% of full Kelly, if formally qualified |
| \`MAX_BET_FRACTION\` | 0.0175 | Single wager maximum 1.75% of bankroll |
| \`MIN_REMAINING_CARDS_FOR_BET\` | 25 | End-shoe wager blocker threshold |
| \`END_SHOE_OVERRIDE_MIN_EV\` | 0.025 | Exception requires ≥2.5% modeled EV |
| \`END_SHOE_OVERRIDE_CONFIDENCE\` | 0.90 | Exception requires ≥90% modeled confidence |
| \`END_SHOE_OVERRIDE_MAX_EV_SE\` | 0.030 | Exception requires selected-side EV SE ≤3 pp |

Estimated remaining cards are the **mean across all posterior particles**, not a known card count. The existing 72-hand maximum means <25 estimated cards is unlikely in the default configuration, but the condition is explicitly implemented and unit-tested.

The model also retains the previous conservative EV lower-bound requirement: \`EV_candidate − 1.645 × estimated_EV_SE > effective_min_EV_threshold\`. Formally qualified stakes are zero unless **every requirement** is met and a real-shoe trained, calibrated, independently held-out matching \`soft_shoe_bpt_48d_v2\` XGBoost decision model is installed. No such validated model is currently included, so official bet authorization remains \`Skip\` and bankroll fraction remains 0. Research-view B/P/SKIP display rules remain identical to the earlier implementation.

These policy values are **hypotheses, not empirical validation**: a 53–54% conditional hit rate has not been demonstrated. Backtesting must track actual qualifying bets and uncertainty using separate held-out real shoes, including class frequencies, actual commission, bet frequency, observed hit rate, and confidence intervals. A more restrictive policy may lower bet frequency without increasing hit rate.
