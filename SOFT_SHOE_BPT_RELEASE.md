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
