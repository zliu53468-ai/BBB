# Independent Deplete MC V1 — web shadow release

The existing PF96/V4 model remains the sole source of formal Banker/Player/SKIP display. The user explicitly selected comparison-only deployment after the candidate failed to demonstrate higher hit rate or commission-adjusted return. LINE is unchanged.

`deplete_mc_runtime.js` creates 256 conditional hypothetical shoes (8 batches of 32), independent of the primary PF random stream. Each shoe runs 128 target forecasts paired with a fresh-shoe forecast using the same uniforms. The fresh-shoe baseline plus the mean paired depletion delta estimates probabilities with reduced sampling noise. All conditioning follows legal baccarat draws and matches both final scores. Each proposal has a finite 4096-attempt budget. No actual remaining cards are claimed.

The module and PF share the SAME final-score observation and eight-deck assumption. They are not independent evidence. No likelihood multiplication or independent-evidence confidence boost is allowed. A fixed experimental 80% PF / 20% deplete candidate is evaluated offline, and existing quality uncertainty is never reduced by the fusion. Candidate direction is not deployed.

The active web path calls `attachShadow`. It adds diagnostic fields only; every original report field, EV, V4 decision, research gate and formal BET qualification remains identical. If the auxiliary calculation fails, the original PF report remains usable and the UI labels the comparison unavailable. The web labels successful comparisons as not affecting the displayed signal.

## Results (synthetic; not live-casino evidence)

- New held-out seed: 20261105; 1,000,000 independent fresh-shoe two-hand scenarios; 2,000,000 hands.
- Target simulator uses a separate physical deck and partial Fisher-Yates implementation. Choices are frozen before dealing the target hand. No parameters are fitted to this holdout.
- Original display distribution over 100 score inputs: B36 / P24 / SKIP40.
- Experimental candidate: B32 / P22 / SKIP46.
- Non-tie hit rates: original 50.0301%; candidate 49.9934%. Different coverage means these percentages alone are not a controlled gain estimate.
- On identical candidate entry cases, candidate direction and primary raw direction are identical; paired profit difference is zero.
- Neither overall candidate ROI nor the paired comparison to fixed Banker establishes an advantage. Promotion eligibility is false; production mode is shadow.
- EV RMSE against the exhaustive fixed-prior reference drops from about 0.00601 to 0.00481 for the candidate. Numerical precision is not a win-rate improvement.

Reproduce from repository root:

```text
node tools/validate_deplete.js
node test_deplete_shadow.js
node test_v23_point_ui.js
```

Set WRITE_DEPLETE_TRACE=1 to save the raw two-byte-per-scenario trace. Its schema is score index (player*10+banker), then target outcome 0=B, 1=P, 2=T. The validation workflow uploads the trace and report as an artifact. The exact-reference file is used offline only and is never loaded into the deployed panel.

Rollback: restore the previous auth-client.js, index.html, quant256_online_panel.js and loader tests from 7bb37ba9; remove the auxiliary loader. No PF core or V4 thresholds need to change.
