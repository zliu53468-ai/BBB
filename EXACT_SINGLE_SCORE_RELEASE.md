# Web exact enumeration release

The web point-only panel now uses `BBB_EXACT_SINGLE_SCORE_V1` via `exact_score_runtime.js`. LINE retains the previously deployed independent PF96 sampler until separately authorized. This changes web forecasts: under the existing V4 threshold all 100 possible score pairs abstain. All exact Banker and Player EVs remain negative under the eight-deck prior. The relative EV candidate is Banker for every pair and is not a betting recommendation.

The reference builder exhaustively enumerates 339,400 legal rank-value sequences and merges them into 6,958 depletion multisets. It integrates exact next-hand outcome probabilities over every latent state with its conditional evidence weight. It uses double arithmetic with normalization checks at 1e-10. No history or observed card ranks are fabricated; every request uses only one final-score pair.

Reproduce from repository root with PowerShell 7 and Python/Node:

```text
pwsh -NoProfile -File tools/exact_score/build.ps1
python tools/exact_score/package_table.py
node tools/exact_score/compare_legacy.js
pwsh -NoProfile -File tools/exact_score/validate_million.ps1
python tools/exact_score/report.py
node test_exact_score_runtime.js
```

The million-case validation uses a separate physical-deck partial Fisher-Yates simulator, 1,000,000 independent fresh-shoe two-hand scenarios (2,000,000 dealt hands), seed 20261104, no fitted parameters and choices frozen before target-hand dealing. It validates calibration within this model, not a live-casino edge. Ties are pushes and excluded only from non-tie hit rates. Formal display abstains, so its hit rate is undefined; do not substitute the candidate's hit rate as the signal hit rate.

Rollback: restore the previous web panel, guard and loader asset versions from commit 8064f5fe together. Authentication, license handling and LINE are unchanged.

## Previous release

# Independent conditional particles v1

Each point-only input creates a fresh PF96. Each particle independently draws a legal first-four proposal, enumerates legal third-card branches matching both final totals, and accepts with the matching probability mass. This yields conditional draws without the previous finite-proposal likelihood weights. Each accepted hand depletes only its own hypothetical shoe; M96 next-hand simulation and V4 display thresholds remain unchanged.

A per-particle 4096-proposal limit fails the request atomically rather than substituting an unconditioned shoe. Reusing this mode across multiple hands is rejected. Legacy historical/BPT modes retain their existing sampler.

The initial composition remains an eight-deck assumption, not observed remaining cards. Determinism, complete conditional draws and cross-platform parity are correctness checks, not evidence of increased real-world hit rate. Roll back by removing exactSingleScore:true from the web CONFIG and LINE worker CONFIG together, and repin the LINE source.
