# V14.8 Safe Patch — Offline Shadow Report

- Baseline: `7777a321be4476d38979f33625294fdfeecc118a`
- Dataset: 20 seeded synthetic eight-deck shoes / 1,397 completed hands
- Evaluation: latest 20% chronological shoe holdout (4 shoes, 273 eligible predictions)
- Command: `node tools/ev_consistency_shadow.js --synthetic 20 /tmp/bbb-v14-8-safe-shadow.json`
- Limitation: synthetic mechanics-only replay checks causality and runtime behavior; it is not evidence of live predictive advantage or profitability.

## Production-default equivalence

`EV_CONSISTENCY_V1=false` and `PARTICLE_BLEND_MODE="quality"` matched the exact baseline commit on every holdout prediction. Direction, Final P(B), final EV values, Physics candidate, and Volume Guard state had zero mismatches.

## Patch-1 EV consistency shadow

| Metric | V14.8 baseline | EVC1 candidate | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Hit rate on non-tie bets | 0.611940 | 0.606061 | -0.005880 | FAIL |
| Skip rate | 0.710623 | 0.714286 | +0.003663 | PASS |
| Realized EV per placed bet | 0.176582 | 0.166667 | -0.009916 | FAIL |
| Absolute correct bets | 41 | 40 | -1 | PASS (97.56%) |
| Conditional-direction Brier | 0.249327 | 0.249340 | +0.000014 | Report-only |

Acceptance requires hit rate not below baseline, realized EV at least baseline minus 0.001, correct bets at least 95% of baseline, and Skip delta within [-5pp, +2pp]. EVC1 failed the hit-rate and realized-EV gates, so `EV_CONSISTENCY_V1` remains `false` in production.

## Patch-2 Particle500 ablation

| Mode | Hit rate | Skip rate | EV / bet | Brier | Mode invariant violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quality` | 0.611940 | 0.710623 | 0.176582 | 0.249327 | 0 |
| `off` | 0.565217 | 0.706960 | 0.098750 | 0.249188 | 0 |
| `fixed_0.06` | 0.590909 | 0.717949 | 0.142857 | 0.249106 | 0 |
| `fixed_0.10` | 0.597015 | 0.706960 | 0.150000 | 0.249268 | 0 |
| `fixed_0.16` | 0.600000 | 0.703297 | 0.159259 | 0.249400 | 0 |
| `uncertainty_only` | 0.565217 | 0.706960 | 0.098750 | 0.249197 | 0 |

The invariant audit verifies exact fixed weights, no Particle call in `off`, and unchanged winner/EV heads in `uncertainty_only`. These results do not promote an ablation mode; production remains `quality`.

## Macro EMA

The checked-in model remains unchanged with coefficients `{six_card: 0, low_score: 0, point_diff: 0}`. `macro_ema.js` only adds readiness metadata; with the production bundle its formal probability delta remains exactly zero.
