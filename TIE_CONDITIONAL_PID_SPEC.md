# Tie-Conditional Decision Control v2

This patch preserves the Frozen 256D Core, 213D→48D Physics MLP, 57D feature order, and direct binary XGBoost classifier. It changes only the decision-space probability handling and entry-volume controller.

## 1. Time-Series 3-Class Conditional Calibration

The 48D Physics winner distribution is read with fixed zero-based indices:

| Physics index | Name | Meaning |
| --- | --- | --- |
| 23 | `winner_p_b` | Physics Banker probability |
| 24 | `winner_p_p` | Physics Player probability |
| 25 | `winner_p_t` | Physics Tie probability |

Runtime and backtest defensively clip and normalize these three values. When invalid, they use the standard fallback `[0.4586, 0.4462, 0.0952]`.

```text
pNonTie = max(0.001, 1 - pTie)
pBCond  = stage/noise/remaining-ratio calibrated Final P(B)
pPCond  = 1 - pBCond

EV(B) = pNonTie * (pBCond * 0.95 - pPCond)
EV(P) = pNonTie * (pPCond - pBCond)
```

Calibration is stateless and browser-safe. It applies stage temperature scaling, zero-default learned bias, noise shrinkage, and remaining-ratio uncertainty shrinkage. Its parameters are deployment configuration, not a new model or extra feature.

## 2. Continuous PID-compatible Volume Controller

The controller uses only preceding rows from the same shoe. It never reads settled labels, so it cannot leak future outcomes into the decision.

```text
actionRateDeficit = max(0, targetActionRate[stage] - actualActionRate)
correctDeficit    = max(0, 0.95 * baselineExpectedCorrect - expectedCorrect)

P = max(actionRateDeficit / 0.10,
        correctDeficit / (0.05 * baselineExpectedCorrect))
I = mean(previous relaxation weights in the rolling 16-row same-shoe window)
D = P - previous relaxation weight

r = clamp(P + 0.10 * I + 0.05 * D, 0, 1)
effectiveBand = max(0.010, band - 0.006 * r)
effectiveEV   = max(0, activationEV - 0.004 * r)
```

This removes the old Boolean jump. A 10pp action-rate deficit or a 5% expected-correct deficit drives `r` to its capped maximum; smaller deficits yield proportionally smaller relief.

## 3. Physics conflict filter

An otherwise eligible entry becomes `Skip` when:

1. conditional XGBoost distance is at least `0.035`; and
2. the opposite Physics winner probability exceeds the model-side Physics winner probability by more than `0.08`.

The filter is a hard safety override. Volume control cannot bypass it.

## 4. Snapshot Schema 7 additions

Each prediction persists `p_tie`, `p_non_tie`, conditional B/P probabilities, conditional temperature/shrinkage, Physics B/P probabilities, sanity-conflict state/reason, action/correct deficits, PID integral/derivative, and `volume_relaxation_weight`. Older snapshots remain readable with safe fallbacks.

## 5. Acceptance requirement

The existing chronological shoe-level hold-out gates remain mandatory: Brier, Skip band, placed-bet hit-rate, EV per bet, and Absolute Correct Bets. This patch must not be represented as a demonstrated accuracy gain until a new chronological hold-out retrain passes those gates.
