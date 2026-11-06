# Adaptive PF192/MC512 release

Supersedes the fixed-five upgrade. Base prediction flow restored from BBB5487b8a; only particle count and primary next-hand simulation budget increased (with the necessary particle cap192).

PF192, MC512, adaptive3–5 batches; EOR24x96 and Deplete32768 per batch restored. Entry policy unchanged: SE>0.55%, unequal aligned smoothed/raw EV values with no minimum gap, agreement>=55%. Deplete is diagnostic only. Fresh seed/replay and baseline/smoothing formulas are unchanged.

The 100 score pairs are the full domain of Player0–9 x Banker0–9, not an input-history limit, run limit or simulation limit. Actual sampling uses192x512 draws per primary batch and can run indefinitely for successive valid inputs. Exact-reference table is a numeric check, not a lookup that selects the prediction.

This release does not claim to eliminate Banker preference, improve hit rate or generate positive returns. Higher sample budgets can reduce SE below the retained lower threshold and therefore increase abstention.
