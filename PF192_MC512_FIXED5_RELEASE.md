# PF192 / MC512 / fixed-five sampling upgrade

Sampling budget only; entry policy unchanged: SE >0.55%, no minimum EV gap, unequal and aligned smoothed/raw EV choices, batch agreement >=55%. Negative EV values remain eligible for display. Deplete remains diagnostic, not mandatory.

- 192 actual particles per batch (validation and estimator cap both raised).
- 512 next-hand simulations per particle.
- Exactly five independent batches, 960 particles total and 491520 primary draws.
- EOR: 48 stratified particles x192 paired samples.
- Deplete: 16 internal batches x32 shoes x128 samples =65536 per primary batch; 327680 across five batches.
- Fresh random streams and explicit-seed replay are preserved.
- Entry policy, smoothing formula, commission, model inputs, exact reference, authentication and storage are unchanged.

Local diagnostic for seed0 score pairs 59,00,84: approximately 0.95–1.23 seconds per full fixed-five calculation on this machine; not a Render/mobile latency guarantee. All three had SE below0.55% and therefore correctly observed under the retained policy. More computation reduces numerical noise, not missing-card uncertainty, and does not establish higher hit rate or profitability.

Updated raw numerical fixtures intentionally reflect the increased sample budgets. Do not compare their hashes with the previous sampling configuration.
