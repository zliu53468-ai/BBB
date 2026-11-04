# Local-card EOR / PF5000 / MC10000 / strict V4

This replaces the web's PF80/Deplete20 final-total input path. The license client
loads only pure_card_panel.js; the panel starts a dedicated Worker which loads
the independent model, profile and combinatorial kernel. Legacy files remain
in the repository for rollback but LinUCB, contexts and road models are not loaded.
License/authentication/reset logic outside the script inventory is unchanged.

## Input and model order

Both sides require their actual 2-3 ordered card values, not a single final total.
A=1, 10/J/Q/K=0. Natural and all third-card rules are validated. Invalid input,
assets missing, Worker errors, timeout or storage failure produce SKIP. There is
no old-model fallback, and stale requests cannot restore a previous direction.

1. EOR sums the observed removal vector and multiplies by alpha=.05.
2. PF independently samples 5000 integer composition scenarios each request.
3. Exact composition probabilities are calculated for the weighted mixture.
4. Deplete MC draws 10000 weighted-mixture next hands, without replacement
   inside each hand; inventory is reset between independent trials.
5. Strict V4 requires EOR sign, highest theoretical EV side and highest MC EV
   side to agree; selected theoretical EV uplift vs the same fresh-shoe side
   must be >=.0025, and selected MC win-probability standard error <=.0015.
   These retained settings will commonly yield SKIP. No accuracy gain is claimed.

## Explicit interpretations and supplemental settings

The latest user-specified coefficients A=.0005, 4=-.0011, 9=.0008 are retained.
Other seven values use previously verified standard banker-minus-player EOR
differences. This hybrid is labelled CUSTOM, not a standard EOR or measured EV.
The 416-card denominator is already in per-card reference EOR; there is no
second division by 416. alpha=.05 shrinks the sensitivity score.

Each rank's count is centered on fresh-eight-deck minus known cards, with rounded
Gaussian sigma=2.5 and rejection above that center or below zero. This is a
one-sided sensitivity prior, not an inferred hidden-shoe posterior. Importance
target = proposal * exp(-.5*((particleEOR-observedEOR)/.0001)^2); target/proposal
therefore cancels the proposal term. Bandwidth .0001 is an explicit heuristic.
It does not add independent evidence or reconstruct unknown shoe depth.

The exact kernel groups all 339400 legal point-value deal paths into 6958 removal
multisets. Every path probability uses falling factorials and the appropriate
4/5/6-draw denominator. Equal compositions are cached for one prediction only.

"Simulation standard deviation" is explicitly implemented as sqrt(p*(1-p)/10000),
the direct independent-MC win-probability standard error. Particle heterogeneity
and real-world model uncertainty are not hidden inside this number. Threshold
.0025 means EV uplift, not absolute positive EV. bet_allowed remains false and
no recommendation is labelled as a proven betting advantage.

## Diagnostic and offline records

PF and EOR do not read previous rounds. Markov is the only stateful policy overlay.
Users must check that the new input is exactly the prior prediction's next hand
before it is used as a settlement. The checkbox defaults off and resets off.
No inferred labels are generated across gaps. Reset diagnostics when changing
tables or omitting hands. Window=20 settled rows; ties are recorded but excluded
from opposition-rate evidence. With >=8 non-ties, last three opposed, and overall
opposition rate >=.6, add .0005 to the EV threshold; never switch a direction.
These trigger constants are documented heuristics, not validated performance gains.

Local, license-scoped exports retain point difference, card counts/type, EOR
direction and confirmed next-hand result. No online context learning or automatic
EOR updates; no upload to another service. Existing authenticated end/reset is
preserved. The reset-diagnostics button clears new model local records.

## Checks and rollback

node test_pure_cards.js
node test_pure_card_panel.js

Checks cover exact eight-deck baseline, parser, draw rules, particle bounds,
weight normalization, deterministic reset, gate boundaries, Markov tightening,
actual panel report handling, Worker failure, invalid input, stale results and
causal settlement. These are functional checks, not a win-rate backtest.

Rollback auth-client.js and index.html to commit 1dfeebf98e01cbcfc175ddef0d6f15603c772d84.
Old web source files remain untouched for that purpose. No LINE changes.
