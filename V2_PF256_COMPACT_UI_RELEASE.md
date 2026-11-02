# V2 PF256 / L384 / M160 — compact panel release note

## Existing sampling precision (unchanged)
The production `main` already shipped the requested sampling-only upgrade in PR #118, merge commit `d5d2e10760d43431404196356db61bdc625a8982`.

| Setting | Existing production value |
| --- | ---: |
| BlindPhysicsEstimator particles (default and hard cap) | 256 |
| Outcome-conditional legal-hand likelihood proposals per particle | 384 |
| Physics EV Monte Carlo simulations per particle | 160 |
| Nominal likelihood proposals per result | 256 × 384 = 98,304 |
| Nominal EV simulation deals per forecast | 256 × 160 = 40,960 |
| ESS systematic resampling threshold | 0.5 × N = 128 |

No sampling implementation was changed again: core formulas, B/P/T-only evidence limitations, physical EV, gates, LinUCB, Early-Shoe Guard, observed-card rules, RNG, authentication, member storage and end-analysis reset remain as in PR #118.

## Separate requested UI-only changes
- `index.html`: Remove the entire visible bankroll/stake control card; keep the B/P/T input, Start, Back, End, logout/license flow, and exports intact. Bump only the modified CSS asset cache key.
- `styles.css`: Slightly reduce the desktop maximum panel width (760→700px), mobile horizontal inset (14→20px total), and the mobile forecast orb (90→86px); retain fixed outer viewport, independently scrollable content, and touch-accessible controls.
- `test_fixed_view_ui.js`: Check that removed stake controls are absent, validate the compact layout, updated CSS cache key, and existing prediction core URL.

The previously shipped V2 module cache keys `20261103-pf256-l384-m160-v1` are intentionally unchanged, because their code is already current. Analysis End already navigates to a cache-busting `analysis_reset` URL after durable local data deletion.

**Limit:** Only B/P/T results do not reveal the real shoe's card ranks; additional Monte Carlo samples cannot guarantee a higher real baccarat hit rate.
