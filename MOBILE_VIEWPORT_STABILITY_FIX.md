# Mobile iPhone / in-app browser fixed-viewport repair — 2026-11-02

## Reported bug
On phones, particularly iPhone in-app browsers, the main page repeatedly became microscopic, jumped in size, and displayed the long header/mode text vertically. The previous `viewport-fit.js` observed the visual viewport, full content subtree, and ResizeObserver, then applied `transform:scale(availableHeight/contentHeight)`. Mobile toolbar changes and content updates triggered endless repaint/rescale cycles; forcing the entire long form into one viewport made text unreadable.

## Minimal UI-only fix
- `viewport-fit.js` now monitors **only** `predictionApp.hidden` transitions to lock/unlock the outer page on authorization. No resize handlers, no mutation observer on content, no visualViewport measurements, no height calculations, and no transform scaling.
- `styles.css` uses a stable `100svh` viewport (fallback `100vh`). The **outer document stays fixed**, while the authorized app's content can scroll **internally** when it is taller than the device screen. This is the only reliable way to retain readable buttons and settings on small screens or with large font accessibility settings. Login remains freely scrollable.
- The top header remains sticky *within* the app frame, and mobile mode/status displays in its own constrained row with ellipsis. It will not force the Chinese brand/title into a one-character-per-line column.
- `index.html` updates **only** the stylesheet and viewport helper cache keys. The production `auth-client.js?v=20261102-v2-early-shoe-guard-v1` loader is unchanged.
- Keep existing bankroll/slider, B/P/T entry, controls, history, export, and the removal of the side-bet card.
- Update `test_fixed_view_ui.js` and the early-shoe guard's UI-loader test for the stable viewport.

## Non-goals / guarantees
**No changes** to `quant256_online_runtime.js`, `physics_forecast_first_runtime.js`, `quant256_online_panel.js`, `quant256_bankroll_ui.js`, auth logic, EV gates, or the `96 / 256 / 96` particle sampling configuration. Early-Shoe Guard is unaffected.

The old requirement of absolutely no scrolling is incompatible with preserving all readable content on very short phones. The page **itself** does not scroll. Its contained, user-operated content region can scroll if needed, to ensure no controls become inaccessible.

## Verification
CI checks static asset versions, original control IDs, no scale/resize loop, stable viewport CSS, and login/page scroll-lock transitions. Browser-specific visual confirmation is still recommended using the actual member's iPhone webview. Rollback commit: `4eb0f223484a9613424c27c768ecffbc9f39b438`.
