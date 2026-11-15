# BBB 30-minute trial and SEO rollout

Source branch: \`feat/30min-trial-seo-dg-flow\`. Paid-license authorization, engine,
V5.1 thresholds, Soft Shoe worker and the B/P/T UI computation are untouched.

## One-time configuration

Edit root \`site-links.js\`, replacing each empty string with the **verified**
HTTPS URL:
- \`dgTrialUrl\`: DG's actual playable/demo page; click opens \`target="_blank"\`
  and leaves the BBB analysis page in the current tab.
- \`casinoRegisterUrl\`: the approved casino signup or promotion information page.
- \`lineContactUrl\`: official administrator LINE contact, e.g. a verified https link.

Until configured, the corresponding buttons are visible but **disabled** rather than
sending users to fabricated destinations. Never promise a bonus without published terms.

## Deploy order (important)

1. Apply Cloudflare D1 migration \`license-worker/migrations/0002_trial_sessions.sql\`
   to the existing **production** D1 database.
2. Deploy \`license-worker/src/index.js\` to the existing Cloudflare Worker with
   the current \`wrangler.toml\` and pre-existing secrets; **never** expose
   \`LICENSE_PEPPER\` or \`ADMIN_TOKEN\` in GitHub.
3. Set the outbound URLs above.
4. Merge/test the frontend branch into \`main\` so GitHub Pages publishes it.
   The existing workflow will deploy the root static website.

Suggested commands from \`license-worker/\` with Wrangler authentication:
\`\`\`sh
npx wrangler d1 migrations apply bbb-license-db --remote
npx wrangler deploy
\`\`\`
Verify API availability before publishing the new free-trial frontend:
- \`POST /api/trial/start\` with \`{"device_id":"<UUID>"}\` returns a random
  token, the server clock and a deadline 30 minutes later.
- \`POST /api/trial/status\` with that token gives current server time and expiry.
- Retrying start with the same device ID cannot reset the deadline.
- A paid license still uses the unchanged signed challenge/verify flow.

## Trial behavior and limits

- Free trial users enter through the public front page, with a full V5.1 panel.
- The demo is **30 minutes from first activation**, not 30 minutes of open-tab time.
- A server timestamp and periodic verification control the time; refresh does not
  renew a previously issued trial. At expiry, the BBB page redirects to
  \`trial-ended.html\`. The external DG tab cannot be closed or redirected by BBB.
- Browser storage contains only the anonymous token and device ID for resumption.
  The user must click to open a DG tab: popup blockers forbid guaranteed automatic
  popup/independent window creation.
- This is a **trial UX gate, not a secure licensing or anti-cheating system**.
  A browser clearing local storage may receive a new anonymous device ID.
  GitHub Pages serves JavaScript publicly; those who can access the model files
  may bypass front-end restrictions. Genuine hard enforcement requires account
  identity and server-side controlled inference.
- Anonymous trial data stays outside licensed IndexedDB stores. Paid licenses
  continue to bind device keys and keep their original functionality.

## Search metadata

The new public index contains crawlable Mandarin product descriptions,
FAQ questions on baccarat analysis vs slot games (including 賽特), a canonical
URL, Open Graph, SoftwareApplication structured data, robots.txt and sitemap.xml.
It does **not** claim guaranteed win rates or misrepresent a slot model.
Google indexing and ranking are not guaranteed. Use Search Console to submit the
sitemap and monitor impressions and indexing.

## Quick smoke checks

- Existing valid license activates and retains existing history as before.
- Invalid code still blocks the paid route.
- Free trial starts, displays countdown and the existing B/P/T panels and big road.
- Clicking "DG 官方試玩" opens a **new tab**, BBB remains.
- Reload while trial active resumes same deadline; expiry locks and redirects.
- Reload after expiry does not grant a new 30 minutes with existing browser ID.
- All off-site links are disabled until actually configured, and only accept HTTPS.
- Test mobile layout, new-visitor Google indexing, and worker/API CORS.
