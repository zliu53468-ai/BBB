## 2026-11-15: trial API 404 recovery and +60% bankroll objective
The public page at bgsv1.xyz may use an old Cloudflare Worker that returns
{error:"not_found"} with HTTP 404 for POST /api/trial/start. The root trial
client now offers a **degraded client-side trial** only for that exact case.

- Fixed local window: 30 minutes from first recorded activation, in localStorage.
- UI explicitly labels this fallback as a browser-only timer.
- It is not secure anti-abuse: deleting storage or changing the browser clock
  can circumvent the limit, and public GitHub Pages JavaScript is readable.
- Paid license verification and its original IndexedDB separation are unchanged.
- CORS failures, 5xx failures, invalid tokens and explicit revocation never
  silently fall back to a new free trial.

For true server-enforced trials, configure repository GitHub Actions secrets
CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID with least-privileged Worker
and D1 permission. The deploy-license-worker workflow applies pending D1
migrations to the EXISTING bbb-license-db, deploys the existing Worker and
checks its /health response (trial_api_version=1). Never paste tokens into
GitHub source code or frontend JS.

The 60% feature is a **profit target**, not a 60% model accuracy threshold.
The user manually enters the bankroll amount; it is not synced with casino
accounts. After the entered current balance reaches 1.6 x initial bankroll,
the client shows target-hit status, hides the reference stake and redirects
to trial-ended.html?reason=profit60 after 1.8 seconds, with the existing
LINE and casino registration links. Losses do not invoke an automatic stop.
