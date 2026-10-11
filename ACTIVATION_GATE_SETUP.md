# BBB 單裝置開通碼系統

## 版本與預測核心保證

- 本次新增開通介面前，已把 `main` 所有檔案**精確還原**至 `10ce55b33de6a5398c199ca30bd468a7f831285d` 的 Git tree（安全新提交，不 force-push）。
- 還原前已備份：`backup-before-10ce55b-activation-gate-20261011`。
- 之後只新增授權頁面、工作流程和 Worker。沒有修改 `app256forward.js`、`app256continuation.js`、`final_probability_runtime.js`、`final_probability_model.json`、`particle_filter_runtime.js` 或 Physics 權重。
- 網頁分析流程仍使用原 10ce55b 的 Frozen Core、Direct Physics 213→48D、Particle500、Physical EV、動態 Clip/EMA、57D XGBoost no-flip、Final EV Guard、Volume Guard。
- **本次沒有重啟零觀望模式。**

## 功能

1. 使用者打開 `index.html` 先看到開通碼登入頁，預測主面板隱藏。
2. 由管理者開啟 `activation-admin.html`，輸入伺服器管理金鑰，建立 24 字元高熵開通碼（介面自動分組）。每組碼只顯示**一次**。
3. 使用者輸入碼，瀏覽器在 IndexedDB 產生並保存 P-256 不可匯出的私鑰。
4. 瀏覽器先向伺服器取得 120 秒有效的隨機挑戰，再用私鑰簽章。伺服器確認該碼有效並檢驗簽章，**首次把公開金鑰原子綁定至此碼**。
5. 下次相同瀏覽器/網站儲存空間可讀取碼與私鑰，自動完成新挑戰認證；每五分鐘及回到分頁時再次驗證。
6. 其他裝置即使知道相同開通碼，也無法提供原裝置的私鑰簽章，會收到 `code_bound_to_another_device`（HTTP 409）。
7. 管理者可以查詢綁定狀態、**重置**綁定（會讓舊裝置失效且允許持碼者重新競爭首次綁定）及**停用**碼。
8. Worker 使用 D1 保存 HMAC-SHA256 的開通碼索引，不存明文。管理金鑰與雜湊祕密存在 Cloudflare Secret，不放前端或 GitHub。

## 重要安全界線

- 這是綁定一個 **瀏覽器資料區／裝置私鑰**，不是無法複製的硬體序號。一般使用者更換瀏覽器、清除網站資料或重新安裝瀏覽器，可能需要管理者重置。
- 非匯出 WebCrypto 私鑰可抑制一般的跨裝置複製，但不能保證抵抗受感染的裝置、瀏覽器漏洞或完整用戶設定檔複製。
- **GitHub Pages 是公開靜態網站，不能從根本上阻止技術使用者直接讀取公開 JS 和模型檔**。這個登入閘門能保護一般網頁流程，不能保證底層預測演算法的伺服器授權。若必須防止程式碼被直接取得，需要把推理與模型檔搬到經過授權的私有伺服器，不再以公開 Pages 交付。
- 預設 `auth-config.js` 的 `BBB_AUTH_API_BASE=""`，因此 **正式驗證服務沒有部署前，網站會鎖住分析頁，不會用假密碼放行**。
- **請勿**把 Cloudflare API token、`ADMIN_TOKEN`、`LICENSE_PEPPER`、真實用戶開通碼寫入 GitHub。

## Cloudflare Worker + D1 部署（管理者一次完成）

需要 Cloudflare 帳號、Node.js 20+、可使用 Wrangler CLI 的登入狀態。官方文件：
- [Cloudflare D1 create / migrate](https://developers.cloudflare.com/d1/wrangler-commands/)
- [Cloudflare D1 migrations](https://developers.cloudflare.com/d1/reference/migrations/)

```bash
git clone https://github.com/zliu53468-ai/BBB.git
cd BBB/license-worker
npx wrangler login
npx wrangler d1 create bbb-license-db
```

上一步取得 UUID 後，將 `license-worker/wrangler.toml` 的 `database_id` 中
`REPLACE_WITH_YOUR_D1_DATABASE_UUID` 換成實際 UUID。

```bash
# 仍在 license-worker 目錄
npx wrangler d1 migrations apply bbb-license-db --remote

# 請使用彼此不同的高熵隨機字串作為祕密；下列命令會互動要求貼入
npx wrangler secret put ADMIN_TOKEN
npx wrangler secret put LICENSE_PEPPER

npx wrangler deploy
```

部署成功後會有 HTTPS Worker 網址，例如
`https://bbb-license-gateway.<account>.workers.dev`。

將倉庫根目錄 `auth-config.js` 更新為：

```js
window.BBB_AUTH_API_BASE = "https://bbb-license-gateway.<account>.workers.dev";
```

**這是公開 Worker 網址，不是祕密。** 保存並推送 `main`，等待 GitHub Pages 發布完成。Worker 的 `ALLOWED_ORIGINS` 預設為 `https://bgsv1.xyz,https://www.bgsv1.xyz`。如另在 github.io 測試，要加入該網域的精確 origin。

### 產生開通碼

打開 `https://bgsv1.xyz/activation-admin.html`，輸入你剛才存在 Cloudflare 的 `ADMIN_TOKEN`，點「產生開通碼」。切勿向會員提供管理金鑰。各組開通碼只會在產生時顯示一次；請安全地發給會員。

開通碼最多可一次建立 30 組，可以設定標籤與到期日期。停用會立刻阻止下次認證；已開啟的面板通常會在回到分頁或下一個 5 分鐘週期鎖定。

## 測試

```bash
# 此測試使用假的內存 D1 與測試用 secrets，不需 Cloudflare
node license-worker/test-license.mjs
node --check license-worker/src/index.js
node --check auth-client.js
node --check activation-admin.js

# BBB 原始核心回歸測試
node test_macro_runtime.js
node test_final_probability_runtime.js
```

GitHub Actions：`.github/workflows/license-checks.yml` 會檢查登入 JS、Worker 及單裝置／重播／重置／停用流程。原本 `runtime-checks.yml` 繼續測試預測系統。

## 回滾

在還原前的 `10ce55b` 版本基準下，本次新增的授權閘門可以獨立撤除，不能用 `git reset --hard` 破壞主分支歷史。建議先建立備份、檢查此次改動提交，再使用 `git revert` 或建立以基準 tree 為內容的新回滾提交，經 GitHub Actions 驗證後發布。

由於後續其他人的提交可能發生，回滾時務必先 `git log --oneline --graph` 核對提交範圍。Worker 若已發出去，另至 Cloudflare 取消 Worker 路由，避免留有未使用 API。
