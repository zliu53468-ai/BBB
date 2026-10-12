# BBB：開通碼隔離與「結束分析」全資料清除

> 2026-10-12 儲存／登入層修改。**Frozen Core、213D→48D Physics、Particle500、57D XGBoost、Dynamic Clip、EMA、Final EV Guard、Volume Guard、V15 診斷與決策門檻均未修改。**

## 1. 資料隔離

從 Cloudflare Worker `/api/verify` 成功取得不可猜測的 `license_id` 後，網頁先載入 `bbb-license-storage.js` 並開啟 `bbb-analysis-isolated-v1` IndexedDB，再載入原本的 Frozen Core 等模型 JS。Storage.prototype 上**僅 BBB 網頁內**的 `localStorage.getItem/setItem/removeItem` 針對 `bgs*` 分析鍵被代理到每個 `license_id` 獨立的 IndexedDB 作用域，其他瀏覽器資料維持原本行為。同步讀寫介面保持相容，Frozen Core 原始碼沒有更動。

- 會員 A、B 就算共用同一瀏覽器，牌路、歷史快照、Physical EV/EMA/Particle 診斷、牌靴狀態也不互相讀取。
- 巨量 `bgs_xgb_final_training_v5`、`bgs_macro_observation_shoes_v1` 不再寫入容量較小的 localStorage，而使用 IndexedDB。
- **舊版未標記會員的 localStorage 資料不能證明屬於哪個開通碼，絕對不自動匯入。** 新版本第一次登入會從該會員獨立空白資料區開始；原始舊資料暫時保留於舊 localStorage 中、且不會被預測模型讀取，直至該會員按「結束分析」才一併刪除。若需要保存，升級前請先匯出舊版 JSON。
- 原裝置的 `bbb-license-device-v1` 中的裝置金鑰、已使用開通碼仍維持原狀，不會因「結束分析」而重新綁定或洩露秘密。

## 2. 結束分析

按「結束分析」後 Frozen Core 原本的零局數初始化照常執行，接著 Storage adapter：

1. 等待所有該會員待寫入的 IndexedDB 寫入交易結束。
2. 使用原子 IndexedDB 交易 **刪掉同一 `license_id` 作用域的所有 bgs* 紀錄**（牌路、訓練快照、觀測、pending、EMA、巨觀和舊鞋檔）並更新 epoch。
3. 刪除原本未綁開通碼、無法可靠指派會員的 `bgs*` localStorage 遺留資料，釋放容量。
4. 透過 BroadcastChannel 和 Storage 事件通知同瀏覽器的其他分析分頁重載；使用者本頁也重載、保持裝置授權。
5. 嘗試用裝置私鑰所簽署的一次性 challenge，向新 Worker 的 `POST /api/session/end` 送出結束事件，令 D1 `licenses.generation` 遞增。其他可成功驗證的網頁／裝置在下次開啟或驗證時發現 generation 改變，即刪除本地該會員的舊資料。**離線裝置必須等待它再次連線；網站不可能直接操作完全離線的裝置。**

若本地清除交易失敗，頁面必須顯示失敗訊息而非聲稱清除成功。若遠端 API 未部署或當時無網路，頁面仍清掉本機資料，重新登入後顯示「跨裝置同步尚未完成」警告；請稍後在 Cloudflare Worker 更新版本並重試。

## 3. 必須手動更新 Cloudflare Worker

GitHub Pages 已自動部署，但 Cloudflare 上的 `bbb-license-gateway` 是曾由管理員 **在 Web UI 貼上程式碼手動 Deploy** 的獨立 Worker。GitHub commit **不會**自動改到 Cloudflare！跨裝置清除需再進行以下一次性動作：

1. 開啟 [GitHub Worker 程式](https://github.com/zliu53468-ai/BBB/blob/main/license-worker/src/index.js) → `Raw`，複製整份最新程式碼。
2. 登入 Cloudflare → Workers & Pages → **bbb-license-gateway** → Edit code → **Latest**，全選 `worker.js`、貼上最新程式 → Deploy。
3. **保留** D1 Binding `DB`、Secret `ADMIN_TOKEN`、Secret `LICENSE_PEPPER`，不必再建立資料表。既有 D1 的 `licenses` 表本來就有 `generation INTEGER NOT NULL DEFAULT 0`。
4. 開 `https://bbb-license-gateway.zliu53468.workers.dev/health`，確認 `configured:true`；健康檢查只驗證基本配置，不能證明 POST sync endpoint 成功。再用一組測試開通碼走「開始分析 → 報結果 → 結束分析 → 重整後 0 局」，並確認頁面不顯示遠端同步警告。

**若不更新 Worker：** 前端的每會員隔離、IndexedDB 和本機／同瀏覽器分頁清除可使用；**跨裝置**同步則尚未生效。不要宣稱跨裝置即時刪除。

## 4. 回歸測試

```sh
node --check bbb-license-storage.js
node --check auth-client.js
node test_license_storage.mjs
node license-worker/test-license.mjs
node test_bbb_result_persistence.js
node test_final_probability_runtime.js
```

GitHub Actions：`BBB License Gate Checks` 和 `Physics Primary Runtime Checks`。

安全要求：沒有任何會員開通碼、管理員 Token、裝置私鑰或 Pepper 被寫進預測倉庫程式／模型。伺服器 remote endpoint 只接受**目前綁定裝置私鑰**的一次性簽名，管理密碼不會傳到前端。

## 5. 注意事項

- 伺服器不存歷史牌路；它只存開通碼、綁定公鑰和代數。資料實際在每台裝置本地的 IndexedDB，遠端 generation 是下一次線上驗證時通知刪除的機制，而非強制對離線硬碟立刻刪檔。
- 若需要「多台不同裝置共享同一碼」，需另外修改本來嚴格一機綁定的授權政策；本升級不修改一機限制。
- 非同步 IndexedDB 若瀏覽器意外中斷，極短時間內尚未提交的最後一筆資料可能尚未持久化。關閉前應等待回報完成。
- GitHub Pages 公開模型原始碼未因這次資料隔離變成不可讀，隔離指的是瀏覽器應用資料，不是伺服器上完整模型的存取控制。
