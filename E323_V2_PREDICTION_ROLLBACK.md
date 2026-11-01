# 預測邏輯還原至 e323a243（Physics Forecast First V2）

還原來源：`e323a243e814b730828073c751cf170689a5c730`

## 還原範圍

- `quant256_online_runtime.js`：完全採用來源 commit 的版本 `BBB_QUANT256_ONLINE_V3_HYPER_MICRO`；原始 BlindPhysicsEstimator 虛擬殘牌條件抽樣、LinUCB、36 個活躍特徵恢復為指定版。
- `physics_forecast_first_runtime.js`：完全採用來源 commit 的 `BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE`；移除後來新增的 Micro-Fusion V3、靴進程非線性縮放、Beta 路單融合、一次性路單衝突護欄與前次 PR #110 的虛擬扣牌解耦，改回指定版原貌。
- V2 基礎門檻：`0.005`；上一局研究方向正確後：`0`；單次錯誤緩衝：`0.035`。維持指定版原本的物理 5×／牌路 0.1× 特徵縮放。
- `quant256_online_panel.js`：依 e323a243 的預測報告欄位重接，設定 `threshold:.005`；不讀取 V3 才有的 `fusion`、`shoe_phase`、`pairwise_depletion` 欄位；保留目前大字「莊／閒／觀望」和原始按鈕與授權流程。
- `quant256_bankroll_ui.js`：只修正標示為 V2，保留本金選項、百分比滑桿、localStorage 儲存及觀望 NT$0 的功能。曝險試算仍然不影響預測。
- `index.html`：僅將舊的「V3 訊號狀態」改成版本中立的「模型訊號狀態」，並更新 JS 快取版本；其餘標記與樣式保持。
- `auth-client.js`：僅更新預測腳本 cache key，確保用戶不會混載不同版本的 JS；授權 API 與裝置綁定行為不變。
- `test_quant256_online.js`、`test_physics_forecast_first.js`：還原來源版本對應測試，並修正大字「觀望」UI 斷言。
- `test_v3_bankroll_ui.js`：改為驗證 V2 引擎與 0.005 門檻，持續驗證本金和方向互不干涉。

## 特別注意

這次是**指定版本回退**，而不是將 V2 的原始模型與後續修復混合。來源版使用 B/P/T 結果條件化模擬扣牌及歷史標籤計算隨機種子；因此 **PR #110 曾修復的 result-to-physical-EV 反饋路徑會回來**。僅有 B/P/T 無法確認實際殘牌 A～K 組成，模擬 EV 不保證真正投注優勢，也不能保證下一局命中。

## 執行驗證

```bash
node --check quant256_online_runtime.js
node --check physics_forecast_first_runtime.js
node --check quant256_online_panel.js
node test_quant256_online.js
node test_physics_forecast_first.js
node test_v3_bankroll_ui.js
```

正式 GitHub Pages 仍透過會員授權後載入主面板；這次不清除已保存的牌路或本金配置。