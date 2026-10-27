# 恢復 PR #56 預測｜精簡 B/P/T 版面

來源：PR #56 合併版本 `e11b52424d0641aeaa16437a15e744808848dfba`。

- `final_probability_runtime.js`、`particle_filter_runtime.js`、`browser_calibration.js`、`cardcount_profile.json` 完整恢復來源版本，沒有混入後續點數 PF、第一層 EV／專用平滑及額外選擇性閘門。
- Frozen Core、Physics／XGB 模型權重、Macro EMA 未修改，也沒有重訓模型。
- 保留現行登入、授權、裝置綁定、資料隔離及 30 分鐘限制；`auth-client.js` 僅更改預測腳本版本與移除點數輸入模組載入。登入表單不變。
- 莊／閒／和按鈕直接顯示，移除點數、總張數、點數模式切換及說明。不恢復舊截圖掃描介面。
- 保留固定 2pp 顯示門檻、不觀望測試、返回、結束與匯出按鈕。
- 圖片及動畫維持 fixed/inset:0 全頁背景，不占版面、不攔截點擊。縮小卡片留白，手機按鈕最小觸控高度仍為 44px。

## 驗證

`validation/pr56_restore_manifest.json` 固定來源、四個預測檔的 SHA256、Frozen Core／模型／Macro／授權設定及授權程式本體雜湊。

```sh
node test_pr56_restore.js
node test_bbb_button_only.js
node test_bbb_result_persistence.js
node test_browser_calibration.js
node test_final_probability_runtime.js
node test_macro_runtime.js
node test_cardcount_profile.js
node test_middle_late_upgrade.js
node test_physics_forecast_first.js
```

CI 改回 PR #56 對應回歸測試；點數專用工作改為 B/P/T 回復檢查。後續版本留下的點數／第一層 EV 專用測試及歷史研究文件仍保留供參照，但不代表目前功能，也不在此版正式測試流程中執行。

Chrome 實機引擎本機檢查 320、390、1365px 寬度：無水平溢出，背景覆蓋整個 viewport，B/P/T、分析、不觀望與撤銷操作成功，無 JavaScript 例外。視覺測試在本機隔離環境手動載入預測腳本，未使用或修改真實授權。

## 回滾

使用 `git revert <此次合併提交>` 可恢復修改前 main。若單獨恢復 PR #56 預測，四個來源檔及完整 SHA256 見 manifest；不要只替換單一 runtime 而混用後續點數設定。
