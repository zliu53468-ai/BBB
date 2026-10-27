# 結束分析：清除後自動重新載入

本次只修正結束分析流程，不修改預測模型、機率、EV、開通碼驗證或 30 分鐘設定。

- 本機 IndexedDB 清除成功後，禁止繼續分析與重複清除。
- 清除期間停用操作按鈕；跨裝置同步最多等候 4 秒，逾時保留既有待同步標記，下次驗證重試。
- 使用帶 analysis_reset 時戳的新頁面導覽，保留原有網址參數與錨點，讓本次記憶體運算狀態隨頁面卸載。
- 本機清除失敗不導覽，恢復操作並顯示錯誤；其他裝置仍需再次驗證才能同步。
- 這是狀態生命週期修正，不能據此宣稱命中率提高；不保證強制清除瀏覽器所有靜態資源快取。

驗證：node test_end_analysis_reload.js（正常、斷線、同步卡住、本機刪除失敗、重複點擊）；node test_license_storage.mjs；node test_experimental_no_skip.js；node test_pr56_restore.js；node test_bbb_button_only.js；node test_bbb_result_persistence.js；node license-worker/test-license.mjs。

auth_body_sha256 因本次經授權的清除流程修正更新，驗證與授權函式未修改。回滾可 revert 本次提交。
