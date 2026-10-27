# 預測退回 PR #59｜僅保留 PF 逐局更新加強

2026-10-27。依使用者最新指定，final_probability_runtime.js 逐位元還原 PR #59 合併後版本62d84b78bcd01b7465a4ddc02428d1e9644cc3f5；版本標籤回到 PHYSICS_57D_V14_8_EV_LED_DIRECTION。

移除PR #60新增的adaptive_smoothing設定及額外alpha加成，包含該次平滑資料處理變更，回到PR #59原始行為；仍保留PR #59的Macro、原決策EMA、補牌後含佣金／和局退回的EV選向與原風控。

cardcount_profile.json與PR #59相比，僅PF多出update_profile=per_hand_posterior_v2、likelihood_draws=4、fallback_draws=4。500粒子、補牌規則與混合上限保留。這是依實際B/P/T逐局更新粒子後驗，不是重訓Physics／XGBoost權重，也不代表真實牌值已知。

index.html、styles.css逐位元保持目前PR #62的畫面：全頁動態背景、圖片分析移除及原按鈕均保留。開通驗證與30分鐘限制不變；auth-client.js僅更新預測runtime的快取版本。

node test_pr59_prediction_baseline.js檢查runtime與PR #59雜湊一致、profile只差PF三個設定、畫面兩檔雜湊不變。另執行PF更新、EV選向、抽補牌、runtime、Macro、profile、中後段、按鈕及保存回歸。

歷史ADAPTIVE_SMOOTHING.md與合成報告保留供追蹤，已非現行邏輯。若要恢復此次變更前的預測，可revert本次merge commit；若只要降低PF計算量，可把兩個draws設定改回2。
