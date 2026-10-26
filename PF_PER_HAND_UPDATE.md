# PF 逐局粒子更新加強｜移除圖片分析

2026-10-27，profile：per_hand_posterior_v2。

依使用者最後確認，本次只加強 PF 的粒子後驗更新，不做 Physics／XGBoost 完整重訓，也不改剛上線的平滑或 EV 選向。

## 逐局學習的實際意義

每次分析從500個八副牌粒子開始，依本靴已公布的B/P/T順序重播。每個粒子模擬完整抽補牌，再用實際B/P/T作likelihood權重更新；有效粒子數過低時進行原有重採樣。預測下一局時不讀取該局尚未公布的結果。新結果輸入後，下一次分析會包含它；不觀望測試仍按既有方式自動更新。

這是粒子貝氏更新，不是梯度訓練，也不會替換正式模型檔。EV是抽補牌後機率扣佣、含和局退回的計算結果，本身不是另一個要重訓的模型。只有B/P/T，無法以真實牌值監督點數、剩餘A–K或張數；模擬牌不當成真實訓練答案。

## 本次參數

cardcount_profile.json / pf：update_profile=per_hand_posterior_v2；likelihood_draws：2→4；fallback_draws：2→4。每次結果更新的匹配樣本更多，但不保證命中率或EV提升。粒子仍500，ESS門檻0.50、tie影響0.30、下一手預測抽樣3/4/5、混合上限及方向保護均保留。抽補牌規則完全不改。

runtime允許likelihood_draws與fallback_draws配置2–8；缺省仍2，可兼容舊版。較高抽樣會增加手機／瀏覽器運算時間。此版尚無真實hold-out收益證據。

## 圖片分析

index.html移除圖片上傳、截圖、裁選、掃描與確認匯入區域及專用樣式；auth-client.js停止載入bbb-screenshot-road.js和bbb-desktop-capture.js。保留原始檔供回滾，不在正式頁面執行。保留莊／閒／和、開始分析、不觀望測試、返回、結束與匯出按鈕。開通碼與30分鐘授權邏輯不改。

## 驗證與回滾

node test_pf_per_hand_update.js：2/2設定對舊版fixture結果逐值相同、抽補牌與forecast函式來源雜湊不變、新設定確實執行4/4、實際B與P造成不同後驗、同歷史重算／undo完全一致、機率歸一化、500粒子與混合上限、圖片入口及script不載入、操作按鈕存在。

原先「PF檔案不可改」整檔雜湊測試，改以獨立抽補牌規則來源與舊版結果fixture鎖定；Core與模型整檔雜湊仍保留。中後段舊測試以相同PF預算比較階段設定，正式方向檢查採已部署的EV候選。

回滾PF參數：將likelihood_draws、fallback_draws都改回2。完整還原圖片入口與PF可revert本次merge commit。前版main為0b8d4d69101ef17f691c7cb1ad6d4006cf3a5ee7。更新設定後重新載入網站。
