# 同靴雙向誤差校正 V1

取代 PR91 固定相對基準輔助，並非 PR92 最近比例換邊。

只在雙核心 live 且 relative_baseline_v1.enabled=true 啟用。紀錄本頁事前預測、結算後的非和局誤差 y−p，至少 25 個有效樣本，delta=clamp(sum(y−p)/(100+n),−0.0025,+0.0025) × noise_gate。noise_gate 從 noise=0.55 至 0.85 線性由1降至0；缺 noise 不校正。這是排除和局後 P(B) 的 ±0.25 個百分點上限，P(P)反向調整；和局機率不變。機率可跨0.5，沒有固定B2P或強制交替。

畫面以校正後莊閒機率差1pp為門檻（50.5%／49.5%），不足則觀望；不觀望實驗按鈕仍保留，只控制測試顯示，無權放寬正式下注資格。正式 finalPB／EV／B/P/Skip／Volume Guard 使用原值不變。

為避免重複誤差校正，新模組啟用時不疊加既有 browser_calibration 的修正；既有紀錄與閘門程式不改動，關閉後恢復使用。原本機率映射與參考平滑保留。新模組以修正前、平滑後機率作事前票據，實際結果寫入成功後才學習，不以輸入的歷史結果反造預測。普通／不觀望模型紀錄分開。和局不納入有效樣本。

新校正紀錄只在本頁記憶體，重新整理或結束分析即清空；換靴清空，撤銷按前綴截斷，不沿用其他桌或其他裝置。從已匯入歷史直接開始時，需再累積25個本頁有效事前預測。單靴50–70局可能很晚才啟動，這是保守設計。

匯出保留 raw_next_hand_forecast 與 relative_baseline 的 base_pb、corrected_pb、delta、directional_samples、noise_gate、reason、display_only、formal_decision_changed=false。畫面校正機率不是已證實的真實勝率，亦不是正式正EV的新證據。尚未完成真實獨立hold-out，無命中率提升保證。

參數：min_samples=25、prior_strength=100、max_delta=0.0025、noise_start=0.55、noise_stop=0.85、display_gap=0.01。max_delta 硬上限0.0025，min_samples最低25。

回滾：relative_baseline_v1.enabled=false 關閉新修正，display_gap=0.005 並同步恢復畫面選項可回原顯示門檻；完整PR91則revert本次合併提交。模型權重、開通碼、30分鐘限制、Frozen Core與正式風控均未修改。

驗證：node vnext/test_relative_baseline.cjs；node vnext/test_live_runtime.cjs；node test_road_primary_runtime.js。
