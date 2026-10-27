> 歷史方案：2026-10-27 已依使用者要求退回 PR #59，不再啟用本文件的自適應平滑。現況見 PR59_PF_ONLY.md。

# 僅平滑改善｜clean_signal_responsive_v1

日期：2026-10-26。此版只調整決策 EMA，未重訓、未替換模型。版本 PHYSICS_57D_V14_8_EV_LED_DIRECTION_SMOOTH1。

## 行為

原有最新機率權重約前段45%、中段60%、後段72%，並有既有噪音／不確定性減項。保留這些基礎值；第1–20局完全維持。第21局開始逐步允許增加最新機率權重，到46局完整啟用。

只有當目前經 Clip 的機率離0.5超過0.015，且噪音與Physics不確定性都低，才增加權重。距離達0.04、noise≤0.45、uncertainty≤0.45時，加成最大；噪音≥0.70或不確定性≥0.75時加成為0。最大增加0.10，alpha硬上限0.84。例如後段乾淨明確訊號可由72%提高到82%。中段加成隨進度緩升，並非直接套用最大值。

公式：alpha = min(cap, 原alpha + 0.10 × 階段進度 × 訊號強度 × noise gate × uncertainty gate)。輸出仍為 alpha × 本局機率 + (1−alpha) × 前次平滑機率，再受原Clip範圍限制；不額外偏莊或偏閒、不擴大原始機率。方向切換時同樣只減少舊值拖延，不會因上一局輸贏強制翻向。

缺少前次同靴有效機率時直接使用當前值。修正舊資料null被轉成0的平滑問題。所有條件只使用預測當下資料，沒有讀取本局尚未公布結果。

## 參數與回滾

所有新參數在 cardcount_profile.json 的 adaptive_smoothing，schema_version=1、profile=clean_signal_responsive_v1；enabled=false 即退回原決策EMA（保留null資料修正）。不合法設定不啟用加成。

start_round=21、full_round=46；edge_start=0.015、edge_full=0.04；noise_low=0.45、noise_high=0.70；uncertainty_low=0.45、uncertainty_high=0.75；max_alpha_boost=0.10、max_alpha=0.84。上述為可驗證的保守起始設定，未宣稱為最高勝率參數。改設定後需重新載入；auth-client.js僅更新runtime快取版本。

完整回滾可revert本次merge commit；上一版BBB main為62d84b78bcd01b7465a4ddc02428d1e9644cc3f5。

## 保留範圍

EV選向、佣金與和局退回公式、三階段EV門檻、Volume Guard、2pp顯示門檻、不觀望按鈕、本機校準、PF核心與參數、Macro EMA及方向係數、Frozen Core、213→48拓撲、57D特徵與模型權重、訓練設定、開通碼與30分鐘限制全部保留。Macro方向係數未校準仍為0，此次不調Macro速度以避免無證據增加變動。

## 驗證

node test_adaptive_smoothing.js：驗證前20局不變、高噪音與高不確定性不加速、訊號弱時不加速、乾淨反轉減少拖延、輸出不外插、null略過、跨靴隔離、EV口徑一致及48D／57D未改。

另執行既有EV、補牌、runtime、Macro、profile、校準、按鈕、保存與中後段回歸測試。validation/adaptive_smoothing_smoke.json記錄固定4靴合成回放，只用於機制檢查，不能作為實戰命中率或獲利提升證據。未取得真實獨立hold-out，因此本版沒有模型晉升或重訓的主張。
