# BBB 回退 V3 Micro-Fusion／平衡觀望門檻（2026-11-01）

## 本次範圍

依照使用者要求，將預測核心從 V4 Card Marginal 回退至先前的 V3 Micro-Fusion，V3 參考 commit：6548bd311a2c91f062043be47785b964397b3b3b。

直接還原 V3 版本檔案：
- quant256_online_runtime.js — 以 V3 原始條件抽樣／成對消耗統計為主，不使用 V4 的 conditional marginal SMC 重採樣。
- physics_forecast_first_runtime.js — 以 V3 原始物理 Monte Carlo、靴尾研究倍率、Beta-Binomial 弱路單共振及衝突提示為主，不使用 V4 的 0~9 點邊際 EV 矩陣。
- quant256_online_panel.js — 回到 V3 顯示與分析欄位；大字只輸出「莊」「閒」或「觀望」，移除「（研究）」尾綴。

## 唯一門檻微調

基礎研究訊號門檻 BASE_EV_GATE 由 0.005 微幅提高為 **0.008（0.8% 的 EV gap 偏移訊號）**；連勝 OVERDRIVE_GATE=0、錯局單次 ONE_HAND_CUSHION=0.035 保持原樣。另保留 V3 首次物理與路單衝突的單次短暫觀望。門檻越高一般局訊號會較少，但不保證提高命中率；連勝時的 0 門檻仍以原 V3 行為運作。

## 主畫面及證據性質

若研究訊號通過，正式大字只顯示莊或閒；若未通過，顯示「觀望」。下方仍保留 SIMULATION ONLY、未確認正 EV 等資訊，避免 B/P/T 的虛擬算牌被誤認成已知殘牌。保留所有登入、裝置授權、原始 B/P/T 按鈕、返回上一局和歷史存檔，相容 bgs256d_128plus128_frozen_direct_tech_panel_v1。

## 測試與部署

- test_quant256_online.js — V3 原計算、歷史回放、面板僅顯示簡短方向、正確 SKIP、舊存檔及返回功能。
- test_physics_forecast_first.js — V3 正統補牌、虛擬與實測殘牌區分、路單弱融合、靴尾訊號、0.008 / 0 / 0.035 狀態機。
- GitHub Actions：.github/workflows/quant256-online.yml。

**不能由此保證勝率**：只輸入 B/P/T 無法還原真實 A~K 耗牌，調高門檻只是減少部分低差距研究訊號，是否真的減少連敗須以逐靴留出資料驗證。