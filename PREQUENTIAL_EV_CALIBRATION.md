# 逐局機率校準 → 同口徑 EV｜PR #63 基準

2026-10-27，版本 PHYSICS_57D_V14_8_EV_LED_DIRECTION_PREQUENTIAL_EV1。本次新增逐局校準參與正式EV，保留PR #59平滑、PF更新、模型與畫面。不是逐局完整重訓Physics／XGBoost。

## 每局流程

1. 以尚未校準的最終機率，預先計算原決策與校準候選決策；兩者使用同一套佣金、和局退回、EV／noise／confidence及Volume Guard規則。
2. 使用先前已公布結果的殘差，產生小幅校準候選。預測本局時不讀本局結果，重複推論不學習或寫入結果。
3. 當使用者按B／P／T，先取得本局結果寫入歷史前的預測ticket；只有實際歷史持久保存成功、前綴吻合且尚未學習過，才提交更新。
4. 下一局重新計算。所有啟用閘門只使用之前已揭曉的比較結果。

使用未校準的最終機率作為學習輸入，避免將已修正的機率再重複修正。原始Physics/PF winner-head仍保留供診斷；不把它與最終決策機率混用。

## 啟用閘門與參數

local_calibration新增formal_ev_enabled=true、min_bet_validation=8；其他參數沿用PR #63：min_samples=24、min_validation=16、max_delta=0.005、min_brier_gain=0.00005，以及既有prior_strength、residual_decay與validation_window分段設定。

校準候選須同時滿足：

- 至少24筆已揭曉的非和局學習資料，並有至少16筆後續逐局機率比較。
- 在近期比較視窗內，Brier改善至少0.00005；候選即使沒有改變機率，也納入正式模式的比較，避免只留下曾有變動的舊樣本。
- 原決策及候選各至少8筆非和局下注比較；候選命中率及實現EV/注均不得低於原決策。
- 儲存正常；本局noise gate允許。高噪音、無樣本、缺正式比較紀錄或驗證失敗均維持原機率。

上述是裝置內小樣本、因果的線上驗證閘門，不等同獨立真實hold-out或收益保證。每靴重置，因此一般要到至少約40筆非和局結果後才可能套用；也可能整靴都不啟用。

## 機率與 EV

保留和局機率T，只小幅調整排除和局後的qB，最多±0.005，仍受既有Clip上下界與方向保護限制。B=(1−T)qB，P=(1−T)(1−qB)。EV(B)=(1−莊佣金)B−P；EV(P)=(1−閒佣金)P−B。和局淨收益0。

畫面、EV選向、正式EV Guard及下注資格均使用同一最終機率；不觀望仍僅為顯示測試。EV不是另一個模型，不用對收益公式重訓。只有B/P/T，沒有把模擬點數、牌值或張數當真實監督標籤。

## 隔離與診斷

沿用本機校準日誌與bgs授權儲存隔離。不同牌靴、模型或完整profile不共用狀態；新版fingerprint使舊winner-head校準日誌不會誤套到新final-p口徑。返回、重新整理及重複按鈕有前綴／去重保護；結束分析沿用原清空流程。

快照增加calibration_input_forecast（含事前原／候選動作與費率）及browser_calibration指標，保留原始機率、採用delta、樣本數、Brier增益與下注比較結果，便於追查。T不當成P訓練，但會以退回本金納入EV/注分母。

## 驗證與回滾

node test_prequential_ev.js覆蓋實際啟用、機率／EV／畫面一致、差候選被拒、Brier好但下注表現差也被拒、缺資料、非和局與和局、重複／未寫入結果、重載、跨靴／模型隔離、高noise、undo及48D／57D／EMA不變。

node test_pr59_prediction_baseline.js鎖定PR #59平滑函式及目前畫面雜湊；正式runtime因本次新增校準已不再整檔等同PR #59。既有EV／PF／補牌／Macro／UI／保存測試繼續執行。

因果回放：node tools/top_parameters/replay.js . complete-hands.json report.json --local-calibration。未帶旗標不提交逐局學習，供基線比較。validation/prequential_ev_smoke.json是4靴合成機制測試，不是實戰準度或模型晉升依據。

回滾：把local_calibration.formal_ev_enabled設為false，重新載入，即恢復PR #63的正式機率與EV路徑，校準僅背景診斷；完整還原可revert本次merge commit。上一版main為7376cd8e38c97f34859a0ec6547e5bcd9df143ed。auth-client只更新兩個腳本快取版本，授權與30分鐘設定不變。

本次固定4靴合成回放共274筆，9筆實際套用校準、1筆下注改變，最大delta約0.005。整體命中率相同，EV/注由−0.02072變成−0.02091，Brier由0.250998變成0.251001；均不能作為收益改善證據。線上歷史閘門不保證未來結果。
