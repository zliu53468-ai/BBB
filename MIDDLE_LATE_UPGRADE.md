# 中後段升級｜前段鎖定｜算牌優先與逐局校準

本次以 main 的 `60e449c6371ab483677aecf87bb45aa60031fab0` 為基準，套用 `cardcount_middle_late_v2`。這是有界的參數與資料取樣升級，沒有更換模型拓撲或正式權重，亦不能宣稱是已驗證的最佳參數。

## 前 1～20 局保留

前段保留原本的決策 EMA、Macro 短長期 EMA、PF 比重與 3 次下一手抽樣，連原版第 18～20 局的交界插值也保留。原版設定存於 `validation/cardcount_moderate_v1_baseline.json`，可直接回滾與做對照。

60 組實際模型比較（3 種歷史 × 第 1～20 局）確認目前權重下，前段畫面莊閒機率差異為 0、正式方向一致。另逐局比對 noise／uncertainty 網格下的 α。這是回歸檢查，不是命中率、EV 或所有可能牌局的成效證明。

## 中後段參數

中段為下一局第 21～45 局，後段第 46～70 局，和局也算局數。表格列的是平台值；第 20～24、43～49 局漸變，PF 抽樣數在第 20～24、45～50 局漸變後四捨五入。

| 參數 | 原中段 → 新中段 | 原後段 → 新後段 |
|---|---:|---:|
| 決策 EMA 最新機率權重 | 55% → 60% | 65% → 72% |
| Macro 短期 α | 0.30 → 0.33 | 0.32 → 0.35 |
| Macro 長期 α | 0.10 → 0.11 | 0.11 → 0.12 |
| 每粒子下一手抽樣次數 | 3 → 4 | 3 → 5 |
| PF 混合上限 | 14% → 15% | 16% → 16% |
| 本機校準 prior_strength | 100 → 90 | 100 → 80 |
| 本機方向殘差保留係數 | 1 → 0.98 | 1 → 0.96 |
| 本機近期提案比較視窗 | 32 → 24 | 32 → 20 |

噪音與物理不確定性仍會壓低決策 α，硬限維持 35%～75%。Macro 未校準的方向係數仍為 0；加速狀態更新不代表擅自讓牌路改變方向。

PF 仍每次獨立重建 500 粒子、重放當下完整歷史。同向、ESS 與後驗不確定性閘門保留，16% 硬上限不增加，不允許 PF 單獨翻轉 Physics 方向。多抽樣是增加計算量，不是擁有真實未提供牌值，亦未證明實際命中率提升。

## 本機逐局校準的近期權重

本機校準仍只作用於畫面與不觀望測試，不改正式下注資格、EV、Volume Guard、48D／57D 特徵或模型檔。

校準依每筆事件當時的局數採用保留係數：`residual = decay * residual + (actual_B - raw_pB)`；有效樣本量以相同 decay 遞減。提案修正使用 `residual / (prior_strength + effective_n)`，不是把加權殘差除以未加權樣本數。前 20 局 decay=1，與原本累計式完全相同；中後段逐漸偏重近期誤差，不使用長龍／單跳硬規則。

仍需至少 24 個有效莊閒結果，再有至少 16 個事前狀態所算出的有效提案，最早約 40 個有效二元結果後才可能套用。近期逐局 Brier 必須優於原始機率，最低差值 0.00005。上限維持 0.005（0.5 個百分點），不翻方向、不改和局機率，高 uncertainty 或儲存失敗回原機率。視窗縮短沒有降低最少驗證數，也不構成獨立 holdout 或成效保證。

新參數逐事件因果回放，先評分既有狀態的提案，再更新該事件結果；不以目前答案回頭修改同一局預測。返回／更正、重載、各瀏覽器與開通碼隔離、結束清除等行為保留。設定版本變更後，本機校準會重新累積，避免把舊設定下的評分當作新設定已通過。

## 中後段完整重訓候選

兩個 trainer 新增 `--training-profile middle_late_focus`。舊 `production` 與 `aggressive_cardcount` 介面保留，後者的階段權重仍讀原 `training_weight`；新 profile 才讀 `focus_training_weight`。

- Physics：213→48 拓撲、所有任務 loss 權重維持既有 aggressive profile。先分完整牌靴 train／calibration／holdout，僅在 train 內額外有放回抽取約 10% 的 21～45 局、20% 的 46～70 局樣本；不額外複製前 20 局，不改 calibration／holdout。取樣使用固定種子，可重現，metadata 記錄取樣筆數。
- XGB：新階段比重平台為 1.10／1.25／1.35（正規化前），前 20 局比重及交界值保留。沿用物理 uncertainty、actionability、類別平衡及 0.35～1.50 限制；規律額外權重仍為 0%。全資料正規化後，前段的絕對樣本權重可能隨資料分布改變；未來全模型重訓也不能保證前段輸出完全不變。
- 搜尋保留既有正則化組，再為新 profile 增加 depth 3／4、800／750 trees、learning_rate 0.018／0.015、min_child_weight 12／16、reg_alpha 0.35／0.50、reg_lambda 14／18 兩組。沒有無限制放大機率。
- chronological whole-shoe 切分、P=0／B=1 直接分類、校準、原品質門檻與候選檔名保護不變。報告新增 1～20／21～45／46～70 區間的 Brier、wager、hit、EV、Skip；從完整時間線算好的決策結果切片，不在分段時重置 Guard。

```sh
python physics_feature_extractor.py train --training-profile middle_late_focus --shoes 6000 --validation-fraction 0.20 --calibration-fraction 0.10 --augment-ratio 0.25 --output physics_middle_late_candidate.joblib --browser-output physics_middle_late_candidate.json
python xgb_final_probability.py train --training-profile middle_late_focus --input complete_shoe_rows.json --physics-model physics_middle_late_candidate.joblib --output final_probability_middle_late_candidate.json --report-output middle_late_candidate_report.json
```

**本次沒有執行完整模型重訓，也沒有替換正式權重。** 瀏覽器當下會執行的是新中後段平滑／PF 參數與本機校準。完整模型候選需有資料與運算環境，並與目前 production＋PF＋Guard 在同一份獨立牌靴回放比較；除了整體品質，中後段需要有實證收益，前 1～20 局亦不能有不可接受的 Brier／hit／EV 退步，才可另行晉升。原 CLI 的 legacy 對照不能冒充這項 production 比較。

## 保留與回滾

2 個百分點預設顯示門檻、1／2／3 切換、不觀望測試、開通碼驗證與各裝置本機隔離保留。既有正式 EV／Volume Guard 分段仍是 1～40／41～50／51～70，沒有為了多出手改門檻。

將 `validation/cardcount_moderate_v1_baseline.json` 的內容還原為 `cardcount_profile.json`，重新整理即可回到前版平台值、PF 3 次抽樣與原本校準；原版無 stage_anchors 時使用 decay=1、prior=100、視窗32。或還原本次提交作完整回滾，不需還原模型檔。
