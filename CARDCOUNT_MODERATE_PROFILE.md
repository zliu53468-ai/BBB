# 分段跟牌｜適中觀望｜激進重訓候選｜算牌優先

## 畫面方向與下注資格

預設採用 `cardcount_profile.json` 的 `cardcount_moderate_v1`。一般畫面在排除和局後，莊／閒機率相差至少 **2 個百分點**時顯示較高一方。例如 51%／49% 顯示莊，50.99%／49.01% 仍觀望，49%／51% 顯示閒。即使觀望，百分比仍保留。

頁面可切換 1／2／3 個百分點；只在當次頁面記憶，重新整理回預設。「全程不觀望測試」忽略此顯示門檻，保留每次報結果後自動更新、返回上一局重算、五五波固定以莊作測試方向等行為。缺模型不編造方向。

顯示方向不等於正式下注。EV、noise、confidence、Volume Guard 仍決定原本 B/P/Skip 下注資格；不會為了跨過顯示門檻而新增下注或繞過結算保護。不觀望測試仍不新增正式下注紀錄。

## 分段參數

局數是「下一局」序號，包含已輸入的和局。70 局後參數停留於後段端點。

| 參數 | 前段 1～20 | 中段 21～45 | 後段 46～70 |
|---|---:|---:|---:|
| 決策 EMA 最新機率權重（名目） | 45% | 55% | 65% |
| Macro 短期 α | 0.28 | 0.30 | 0.32 |
| Macro 長期 α | 0.08 | 0.10 | 0.11 |
| PF 混合範圍（上限，非固定比重） | 6%～10% | 6%～14% | 6%～16% |
| XGB 階段樣本權重（正規化前） | 1.10 | 1.15 | 1.25 |

第 18～23、43～48 局線性漸變，避免跨段突跳；上述為各段平台值。Python 與瀏覽器讀取同一份 JSON anchors，測試逐點比對一致。

決策 EMA 在高 noise、高 uncertainty 時略降低最新機率比重，硬限 35%～75%。沒有同靴可用的先前正式 snapshot 時採用當次機率（α=1），不編造歷史；不觀望測試不新增正式 snapshot。畫面機率仍直接顯示 Physics＋PF，沒有額外套決策 EMA。

Macro 缺測跳過、不當零；和局 Low_Score=0 依既有定義更新。每靴重建、返回／重載由觀測紀錄回放。min_observations=7、max_age=5、max_delta=0.040、progress_start=0.22、progress_power=1.6、noise gate=0.55～0.72。原模型未校準的方向係數維持 0，不因加快 EMA 就擅自設定方向。

這些新分段用於平滑、PF 與候選訓練權重；既有正式 EV／Volume Guard 的 1～40、41～50、51～70 分段及門檻不變。JSON stages 是名稱與區間說明；實際插值以 anchors 為準。

## PF 穩定與靈敏度

每次建立全新 500 粒子、回放當下完整歷史，不延用前次粒子／權重／重採樣狀態。同一歷史固定種子可重現。每粒子下一局預測由 1 次提高至 3 次，降低單次抽樣波動；全歷史回放仍有運算成本，不承諾即時速度或準確率提升。

quality 混合先受階段上限限制，再依 Direct Physics／PF 同向、ESS 與後驗不確定性決定提升幅度；反向時保持最低比重。保留 PF 不可單獨翻轉 Direct Physics 的限制。XGB 仍用未經 PF 校正的 Direct Physics 特徵，維持 57D 訓練分布。

## 激進重訓：獨立候選，不自動換權重

兩個 trainer 新增 `--training-profile aggressive_cardcount`；不帶參數仍為原本 `production`，CLI 舊用法相容。

Physics 候選：Winner 3.50、點數各 1.45、4/5/6 張數各 1.65、Rank 各 0.55，其他任務保留原值。這是相對目前 3.25／1.35／1.55 的小幅增強，不提高假定牌值權重。拓撲、校準與 uncertainty mapping 不變。

XGB 候選：保留原搜尋組，再加四組 depth 2～4、500～750 trees、learning_rate 0.015～0.025、min_child_weight 8～12、subsample 0.90、colsample 0.85、alpha 0.20～0.50、lambda 10～16。依 Brier、bet hit、EV/bet、Skip 順序選擇；保留既有品質門檻，不以規律段成績硬晉升。

候選權重依實際下一局局數插值 1.10／1.15／1.25，低物理 uncertainty 最多加 18%，物理可出手訊號最多加 8%；規律額外權重 **0%**。最後保留類別平衡、平均正規化與 0.35～1.50 硬限。actionability 不讀未來 outcome，只有既有物理特徵；標籤仍是 P=0、B=1。

建議 Physics 6,000 靴，validation 0.20、calibration 0.10、augment 0.25。XGB 依完整牌靴 chronological train/tune/calibration/holdout 切分，保留 Identity／Platt／Isotonic 選擇。

```sh
python physics_feature_extractor.py train --training-profile aggressive_cardcount --shoes 6000 --validation-fraction 0.20 --calibration-fraction 0.10 --augment-ratio 0.25 --output physics_aggressive_candidate.joblib --browser-output physics_aggressive_candidate.json
python xgb_final_probability.py train --training-profile aggressive_cardcount --input complete_shoe_rows.json --physics-model physics_aggressive_candidate.joblib --output final_probability_aggressive_candidate.json --report-output aggressive_candidate_report.json
```

上述輸入檔是使用者提供的完整資料，不是倉庫既有資料。此次沒有可用的完整訓練集，因此未執行正式重訓、未換任何模型權重。工具拒絕 aggressive profile 寫入正式模型慣用檔名，報告標記 candidate_only、production_comparison_required。既有 CLI gate 對照的是 legacy 訓練候選，不能取代與目前實際 production＋PF＋Guard 的回放比較。

候選必須在同一份未參與調參的獨立完整牌靴，與目前 production 比較 Brier（惡化不超過 0.0015）、bet hit、EV/bet、Skip、wager 數、最大連敗與各階段成績；不接受因降低觀望造成實質品質退步。通過前不得替換正式檔；無自動晉升。PR／合併提交使用 `[skip model-training]`，避免 CI 自動重訓覆寫。

## 回滾與驗證

設定 JSON `enabled:false`，重新整理即可回到之前的平滑、PF quality 混合與無顯示差距門檻；模型權重與觀望切換保留。CLI 繼續不帶新 profile 即使用原訓練設定；完整回滾可還原本次提交。

測試涵蓋 51%／49% 邊界、反向與五五波、頁面閾值切換、不觀望模式、Python／JS anchors 一致、噪音抑制、Macro 回放／返回／缺測、PF 500 粒子獨立重建與階段上限、原 48D／57D 契約、候選不可覆寫正式檔與既有回歸。測試通過是行為驗證，不能作為實際命中率、EV 或獲利提升的證據。
