# PF 移除與規律重訓候選（2026-10-22）

基準 main：`0bbd1f7db966be5c67f00809fa4db1822386ac21`。本文件取代舊文件中的 PF 流程敘述。

## 正式流程

B/P/T → Direct Physics 213→48D → Physical EV／candidate → Dynamic Clip＋EMA → Frozen Core 256D＋57D XGBoost 輔助篩選 → Final EV Guard → Volume Guard → B/P/Skip。

`auth-client.js` 是實際載入器，已移除 PF；`index.html` 不直接載入 PF。`particle_filter_runtime.js` 僅保留 deprecated 空檔，不建立任何全域物件。主 runtime 不讀取 PF、不呼叫 estimate、不等待 500 粒子。不因缺 PF 而觀望；Physics 或語義相容的 57D 模型未就緒則觀望。XGB 執行錯誤也不允許繼續出手。

Frozen Core、Physics 權重、57D 權重、特徵維度與順序、三階段 EV、Macro EMA、Volume Guard 和方向不翻轉規則均保留。48D slot 42 的 `particle_uncertainty` 是既有權重相容名稱，值來自 Direct Physics；這個欄名不是殘留的 PF 計算。

歷史文件、舊 shadow／blind reference 工具及對應 PF 測試不是正式載入路徑；其中描述 PF 的數字不能作為本版本驗收結果。`evaluate_macro_replay.js` 僅在讀取不同目錄的原始基準時載入其 PF，以確保比較的是原始 main。

## 可調參數

`final_probability_model.json.smoothing.progress_alpha_anchors` 控制進程平滑。格式為進程 0～1 與 alpha .35～.75 的遞增節點，例如 `[[0,.38],[.5,.53],[1,.72]]`。首尾進程必須是 0 和 1；非法值回到既有預設。JS 與 Python 使用相同驗證和插值。alpha 越高越敏感；仍受原 Dynamic Clip 與 EV 門檻約束。不新增不經驗證的機率偏移。

`python xgb_final_probability.py train --input REAL_TRAIN_ROWS.json --physics-model physics_multitask_model.joblib --pattern-weight 0.35 --min-skip-reduction 0.01 --output candidate_model.json`

- `--pattern-weight`：0～1，預設 .35；0 還原原始權重。對既有 `original7_sx_markov_p_same` 的重複／交替結構對稱加權，歷史 8 手以前不加權，24 手達完整權重；保留原樣本權重上限與類別／階段平衡。只影響搜尋與最終模型訓練，不改校準和 holdout 權重。沒有新標籤或額外模型維度。
- `--min-skip-reduction`：絕對 Skip 比例下降要求，預設 .01（1 個百分點），範圍 (0,1]；取代原本允許 Skip 增加的發布條件。命中率和每注實現 EV 不得倒退，其他既有品質閘門仍需通過。
- Macro EMA 的觀測、係數、noise gate、min_observations、max_age 等原參數保留。改動平滑或訓練參數後必須重做校準及獨立驗證，不能只修改正式 JSON 宣稱成效。

規律強化目前是可執行的重訓候選程式；本次沒有產出或啟用新訓練權重。沒有真實訓練資料，不能證實分類器已學到額外預測能力。

## 驗證與目前未達成項目

通過 50 項 Python 測試、主 runtime／Macro runtime 整合測試、按鈕介面測試及 Macro 校準測試。涵蓋 PF 物件讀取即拋錯仍可推論、缺 57D 模型觀望、48D／57D、因果 EMA、重載／復原、no-flip、正最終預測 EV。

`validation/no_pf_smoke.json`：固定 seed 20261009，12 靴／840 手合成資料，828 次有效分析。先預測再結算；主模型與基準都使用原始權重。

| 全體指標 | 原 main | 移除 PF |
|---|---:|---:|
| Skip | 73.43% | 72.95% |
| hit_rate_on_bets | 50.50% | 47.06% |
| realized_ev_per_bet | -0.00114 | -0.06250 |
| 方向翻轉 | 0 | 0 |

此合成回放只驗證機制。Skip 僅下降 0.48 個百分點，命中率／實現 EV 均退步；**不滿足使用者的全部目標，不可據此發布為成效升級**。預測 EV 為正也不代表實現 EV 為正。沒有為了讓回放好看而在這批資料上調參。

真實資料驗收：準備訓練／校準從未使用、以牌靴分離的完整連續觀測資料，`data_kind` 為 `real_holdout_observations`。只填此標籤不代表資料已驗真。

```
node evaluate_macro_replay.js real_holdout.json report.json ORIGINAL_MAIN_DIRECTORY
node tools/no_pf_acceptance.js report.json
```

驗收預設至少 100 靴，各階段雙版本至少 500 注；總體 Skip 至少下降 1 個百分點，各階段 Skip 不增加，命中率／每注實現 EV 不下降，且沒有方向翻轉。`assess(report, options)` 可調整最低樣本與下降幅度。這是觀測品質閘門，不是統計上保證未來不退步。尚需真實資料、重訓與獨立回放，才能決定是否合併部署。
