# B/P/T 盲盒邏輯升級與驗證

已實作三項升級，維持 Direct Physics → Particle500 → EMA → Frozen Core / XGB → 決策門檻架構，不新增實際點數或張數欄位、不改模型維度、不重訓。正式預設 **baseline**，新方案尚未達成觀望率與命中率同時改善的驗證目標。

## 1. 結果條件下的合法張數消耗

`particle_filter_runtime.js` 的 `conditionalHand` 重複抽取完整合法牌局，僅接受結果與已知歷史相同的牌局，並保存該手實際抽出的假設牌值。這直接抽樣 P(牌局、張數 | 已知結果、該粒子剩牌)。原版在抽不到同結果時可能使用異結果牌局，新方案將該粒子權重設為零，不再使用矛盾回退。粒子耗盡時明確回報錯誤，最終觀望。

每個粒子另用 12 次獨立試抽估計已觀測結果的似然，再更新權重；不能用「拒絕抽樣直到成功」的成功次數直接代替似然。消耗張數與補牌規則是同一模型，不能抽完合法牌再重複套一層張數先驗。

`blind_upgrade_reference.py::exact_initial_matrix` 對八副牌有限牌組、無放回、標準補牌規則完整枚舉，得到新牌靴的條件機率：

| 已知結果 | 4 張 | 5 張 | 6 張 |
|---|---:|---:|---:|
| 莊 | 37.4104% | 34.1975% | 28.3921% |
| 閒 | 38.4459% | 26.8756% | 34.6786% |
| 和 | 37.5613% | 28.0424% | 34.3963% |

這些只是新靴參考值，非所有靴深的固定矩陣。和局 10/40/50、莊閒 45/40/15 並非由標準規則得出的數值；已保留為 `requested_prior` 實驗模式。此模式先抽目標張數，再拒絕不符該張數與結果的完整合法牌局，是人為改變目標分佈的假設，不得宣稱為真實剩牌估計。補牌規則參考：[Nevada Gaming Control Board 標準牌局與張數規則](https://www.gaming.nv.gov/siteassets/content/divisions/enforcement/rules-of-play/4-5-6_Card_Tie_Baccarat_-_Rules_of_play_and_paytables.pdf)。機率數值由本倉庫程式獨立計算。

## 2. 靴深、標準差與有效樣本數門檻

深度等於已結束歷史長度，包含和局。1–15 局將可放寬的 EV 部分降低 30%，信心帶乘 0.85。16–50 局保持原門檻。超過 50 局提高 EV 門檻 0.002；僅在標準差 ≤0.08、Monte Carlo 標準誤 ≤0.015、重抽樣前 ESS 比例 ≥0.50、祖先粒子比例 ≥0.25、無失敗/不足預測粒子時，免除這筆額外後段懲罰。

不採「標準差小就無條件解除觀望」：祖先塌縮、有限試抽或共同模型偏差都能製造假共識。新方案每個粒子使用 64 次合法下一手試抽，計算粒子內莊勝占非和結果比例，再計算加權粒子間標準差與標準誤。這仍含有限次試抽的雜訊，並非已校準的真實預測誤差或勝率保證。ESS 使用重抽樣前值，不能把重抽樣後回到 500 的 ESS 當成新資訊。

放寬僅作用於原決策流程既有位置，保留正 EV 下限 0.002、方向一致性與嚴重 Core/XGB 聯合衝突。不能從 Physics 的觀望直接強造一個反方向訊號。EV 沿用標準莊家抽佣 5% 的既有定義。

## 3. 熵與轉換率分開

前 5/10 個已結束牌局內排除和局後計算 B/P 轉換率 r，以及 H=-r log2(r)-(1-r)log2(1-r)。長龍 r=0、規律單跳 r=1，兩者 H=0；r 接近 0.5 時 H 高。兩個視窗均至少 4 筆方向資料、合計至少 6 筆方向歷史才啟用；不足時保留原權重。這只是描述統計，不證明洗牌成團，也不證明高熵下模擬更準。

Core 輔助支持度權重在 0.05–0.30 之間調整；高熵降低 Core 結構影響，低熵提高。Core/XGB 加權支持度過低時把候選信心收縮 0.85，嚴重聯合衝突仍觀望。Particle 對 Direct Physics 的混合仍限制於 0.06–0.16，後段 ESS 偏低時上限 0.10；不改 Frozen Core 的 256D 或 XGB 57D 特徵/訓練分佈。

## 模式與回退

授權載入器先載入 `blind_upgrade_runtime.js`；不修改開通碼、裝置綁定、資料隔離、OCR 或按鈕輸入。`BBB_BLIND_UPGRADE_MODE` 在模組載入前設定：

- `baseline`：正式預設，原本決策與粒子結果；可用來精準回退。
- `guarded`：上述合法條件粒子、動態門檻與熵適配方案。
- `requested_prior`：加上使用者指定的未校準張數比例，只供實驗對照。

設定只供程式整合及離線 runner 使用，不增加一般使用者操作欄位。模式在模組載入時固定；比對不同模式需新建獨立 runtime，禁止途中混用。

## Python 關鍵函式

見 `blind_upgrade_reference.py`，可直接使用：

```python
from blind_upgrade_reference import sample_history_consumption, dynamic_threshold
hand = sample_history_consumption([128] + [32]*9, "T")
# hand 含假設剩牌與該手消耗張數；不能當成實際賭桌的牌。
limits = dynamic_threshold(len(history), ev=.05, band=.02, diagnostics=diagnostics)
```

瀏覽器正式程式是 JavaScript；Python 是可執行的數學與邏輯參考，不會取代前端引擎。

## 測試與效果界限

`node test_blind_upgrade.js` 驗證 15/16、50/51 局邊界、長龍/單跳/混亂/和局資料不足、ESS/祖先退化、無資料或 NaN、合法消耗、500 粒子、可重現重建、機率正規化、不翻轉方向、原版回退一致及實驗先驗比例。

`python -m unittest -v test_blind_upgrade_reference.py` 驗證完整枚舉、合法條件抽樣與門檻。

`node tools/blind_upgrade_replay.js --synthetic 5 validation/blind_upgrade_smoke.json` 使用按時間排序的五靴，最後 20%（一靴 70 局）作機械流程對照，僅提供歷史 B/P/T，每次先預測才揭露下一局，不傳點數或張數。報告保留觀望率、出手命中率（和局不計勝負）、含莊佣金淨單位、Brier、最長連錯與分階段數據。

本次模擬：baseline 69 次預測、16 次出手、9 勝 7 負；guarded 17 次出手、9 勝 8 負。觀望率 76.81% →75.36%，命中率 56.25% →52.94%，淨單位 1.8→0.8，最長連錯 3→4。**未達雙目標，正式預設不切換。** 不能以此小型模擬估計實戰勝率，也未取得獨立真實牌靴驗證改善。

真實比較可執行 `node tools/blind_upgrade_replay.js observed_hands.json report.json`，輸入至少五靴、逐靴分組並按時間排列的 `{rows:[{shoe_id,round_index,outcome}]}`。不選參數、不自動晉升；來源仍需人工核驗。不要反覆利用同一 holdout 調參後宣稱獨立驗證。
