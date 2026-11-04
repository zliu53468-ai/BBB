# BBB V4：總點數狀態轉移 + 動態適度觀望

## 唯一輸入與預測時序

使用者每次只能輸入「閒最終總點 0–9、莊最終總點 0–9」並按「開始分析」或 Enter。
不需要逐張牌、牌數、任何發牌物理資訊，也不需要手動填寫歷史。
每次操作獨立提交，但是背景保留過往**連續**的雙方總點數，以學習轉移統計。

```text
接收本次閒／莊最終點數
  → 用已揭露的結果結算上次預測
  → 更新先前狀態 → 本次狀態的 100×100 轉移矩陣
  → 以「本次點數狀態」查詢其歷史後續
  → 動態窗口 / 最低樣本檢查
  → 動態差值門檻 + 1024 次後驗機率採樣
  → 輸出「莊／閒／觀望」，並持久化歷史
```

**因果性：** 當次已揭露點數只能用於下一局預測，不能用未揭露的下一局結果當作資訊。

## JavaScript 物件導向結構

- `StateTransitionMatrix`：100×100 的稀疏持久化頻率表。狀態代碼為 `10*閒+莊`。
- `StatisticalV4Filter`：B/P/T 結果彙整、樣本門檻、動態機率差、低變異例外。
- `ParticleConfidenceSampler`：1024 次由點數頻率建立的 Dirichlet 後驗機率向量採樣，估計莊／閒機率分布的標準差；**不是實體牌組粒子**。
- `AdaptiveObservationController`：選擇最近 240／720／2160 次及全部累積的轉移資料；連續觀望後擴大搜尋。
- `PointTransitionPredictor`：持續學習、結算、歷史快照、重新載入、返回上一筆、序列中斷。

## V4 參數表

| 設定 | 數值 | 意義 |
|---|---:|---|
| `minSamples` | 12 | 同一總點數起始狀態至少有 12 次後續 |
| `minDecisive` | 8 | 其中莊／閒結果合計至少 8 次 |
| `priorPerOutcome` | 0.5 | 每一 B／P／T 類別的平滑偽計數 |
| `baseGap` | 12% | 開始學習時的機率差門檻 |
| `minGap` | 4% | 標準門檻下限 |
| `gapDecayPerRound` | 0.15 百分點 | 每累積一次點數輸入下降值 |
| `relaxAfterSkips` | 6 | **已經超過**連續 6 次觀望，下一次評估才放寬 |
| `relaxation` | 0.7 | 暫用正常門檻 70% |
| `particleCount` | 1024 | 點數後驗機率向量模擬次數 |
| `particleStdMax` | 8% | 莊與閒兩種概率估計的標準差均須低於此數 |
| `particleOverrideGap` | 4.5% | 低變異例外仍需超過的莊閒機率差 |
| `targetWindow` | 6 | 最近六次訊號頻率的觀察視窗 |

動態公式（只依累積輸入筆數，不使用任何實體發牌概念）：

```js
const standardThreshold = Math.max(0.04, 0.12 - totalRounds * 0.0015);
const currentThreshold = consecutiveSkips > 6
  ? standardThreshold * 0.7
  : standardThreshold;
```

在滿足樣本下限後，若 `abs(P(B)-P(P)) > currentThreshold`，輸出優勢方向；
或在 1024 次後驗抽樣中莊、閒機率標準差皆小於 8%，且
`abs(P(B)-P(P)) > 4.5%`，也可輸出方向。否則觀望。
出方向後連續觀望次數清零，下一局恢復標準門檻。

「超過 6 次」與「乘 0.7」採用本次指令中**後一組、較明確的規格**；
舊版的觀望 4 次、乘 0.75 規則不再使用。

## 持久化、版本與安全檢查

使用既有 `bgs_point_transition_history_v1` 的會員隔離儲存與
`BBB_POINT_TRANSITIONS_ADAPTIVE_V1` 快照版本，**不破壞原本累積資料**。
重新整理不會重複訓練；新輸入時更新矩陣；`forecast()` 純讀取。
返回上一筆會撤回對應轉移；中斷輸入連續性可選「另起連續序列（保留學習）」。
最多保存 2161 筆近期逐局明細，但完整頻率矩陣不會因此清零。
若儲存或讀取異常，應先顯示觀望，不將未確認的預測當作新結果。

載入清單只有 `point_transition_runtime.js` 與 `point_transition_panel.js`。
歷史檔案中的舊版前端與模型只保留備援，不加入目前的 V4 決策。

## 回測與限制

「近 6 次希望出手 1–2 次」與「觀望率 65%–75%」是**測量目標，不是硬性保證**。
尤其 100 種起始總點數組合往往非常稀疏，70 次輸入不一定能讓任何一種
起始狀態累積到至少 12 次後續樣本。遇到資料不足時，即使已連續觀望，
也不應為了達成目標而強迫出手。

1024 次模擬降低的是條件機率估計的蒙地卡羅誤差，**不會憑空增加原始樣本，
也無法證明相鄰局之間存在可盈利的轉移規律**。若實際結果接近獨立隨機抽樣，
此演算法可能只是在擬合歷史波動。歷史差距、粒子低標準差和模型訊號
皆**不是經過校準的實際命中率**。必要驗證包括不洩漏未來結果的逐局回放、
不同資料集的樣本外檢驗與觀望率統計。

## 檢查命令

```sh
node --check point_transition_runtime.js
node --check point_transition_panel.js
node test_point_transitions.js
node test_point_transition_panel.js
node test_license_storage.mjs
node test_end_analysis_reload.js
```

GitHub Pages 由 `.github/workflows/pages.yml` 在 main 更新時部署至 CNAME
`bgsv1.xyz`，需以實際 GitHub Actions 部署結果確認是否完成上線。
