# 點數條件粒子後驗

## 使用流程

1. 開通後直接使用點數欄位，不必貼入過往 B/P/T 路單。點數後驗每次預測都從新靴先驗重置，只使用最近完成的一局閒、莊點數，不會把前局點數累積進剩餘牌組後驗。
2. 可先選本局總張數 4、5 或 6 張，再輸入閒、莊最終點數 0–9。兩個點數完成後，程式會按高低自動判定 B/P/T、走既有授權結果記錄流程，並分析下一局。
3. 總張數可留空。每局點數、張數與結果仍寫入本機逐局紀錄，供返回上一局及離線匯出；點數 PF 只讀取最新一筆點數。原有勝負序列模型仍依其既有設計讀取自動判定的 B/P/T 結果。
4. 畫面不再顯示 B/P/T 歷史路牌串；局數與點數條件後驗診斷仍會更新。模型權重狀態會說明 Physics／輔助模型仍採用最近一次離線訓練版本。

## PF 更新方式

舊版 `estimate()` 保留原本 13 類牌面粒子與 B/P/T 結果似然。新 `estimatePoints()` 使用 10 類百家樂點數粒子（0–9；新靴組成為 128 張 0 點、其餘各 32 張）。預設每次預測都從新靴先驗建立粒子，只條件於最近一局的點數／張數觀測；不重播前局點數，也不保留跨預測粒子狀態。

對每個粒子，程式套用百家樂閒／莊發牌規則，枚舉所有符合已知閒、莊最終點數（以及可選總張數）的 4／5／6 張路徑，計算觀測似然。當只有舊 B/P/T 或只有總張數時，使用相應較弱的結果／張數條件似然。粒子權重依似然更新，依 ESS 門檻系統重採樣，再依條件路徑抽出隱藏點數牌並從粒子牌組移除。下一手分布、莊／閒扣佣 EV 與不確定度由加權粒子組成計算。

觀測路徑似然使用精確發牌規則的「獨立抽牌」有限狀態 DP；這對有限牌靴屬近似。條件抽牌會按粒子現有點數張數不放回抽樣，並以拒絕重試修正部分有限牌靴差異。點數 0 合併了 10/J/Q/K，因此本後驗不能識別實際 rank 組成，也不能視為觀察到賭場剩餘牌組。診斷會明確標出似然近似方式、ESS、重採樣次數、剩餘點數類別均值，以及新點數 PF 和原 B/P/T PF 的勝負機率差。

## 選擇性出手

啟用點數後驗時，第一層和最終 EV 都以 PF 點數後驗作為 Physics 主分布。既有第一層因果平滑仍保留；點數策略提高 EV 門檻，在 noise 高或粒子品質低時再提高門檻，縮小 Volume Guard 目標出手比例，並要求第一層與最終層方向一致。任一方向為 Balanced、方向不一致、EV 或 Volume Guard 未通過時不出手。

這些條件可能減少出手，理論上能避免較多弱訊號局，但尚未由實際前瞻資料證明可提升命中率或期望值。畫面與程式都不保證正期望。

## 開關與回滾

`cardcount_profile.json` 的開關如下，全部新路徑均可停用：

| 設定 | 預設 | 用途 |
|---|---:|---|
| `point_input.enabled` | `true` | 啟用點數輸入介面；關閉會選用 B/P/T 相容模式 |
| `point_input.default_mode` | `"points"` | 未有本機模式偏好時的預設輸入方式 |
| `point_input.compatibility_mode_default` | `false` | 設為 `true` 會預設 B/P/T 模式 |
| `pf.score_update_enabled` | `true` | 允許點數／張數觀測進入十類點數 PF |
| `pf.independent_point_update_enabled` | `true` | 每次重置 PF，只使用最新一局點數；設為 `false` 可恢復累積整靴點數的舊行為 |
| `score_conditioned_entry.enabled` | `true` | 啟用點數後驗專用 EV 門檻、低出手率目標與方向一致限制 |

在畫面點「開啟 B/P/T 相容模式」即可立即切換舊輸入與舊 B/P/T PF。要停用點數 PF，可將 `pf.score_update_enabled` 設成 `false`；要回到舊式整靴點數累積，將 `pf.independent_point_update_enabled` 設成 `false`。要以設定檔全域回滾，將 `point_input.enabled`、`pf.score_update_enabled` 和 `score_conditioned_entry.enabled` 設成 `false`，並將 `point_input.default_mode` 設成 `"compatibility"`，再重新載入。`estimate()` 舊 PF 程式和授權、30 分鐘限制、本機資料隔離路徑均保留。使用者介面不再提供貼入 B/P/T 歷史路單或顯示 B/P/T 歷史串的功能。

`point_input.enabled` 關閉後可恢復舊 UI 流程；若要完全撤銷本次程式碼，可使用本次 Git 提交的父版本，或反向套用本提交。新增的研究工具不會改部署權重。

## 離線訓練

瀏覽器只更新粒子、計算平滑與 EV，不做完整模型重訓。可在「匯出開牌資料」取得本機逐局 JSON，再於有 `research/requirements.txt` 依賴的離線環境執行候選輔助模型訓練：

```bash
python -m research.train_point_auxiliary \
  --input bgs_observed_hands_export.json \
  --output research/candidates/point_auxiliary_candidate.json
```

工具按整靴固定分割 train／holdout，按時間順序建立特徵並檢查目標局點數沒有洩漏至自身特徵。輸出標記為 `review_only_not_promoted`，需人工檢查、另行驗證與部署；程式不會覆寫 `physics_multitask_model.json`、`final_probability_model.json` 或正式設定。
