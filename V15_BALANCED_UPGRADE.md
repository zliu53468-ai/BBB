# BBB V15-B：雙目標平衡、可回退的最小診斷與離線回放

## 變更範圍 / 保護清單

這次不是更換模型或打開 0 觀望。正式配置中的 `decision_policy`、`ev_thresholds`、MLP 和 XGBoost 訓練權重未改，且不修改 Frozen Core（`app256forward.js` / `app256continuation.js`）、213D→48D MLP、每局重建 Particle500、57D 特徵、no-flip 輔助篩選、Dynamic Clip＋EMA、Final EV Guard、#43 Volume Guard、網頁及開通碼系統。

- `final_probability_runtime.js`：僅加入 `v15SkipDiagnostics` 純讀取診斷，輸出 `prediction.finalProbability.skip_diagnostics`，並隨 `registerPrediction` → `settlePending` 保存於每手已結束快照（`getTrainingRows`／匯出分析紀錄），**不改變 B/P/Skip**。
- `v15_candidate_policies.json`：純離線候選參數，`mode:offline_shadow_only`、`auto_activate:false`；和 `final_probability_model.json` 完全分開。
- `v15_balanced_replay.js`：重放每手 **先預測、後揭曉**，使用各候選的獨立 VM 與儲存狀態；每靴不共用牌史；候選的參數覆蓋只發生在測試用記憶體，**不能影響正式模型**。
- 原有回歸測試新增 Skip 理由對應與結算後保存驗證。GitHub Actions 執行合成牌靴的 shadow smoke；合成牌靴只能證明程式運作，不能證明可以提高中獎率。

## 一、Skip 逐局診斷

每局都有一個 `skip_diagnostics` JSON 欄位：

- `primary_reason`：第一個失敗環節，`entry` 代表非 Skip
- `failed_conditions`：同局全部不滿足條件，避免只把第一個原因當成唯一原因
- `physics_distance`、`physics_band`、`physics_ev_banker`、`physics_ev_player`、`physics_activation_ev`
- `final_distance`、`final_effective_band`、`final_candidate_edge`、`final_activation_ev`
- `xgb_aux_decision`、`volume_guard_active`、`stage`、`round_index`

Reason codes：`physics_unavailable`、`physics_confidence_band`、`physics_ev_gate`、`auxiliary_joint_conflict`、`final_direction_alignment`、`final_confidence_band`、`final_ev_gate`、`unclassified_skip`。`failed_conditions` 可能同時包含多個值；`primary_reason` 是優先顯示順序，不代表因果貢獻比例。診斷不依賴當局開獎結果，因此無資料洩漏。

## 二、候選參數 (不會自動啟用)

正式基準仍是 V14.8/10ce55b 的政策。離線 A/B 比較只模擬第 1～40 局所用的 `decision_policy.confidence_band.early` 和 `ev_thresholds.early` 小幅放鬆：

| 候選 | Early Band 減量 | Early EV 減量 |
|---|---:|---:|
| baseline | 0 | 0 |
| early_precision_1 | 0.00075 | 0.00050 |
| early_precision_2 | 0.00150 | 0.00100 |
| early_precision_3 | 0.00200 | 0.00150 |

所有候選仍呼叫原始 Final EV Guard / #43 Volume Guard、完整 Particle500 和 Frozen Core，不允許方向翻轉。第 41 局後參數不變；某些當局歷史經過的出手紀錄可能改變 Volume Guard 的狀態，因此中後段輸出也可能 **間接受影響**，必須一併核對。離線候選只在 `v15_candidate_policies.json` 維護，**不能複製到正式 JSON 而不再驗證**。

## 三、真實牌靴資料格式與回放

需要每靴連續完整 B/P/T 紀錄（包含 Skip 或和局當局），按發生時間 **依靴排序且每靴獨立**，不得挑選贏的靴或刪除輸局。可從 BBB 網頁「匯出開牌資料」取得；若選填張數和點數，會提高觀測品質。範例：

```json
{
  "data_kind": "real_observed",
  "rows": [
    {"shoe_id":"shoe-A","round_index":1,"outcome":"B","card_count":4,"player_score":3,"banker_score":8},
    {"shoe_id":"shoe-A","round_index":2,"outcome":"T","card_count":6,"player_score":6,"banker_score":6},
    {"shoe_id":"shoe-A","round_index":3,"outcome":"P","card_count":5,"player_score":7,"banker_score":4}
  ]
}
```

`data_kind` 只是使用者自述的資料類型，**程式無法憑此驗證真實性**。建議用原始可稽核的完整會員記錄比對。

```bash
node v15_balanced_replay.js observed_hands.json v15_real_report.json
node v15_balanced_replay.js --synthetic 5 /tmp/v15_smoke.json
```

至少需 5 靴才能執行 smoke；正式放行報告另要求：按靴原始順序切分為訓練 60%、校準 20%、最後一次性保留測試 20%，保留測試集 **至少 30 靴**，且基準與候選都各有至少 500 次非和局有效出手。候選只用訓練與校準資料選出，之後才評估一次 holdout，不能因看到 holdout 分數再次改候選。請勿把所有測試靴已看過的候選重新優化當成獨立測試。

分段：2–5、6–10、11–20、21–40、41–50、51+ 局（每靴第 1 局尚無牌史，預設不可分析）。統計出手率、Skip、非和局出手命中率、每注實現 EV、每分析機會收益與 Skip 原因；tie 實際淨收益 0，計入已下注分母但不計入命中率分母。莊贏 +0.95、閒贏 +1、押錯 -1。

報告中 `promotion.eligible` 僅能當作供人**再審查**的建議：真實來源標籤、獨立 holdout 充足、出手率提升、命中率與 EV 點估計不降，且 paired shoe bootstrap 95% 下限皆不低於 0，才可標記；所有案例 `auto_activate:false`。門檻再放鬆通常會引入低品質局，不能預設命中率或收益改善。

## 四、驗證 / 回滾

```bash
node --check final_probability_runtime.js
node --check v15_balanced_replay.js
node test_final_probability_runtime.js
BGS_USE_GENERATED_MODEL=1 node test_final_probability_runtime.js
node test_macro_runtime.js
node v15_balanced_replay.js --synthetic 5 /tmp/v15_smoke.json
```

GitHub Actions `Physics Primary Runtime Checks` 跑同等的 smoketest + no-auto-promotion 檢查；完成後再比較 baseline 與候選。未取得真實且數量充分的牌靴資料前，只能說已完成診斷與離線評估能力，**不能宣稱命中率／出手率已改善**。

若需回退，先記錄開始時原本 `main` SHA 與目前 `main` SHA；使用 `git revert` 退回本次新增的提交，不用 `reset --hard` 或 `force push`。只需撤去 V15 診斷程式與新增 shadow 檔案，`final_probability_model.json` 無需任何還原。
