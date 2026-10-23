# 算牌優先｜參數候選與正式路徑驗證

這個目錄只提供診斷、候選與唯讀驗證工具。**沒有啟用候選，沒有改正式模型或開通碼。**

完整診斷、參數表、重訓命令、晉升門檻與回滾：[TOP_PARAMETER_PLAN.md](../../docs/TOP_PARAMETER_PLAN.md)。

完整 middle_late_focus 重訓與獨立 holdout 執行流程：[MIDDLE_LATE_RETRAIN_EXECUTION.md](../../docs/MIDDLE_LATE_RETRAIN_EXECUTION.md)。`run_middle_late.py` 沿用 production 的 1.10/1.25/1.35 及 10%/20% 取樣，**不會套用這個目錄之前的較保守 retrain_profile.json**。資料或依賴不足時會寫入阻擋報告，不會啟動訓練。

在倉庫根目錄執行（Python 3.10+、Node.js 18+）：

```powershell
python tools/top_parameters/test_gate.py
python tools/top_parameters/prepare.py . ../BBB-top-experiment-01
node tools/top_parameters/replay.js ../BBB-top-experiment-01/baseline C:/BBB-data/strict_holdout_hands.json baseline_holdout.json
node tools/top_parameters/replay.js ../BBB-top-experiment-01/smooth C:/BBB-data/strict_holdout_hands.json smooth_holdout.json
python tools/top_parameters/gate.py baseline_holdout.json smooth_holdout.json smooth_gate.json
```

資料路徑須換成實際獨立完整靴。第一次執行 prepare 會在倉庫外建立隔離副本；目的地存在時停止，不覆蓋。production_manifest.json 逐檔驗證已審核的正式來源，來源改變時停止，須重新審核。prepare 不修改來源、不自動部署。

- baseline_profile.json：本次凍結 production 的設定。
- smooth_profile.json：平滑、PF 抽樣及本機校準候選。
- retrain_profile.json：另含 middle_late_focus 重訓候選參數。
- replay.js：實際正式 runtime 每次先預測再結算；不模擬所有 UI 操作或畫面本機學習。
- gate.py：整體與分段數值硬門檻。失敗 exit 2；成功也只代表數值子集通過，仍需 bootstrap、資料來源與模型架構人工檢查。報告的 production_promotion_allowed 永遠為 false。

已有合成機制測試發現候選退化，**不能當作已驗證提升的設定**。完整重訓與真實 holdout 尚未完成。請先搜尋開發集，凍結候選後才開封 strict holdout。
