# 更敢出手｜平滑跟牌｜重訓尖銳｜再降觀望｜算牌優先規律 soft

本版以 [CARDCOUNT_AGGRESSIVE_SMOOTH_SKIP.md](CARDCOUNT_AGGRESSIVE_SMOOTH_SKIP.md) 及 `cardcount_profiles.json` 為最新規格，舊權重／搜尋數值已被取代。

- 架構鎖定 Frozen Core、213→64→32→48、57D 特徵定義及 P=0／B=1。
- Physics 預設 5000 靴，建議 5000～8000；validation .20、calibration .10、augment .25。
- 訓練 loss：Winner 2.80、點數 1.20、張數 1.30、Rank .55、composition .45；不新增 Suit head。保留 affine、temperature、isotonic uncertainty 校準。
- XGB 搜尋 depth 2～4、trees 400～800、lr .015～.030、child 6～16、subsample .85～.95、colsample .75～.95、L1 .10～.60、L2 8～18。原 320-tree 模型只作比較 baseline。
- sample weight：41–50 手 1.05～1.10，51–70 手 1.20～1.30；低 uncertainty 最高 +18%，規律最高 +10%（預設 +5%）且高 uncertainty 停用。正規化保留均值 1、上下限 .35／1.50。
- 依序比較 Brier、hit/bet、EV/bet、Skip 變化；pattern hit 僅報告。校準採 chronological whole-shoe probe，先 overall Brier，另檢查 tail hit／EV；不得用 holdout 選模型。
- promotion 對實際 production 完整 runtime 作 chronological whole-shoe 回放。hit／EV 採各自不退步的較嚴格閘門、Brier 惡化≤.0015、Skip 上升≤5pp；Skip 可下降，但不以品質退步換出手。

```text
python physics_feature_extractor.py train --shoes 5000 --validation-fraction .20 --calibration-fraction .10 --augment-ratio .25 --output candidate_physics.joblib --browser-output candidate_physics.json
python xgb_final_probability.py train --input REAL_TRAIN_ROWS.json --physics-model candidate_physics.joblib --pattern-weight .05 --smoothing-profiles responsive_v1 --decision-policy-profiles hard_ev aggressive_entry_v2 --output candidate_final.json --report-output candidate_training_report.json
node evaluate_macro_replay.js REAL_HOLDOUT.json report.json PRODUCTION_DIRECTORY --candidate-model candidate_final.json --candidate-physics candidate_physics.json --entry-profile aggressive_entry_v2
node tools/no_pf_acceptance.js report.json
```

`--ci-smoke` 只能生成受阻候選作機制檢查，不能直接發布。替換 production 模型前，還需 `tools/verify_candidate_promotion.js` 核對已測模型／runtime／config 與當前 production SHA-256；測試的 entry／Macro profile 必須與將部署的預設相同。要部署 aggressive 預設，先在候選分支更改 JSON `default_entry_profile`，再重跑真實 holdout，不能套用其他 profile 的報告。

沒有真實資料時只完成程式、參數、單元測試與合成機制回放；不更換 production JSON／joblib，不宣稱已完成有效重訓。

