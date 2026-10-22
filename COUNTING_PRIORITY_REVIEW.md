# 與上一版對照及修正

| 要求 | 上一版 `7f44d37` | 本次修正 |
|---|---|---|
| PF 移除 | 已完成正式載入與 runtime 移除 | 保留；新增 `PF_REMOVAL.md` 與回滾說明 |
| 原始模型架構 | Frozen Core／權重／維度未動 | 繼續保持不動，與最初 main 逐檔核對 |
| 算牌訓練優先 | Physics 訓練配置未補強 | 新增 Winner／點數／張數小幅權重調整，不動 rank／拓撲 |
| 規律 sample weight | 預設 .35、允許到 1，太大 | 預設 .05、最多 .15；上限 1.15，且受 Physics quality 控制 |
| 選模排序 | 混合加權分數，非嚴格 Brier 第一 | 改為依序比較 Brier、hit/bet、EV/bet、EV/row、Skip delta |
| Skip 驗收 | 強制至少降低 1pp | 預設允許下降，增加最多 5pp；不犧牲既有品質閘門 |
| 規律 soft 閘門 | 沒有明確 `base_eligible` | 先固定算牌合格判定；不合格／高 noise／弱訊號的 bonus 強制為零 |
| 規律段報告 | 尚未加入 | 長龍、單跳、雙跳、轉折、其他；只報告，不作硬晉升門檻 |
| 真實重訓與成效 | 未完成 | 仍需獨立訓練／校準／holdout 資料，未宣稱完成 |

## 算牌訓練參數

Physics 的原始 Winner 權重已是 **3.25**，並非指令舉例的 2.50；因此維持 3.25，不降成 2.6～2.8。新增 `--winner-scale`、`--points-scale`、`--cards-scale`，各允許 1～1.10，預設 1，保留現行值。Rank／Suit 不新增、不加權。原有概率校準、uncertainty mapping、whole-shoe 外部切分均保留。

```
python physics_feature_extractor.py train --shoes 5000 --winner-scale 1.05 --points-scale 1.02 --cards-scale 1.02 --output candidate_physics.joblib --browser-output candidate_physics.json
python xgb_final_probability.py train --input TRAIN_ROWS.json --physics-model candidate_physics.joblib --pattern-weight 0.05 --output candidate_final.json
```

Physics 訓練建議 4k～8k 靴，工作流程由 2500 改為 5000 靴；未在本機執行大規模重訓。57D 的既有中後段／低 uncertainty sample weight 保留；規律權重只對既有 S/X 特徵作對稱輕微加權，8 手以前為零，24 手達完整 evidence。`--pattern-weight 0` 停用，超過 .15 拒絕。

XGB 增加一組仍保留 `reg_alpha=.75`、`reg_lambda=16` 的搜尋候選。選模、平滑與 EV 門檻搜尋改為有先後順序的比較；低順位數字再好也不能抵銷更差的 Brier。原有命中率、實現 EV、Brier 非退步品質保護仍須通過。資料保持 chronological whole-shoe split，不新增 row 隨機切分。

`--min-skip-reduction` 預設改為 0：可降低 Skip，也允許增加至 +5pp；只有明確指定正值時才要求該降幅。清除舊政策中「必須增加 Skip」的限制。規律段指標不參與篩選。

## 決策閘門

`base_eligible` 完全由原有 Physics candidate、Core／57D filter、final_p、EV／confidence band、noise penalty 和 Volume Guard 結果計算。先固定結果，再計算 `pattern_soft_bonus`。

可選配置：`final_probability_model.json.decision_policy.pattern_soft_bonus`，預設停用，示例 `{"enabled":true,"ev_relief":0.002,"band_relief":0.001}`。本次不修改正式 JSON。

強制條件：`base_eligible=true`、noise ≤ .60、`abs(final_p-.5) ≥ .02`、信心與 EV 為正，且存在標籤；否則 bonus=0。EV relief 上限 .004，band relief 上限 .002，非法值歸零。標籤無 B/P 方向權限。

加成只記錄在已合格出手的 soft 門檻欄位，不回流到出手、方向、機率或注碼計算。因此 **開關 bonus 不可能把 Skip 變成 B/P，也不會減少 Skip**。這是遵守「只有已達資格才允許加成」的必要結果。降 Skip 應由算牌重訓／校準、EV／noise／band 的獨立驗證改善達成。

原有三階段 EV、Dynamic Clip、Macro EMA、noise penalty 和 Volume Guard 不移除、不以規律覆蓋。平滑參數仍由上一版的 `smoothing.progress_alpha_anchors` 調整，JS／Python 驗證保持一致。

## 成效與限制

`validation/counting_priority_smoke.json` 是與原 main 比較的 12 靴／840 手固定 seed 合成回放，只能檢查機制。既有模型權重與正式配置不變，應維持上一版出手與機率；本次沒有在該資料上調參。

真實 holdout 驗收仍可用 `evaluate_macro_replay.js` 和 `tools/no_pf_acceptance.js`。報告新增所有非和局分析的 Brier 及 pattern segment 指標。驗收保留命中率／每注實現 EV 非退步、Brier 容差、no-flip、最低樣本要求；Skip 上限修正為 +5pp。pattern 段成績不作門檻。

上一版已觀測到合成回放命中率與實現 EV 下降；不能把本次程式修正說成已解決該退步。需要訓練候選與真正獨立的驗證，才有成效結論。尚未推送／部署到 GitHub main。

驗證結果：55 項 Python 測試通過；主 runtime、Macro runtime、按鈕介面與驗收閘門測試通過。新增測試證明不合格／高 noise／弱訊號的規律加成為零，開關規律加成不改 B/P/Skip、final_p 或注碼；+5pp 以內可通過 Skip 限制，超標或命中率／EV 退步則拒絕。`validation/counting_architecture_audit.json` 保存原始模型／Core 雜湊與特徵合約核對。840 手合成回放的出手、命中率、實現 EV 與上一版逐項相同。
