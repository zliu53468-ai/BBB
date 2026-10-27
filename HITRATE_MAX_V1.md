# 命中率優先配置 hitrate_max_v1

此配置以 Physics + PF500 為核心，僅加入小幅、同向的因果馬可夫先驗，接受出手減少。**目標挑戰 52–53.5%，54% 為樂觀上限，無法保證。** 尚無獨立真實 B/P/T hold-out 支持上述數字，不能把它寫成預期命中率或「最強組合」的已驗證結論。

## 實際路徑

B/P/T → Physics 213→48 → 原 PF500 混合 → 輕量馬可夫 → 原 Macro／Clip／決策 EMA → 補牌完成後無條件 EV → 原資格 + 新嚴格過濾 → B/P/Skip。

Core／57D XGB 仍會計算診斷，但其輸出不混入機率、不否決、不翻向；`xgb_influence` 實際固定 0，即使誤填其他數字也不採用。沒有重訓或更換任何權重，也沒有修改 PF 核心抽樣。

PR72 沒有獨立第一層 EMA journal；因此本次不新增或加重平滑，保留既有 Macro、Clip 與決策 EMA 的公式及參數。`physics_pf_direction` 是馬可夫之前的第一層 EV 偏好方向，與最後 EV 方向核對；不把高勝率外觀當成正 EV。

## 馬可夫規則

- 主要狀態為最近兩手 B/P/T，樣本不足時退回一階。和局作為狀態與結果保留。
- 只掃描預測前同靴已知局序，最多 200 手；沒有持久狀態，返回／重算／跨靴均由傳入局序重建。
- 狀態至少 12 個已完成轉移。以當局 Physics+PF 無條件 B/P/T 作 24 份 Dirichlet 先驗，減少小樣本波動。
- 同向且先驗 EV gap ≥0.003 才混合：`mixed=(1-w)*PhysicsPF+w*Markov`；預設 w=0.08，程式硬 cap 0.12。EV 方向與 B/P 機率方向均須同向；反向或均衡一律忽略，即使歷史看似很強也不可翻向。
- 三類機率一起混合、保留和局質量；不把馬可夫與 PF 當作兩份獨立證據。缺測／樣本不足時權重為 0。
- `markov.enabled=false` 完全取消先驗混合，機率回到純 Physics+PF 路徑，但命中率優先的嚴格過濾仍保留。

## 正式資格（全部成立）

沿用舊候選及信心／EV 要求，再加嚴格門檻，不救回舊候選 Skip。新模式不採用 Volume Guard 的 band/EV relief，也不採用最終 band relief；保留原 window/minHistory/expectedCorrectFloor 的衡量與診斷，沒有最低出手率要求。

| 階段 | 最低 abs(EV_B−EV_P) |
|---|---:|
| ≤20 | 0.012 |
| 21–45 | 0.009 |
| ≥46 | 0.007 |

另須 noise ≤0.65、Physics uncertainty ≤0.65、PF quality ≥0.60、最佳 EV >0.004、第一層與最終 EV 方向一致，且原始正式資格成立。缺少 PF 品質或模型資料會 Skip。設定只能收緊上述安全上限／下限，不能填入較寬數值繞過。

新模式 EV 使用完成補牌後的無條件機率：莊 `.95*pB-pP`、閒 `pP-pB`、和局退回；沒有改變原有標準莊佣金 5%。原 PR72 條件 EV 的舊行為只在完整回滾時恢復。

一般模式保留 2pp；不觀望測試仍不登記正式下注。完全五五波顯示「均衡」，不再固定選莊；其他時候依實際機率方向，不強制莊閒交替，也不保證方向出現頻率均衡。

## 設定與回滾

所有新參數在 `cardcount_profile.json.hitrate_max_v1`，schema_version=1。完全回滾：只把其 `enabled` 設成 false，恢復 PR72 的預測與正式資格；`validation/hitrate_max_rollback.json` 固定多段歷史的 PR72 機率、EV、候選、出手、平滑及顯示結果供比對。

`markov.enabled=false` 只關馬可夫，不關嚴格選擇性。部署設定修改後應刷新頁面，必要時更新腳本／設定快取版本。程式回滾也可 `git revert <此次提交>`。

開通碼、30分鐘限制、Frozen Core、模型權重、全背景與點數欄位移除狀態不變。

## 診斷與驗證

預測及匯出快照的 `hitrate_max` 包含 Physics+PF 機率及方向、馬可夫方向／樣本數／階數／實際權重／不採用原因、最終 EV、分段門檻、品質／noise／uncertainty、被擋原因、Volume Guard 衡量與正式出手方向。XGB 輸出另列診斷，影響力固定 0。

```sh
node test_hitrate_max.js
node test_pr56_restore.js
node test_bbb_button_only.js
node test_final_probability_runtime.js
node test_middle_late_upgrade.js
```

舊分段平滑回歸測試會關閉新入口，單獨確認原平滑沒變；新測試另驗證嚴格入口。五靴因果機械回放見 `validation/hitrate_max_smoke.json`，只有模擬資料、尚無真實 hold-out，不能當作實際下注命中率提升證據。更少出手不必然更高命中；後續需用獨立整靴真實資料檢驗命中率、EV/bet、Brier、出手數與分階段穩定性。

本次五靴機械回放共 345 次預測，舊版出手 113、新版 68（減少約 39.8%），馬可夫實際啟用 83 次。預測顯示方向為 B 216、P 105、均衡 24，沒有刻意調成各半。兩版在這個小型模擬樣本均為負收益；不據此聲稱命中率或 EV 改善，也不以此選參或晉升模型。
