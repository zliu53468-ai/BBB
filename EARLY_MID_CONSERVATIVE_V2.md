# 前中段更保守｜500靴快速修正 v2

**結論：防守限制已落地，但本次沒有證明前中段命中率提升；不自動合併／上線。** 承接PR73未合併候選116ffc9，正式main仍是c6f4a0a；這兩者不可混稱。Frozen Core、Physics／XGB權重、原PF500抽樣與佣金／補牌規則都未改。

## 開關、單位與漸變

`cardcount_profile.json.physics_priority_v1.early_mid_conservative_v2.enabled=true`。

「莊閒機率差」= `abs(PB−PP)`；既有 `probability_band` 是 `abs(PB−0.5)`，故差距門檻是band的兩倍。前版候選1–20局已要求12pp，不能把它誤認為2pp後反而放寬。

|項目|前1–30局|中31–45局|後46–70局|
|---|---|---|---|
|莊閒差門檻|1–20維持12pp，21–30平滑降至5.5pp，至少5.5pp|31–39由5.4pp逐步降至3.8pp，40–45維持3.8pp|46–50過渡至原3pp，之後維持|
|noise／uncertainty上限|0.55|過渡至0.62|過渡至原0.72|
|ESS/N最低|0.75|過渡至0.68|過渡至原0.60|
|粒子品質最低|0.70|過渡至0.65|過渡至原0.62|
|PF混合上限|硬性≤0.08|0.08平滑升至0.14|平滑恢復0.16|
|決策EMA α|基礎0.28，依訊號調整，硬性≤0.40|基礎平滑升至0.50，上限逐步放寬|恢復原基礎0.62／原上限0.80|
|PF等外層Residual對Direct Physics PB的修正|硬性±0.015以內|漸變至±0.025|恢復±0.05上限|
|新模組本機校準|關閉；最低30個有效非和樣本|樣本足夠才由0漸增至正常權重|正常權重；仍保留原delta上限及不翻向限制|

所有曲線都在設定的`curves`中，可調且線性插值，30→31、45→46不硬切。中段過渡初期比3.8pp更嚴格，避免第31局突然放鬆。前30的PF／alpha／Residual限制另有硬上限，不能被曲線參數意外放大。

為了不讓既有65%點數PF繞過前30局的8%總PF預算，**前30局暫不使用點數後驗混合**，但保留已輸入的點數；31–40局由0漸變恢復至65%。只有B/P/T時仍使用Physics＋受限PF殘差，不虛構牌階。

獨立校準及舊顯示校準客戶端在v2啟用時都要求至少30個有效非和樣本；新模組前30局修正為0，31–40局漸進恢復。更新政策版本會重建本模組EMA／校準日誌，不沿用不同政策的殘差；B/P/T歷史、點數紀錄、開通碼及30分鐘授權不刪除。

三連錯維持門檻×1.20，五連錯維持冷卻9個實際牌局；觀望與和局不清零連錯。v2取消連勝放寬，確保前中段不低於設定下限。EV仍取新門檻與舊正式／Physics門檻的較嚴者，不為湊出手率而放寬。

畫面保留Physics機率；未過機率差顯示「訊號不足，建議觀望」。原「不觀望實驗」仍是獨立的非正式測試，不能當作已過正式出手門檻。

## 實際配對測試

全新seed2026102921，500靴、35,000手，每桌最多70手。前50／60手為預先指定的同靴因果前綴，不算額外獨立樣本。只輸入已公布B/P/T，沒有實際牌階或當手未公布結果；原main、前版候選、新v2三路逐局配對。

|範圍|前版候選出手率|v2出手率|前命中率|後命中率|
|---|---:|---:|---:|---:|
|1–30|3.280%（492次）|0.167%（25次）|49.095%|37.500%（僅24次非和，樣本不足）|
|31–45|11.253%|8.307%|53.088%|50.893%|
|1–45|5.938%|2.880%|51.621%|50.342%|
|46–70|8.648%|8.840%|50.717%|51.151%|
|全部70手|6.906%|5.009%|51.216%|50.853%|

跨靴連續推薦最大連錯13→10，靴內最大仍6→6；EV／推薦由+0.530%降至−0.240%。前30局推薦數下降94.9%，但前中段命中率點估計下降，不能稱作更準；所有配對命中率差95%區間都跨0。少出手、降低觀察到的最大連錯不等於識別能力提升，亦不保證未來連錯上限。

完整[Markdown報告](validation/early_mid_conservative_v2/REPORT.zh-TW.md)、[JSON統計](validation/early_mid_conservative_v2/statistics.json)、[鎖定參數](validation/early_mid_conservative_v2/protocol.json)。本次只是500靴快速驗證，不能推廣為萬靴或真實場景收益。

校準回歸檢查補上「總開關關閉時不得沿用v2最小樣本限制」後，用同一批500靴再驗證；35,000手的三路推薦與數值逐欄相同。這仍然只有500靴，不冒稱1000靴，也沒有依測試勝率改參數或挑牌靴。

## 檔案與重現

- `physics_priority_runtime.js`：`limits`／`interpolate`曲線、`infer`階段／alpha／校準限制及政策日誌隔離。
- `final_probability_runtime.js`：前30PF預算、Residual上限、點數延後、本機校準最低樣本及觀望提示。
- `cardcount_profile.json`：可配置曲線與v2開關；`auth-client.js`／`index.html`只更新載入版本，登入、綁定、30分鐘及清除程式不改。
- `research/early_mid_*`：固定500靴、main／v1／v2比較、50／60／70手評估及統計；`research/priority_runtime.cjs`增加指定已提交版本載入。

```powershell
node test_physics_priority.js
node test_early_mid_conservative.js
node research/test_early_mid_runtime.cjs
node research/priority_legacy_regression.cjs test_browser_calibration.js
node test_end_analysis_reload.js
python -m research.early_mid_generate --output research/data/early_mid_v2/observations.jsonl
node research/early_mid_backtest.cjs research/data/early_mid_v2/observations.jsonl research/reports/early_mid_v2
python -m research.early_mid_report research/reports/early_mid_v2
```

產生器及回測拒絕覆蓋既有結果。首次本機路徑若已存在請用新目錄；本次資料已揭露，之後只可當開發資料，下一候選要用新seed。

只關閉`early_mid_conservative_v2.enabled`可回上一候選的決策行為；再關閉`physics_priority_v1.enabled`可回原main路徑。切換政策時只重建本模組狀態，保留授權與原歷史。若要比較相同初始狀態請新靴或隔離瀏覽器資料。
