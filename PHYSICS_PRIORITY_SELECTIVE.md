# 純算牌優先｜不確定性閘門與連錯保護 v1

**2026-10-29 驗證結論：功能完成，晉升未通過，保留候選，不合併正式main。** 已完成獨立10,000靴／802,717手。完整[Markdown報告](validation/physics_priority_v1/REPORT.zh-TW.md)、[JSON統計](validation/physics_priority_v1/statistics.json)、[分布圖](validation/physics_priority_v1/streak_distribution.png)與稽核均保留。

|指標|新版本|舊正式版|
|---|---:|---:|
|出手率|7.1224%|14.6093%|
|非和推薦命中率|50.3250%|50.2138%|
|平注扣佣EV／推薦|−0.9581%|−0.6589%|
|靴內最大連錯|7|12|
|按生成順序跨靴、不清零最大連錯|16|18|
|跨靴每千次非和推薦的三連錯窗口|125.518|121.841|

新版本確實選擇莊38,578次、閒18,595次，不是全莊。命中率差異的整靴配對區間跨0；每推薦EV點估計變差，出手率也遠低於20–35%。靴內連錯下降不能冒充全程最大連錯7次，正規化全程三連錯也沒有改善。原始逐局預測存於`research/reports/priority_v1_holdout/predictions.jsonl`（大檔不提交），其SHA256與輸入、程式及參數鎖定雜湊都寫進統計／稽核。

下一輪優先在開發靴校準noise／ESS／quality與實際誤差關係，研究切14張時原PF假設牌組耗盡的處理，再做點數模式的獨立大樣本。不能直接放寬EV或改掉不利統計來湊出手率；本次holdout已揭露，往後只可當研究資料，新候選需另一組全新holdout。

## 先規劃、後驗證

基準為 GitHub main `c6f4a0afbf9fc9de3806ae662319471daadfb405`。目標為有推薦時更穩，20–35%是建議出手率，不是強制配額，也不是勝率保證。

1. 保留 Frozen Core、213→64→32→48 Physics、57D、模型權重、PF500原始抽樣及既有授權／30分鐘／本機校準程式。
2. 加入獨立、可關閉的 `physics_priority_runtime.js`，正常模式以 Physics+PF 機率、因果校準及聯合閘門決定B/P/Skip。路單、馬可夫、Core、XGB方向權重均為0。
3. 恢復獨立點數後驗庫 `point_conditioned_pf.js`，只處理已公布的閒／莊最終點數；有效點數權重65%。沒有點數則維持Physics+原PF殘差。
4. 先200靴煙霧測試，再鎖定來源／參數／資料雜湊，以不同seed跑10,000靴。不得用holdout調參；未通過不自動合併或部署。

只有B/P/T時，Physics和PF是在推估隱藏牌組的條件期望，並非讀到真實A–K張數。Frozen Core雖保留計算，不能替最終方向投票；Physics既有213D輸入不改，故不能宣稱完全沒有歷史統計資訊。

## 開關與參數

完整設定在 `cardcount_profile.json.physics_priority_v1`，有schema及version，所有infer參數可被測試覆蓋。

|參數|1–20手|21–50手|51手以後|
|---|---:|---:|---:|
|最低 `abs(PB−0.5)`（嚴格大於）|0.060|0.018|0.015|
|noise上限|0.66|0.70|0.72|
|Physics uncertainty上限|0.66|0.70|0.72|
|ESS/N下限（越高越好）|0.60|0.60|0.60|
|粒子品質下限|0.62|0.62|0.62|
|基礎扣佣EV下限|0.015|0.010|0.006|
|基礎EMA最新權重|0.38|0.52|0.62|

EV閘門使用 `max(新階段EV下限, 原policy activation EV, 原Physics activation EV)`，不以降低EV門檻湊出手率。原EV的40/50界線仍作更嚴格的下限；新機率、noise、ESS及EMA階段統一20/50。這個保守交集可能令出手率遠低於20%。

EMA α = clamp(階段α + 0.15×算牌強度 − 0.10×noise, 0.22, 0.80)。只讀同靴已結算的預測；當EMA可能翻轉當前Physics符號時重設為當前校準值。Dynamic Clip保留。Macro EMA仍更新及匯出，但不把路單方向加進新機率；原後段決策EMA仍計算作診斷，其結果不再二次混入新推薦，避免雙重平滑。

本模組使用獨立的因果殘差校準：至少30個非和局，window60、prior_strength120、decay0.98、delta上限0.004，且不得跨越當前Physics符號。既有本機校準客戶端、資料隔離與結算入口保持，用於原顯示／回滾，不再疊到新機率上。

## 點數模式

選填已完成局的閒／莊點數，然後按對應B/P/T按鈕。現有MACRO輸入驗證會拒絕不符結果或非法點數。兩個點數都有值才啟用65%點數後驗混合，上下限硬鎖60–70%。只選張數不啟動點數權重。未來局點數不能使用。

點數庫來源為既有版本 `84dc6c2`，以獨立全域名稱載入，沒有取代 `__BGS_PARTICLE500__` 或修改其原始抽樣。點數後驗採十種點值、500粒子；似然是獨立抽牌DP的有限牌靴近似，條件牌張從假設牌組不放回抽取。0點包含10/J/Q/K，不能識別實際牌階。

新庫只在已完成局追加時更新；返回或修改點數時從固定seed回放，跨靴重置。失敗或缺少有效ESS時強制Skip，不能把fallback稱作成功點數後驗。這是粒子後驗更新，不是神經網路逐局重訓。

## 連錯保護與Volume Guard

- 推薦連錯3次，機率差門檻乘1.20。
- 推薦連錯5次，冷卻接下來9個實際牌局；冷卻後若再錯且仍≥5連錯，再冷卻9手。
- 推薦連勝≥3，機率差門檻最多放寬3%；EV/noise/ESS閘門不放寬。
- Skip與和局都不清除「連續非和推薦」的勝負紀錄。冷卻不會把歷史連錯洗掉。
- Volume窗口16、至少8筆；階段參考率5%/30%/35%，超過時只將band提高10%，不做band／EV relief，不強制湊出手次數。
- 新靴清除所有本模組狀態；撤銷按歷史指紋刪除無效結算並因果回放。重複分析同一手不重複結算，儲存錯誤時Skip。

## UI與匯出

正常模式主方向顯示實際過閘門的B/P/觀望；同時保留莊閒機率與另列和局。新增高／中／低信心、是否達標、不確定性觀望／冷卻與剩餘張數區間。張數範圍按每局4–6張與未知燒牌2–11張推估，絕不標作實際牌階。

`physics_priority`稽核含 raw/calibrated/smoothed PB、alpha、EV、門檻、noise、ESS、quality、cooldown、點數權重、觀望理由、模型version。它會寫入原正式結算快照／分析匯出；`getPhysicsPriorityJournal()`另可查看因果結算日誌。正常模式2pp顯示下限仍在，新推薦門檻更嚴格，不能因畫面有偏向就稱作已達下注資格。

「全程不觀望實驗」按鈕仍走原實驗／顯示流程，不適用新選擇性閘門，也不列入正式下注。本次10,000靴報告只驗證正常模式的選擇性推薦。

## 可直接執行的驗證

```powershell
node test_physics_priority.js
node test_browser_calibration.js
node test_end_analysis_reload.js
python -m research.priority_generate --shoes 200 --seed 2026102910 --output research/data/priority_v1/smoke-observations.jsonl
node research/test_priority_integration.cjs
node research/test_priority_io.cjs
node research/priority_legacy_regression.cjs test_hitrate_max.js
node research/priority_legacy_regression.cjs test_experimental_no_skip.js
node research/priority_legacy_regression.cjs test_physics_forecast_first.js
node research/priority_legacy_regression.cjs test_macro_runtime.js
node research/priority_backtest.cjs research/data/priority_v1/smoke-observations.jsonl research/reports/priority_v1_smoke
python -m research.priority_report research/reports/priority_v1_smoke

python -m research.priority_generate --shoes 10000 --seed 2026102911 --output research/data/priority_v1/holdout-observations.jsonl
node research/priority_backtest.cjs research/data/priority_v1/holdout-observations.jsonl research/reports/priority_v1_holdout
python -m research.priority_report research/reports/priority_v1_holdout
```

需要numpy，繪圖需matplotlib；可用既有研究環境。本次實際環境Python3.10.11、NumPy1.24.3、Matplotlib3.7.5；沒有用Python ML庫重訓正式權重。`priority_legacy_regression.cjs`只在測試程序內將新flag關閉，不改原測試或設定檔。選用瀏覽器檢查另需Playwright／Chromium，可用`BBB_PLAYWRIGHT_MODULE`、`BBB_CHROME_PATH`指定本機路徑，需先在本專案根目錄啟動127.0.0.1:8898的靜態伺服器；此測試僅在隔離本機fixture展示畫面，不接真實授權服務。

生成器以`x`模式拒絕覆寫；collector鎖定來源／參數雜湊並支援完整靴續算。若來源或設定改變，必須使用新實驗目錄及新holdout，不能偷改已鎖定測試。

模擬為標準8副牌、揭牌後依點值燒牌、切14張，完整結束跨切牌的最後一手；沒有最多70手截斷。隱藏組成不匯入預測器。本次大樣本只輸入B/P/T，點數後驗另外作功能／回放測試，不能宣稱點數模式也完成10,000靴效益驗證。

舊版對照直接載入固定main正式JS及權重，完整呼叫原推薦／結算／Macro／本機校準及Volume Guard；不是僅用Direct Physics當舊版。離線adapter只快取相同PF後驗及等價的本機日誌I/O，未改模型、抽樣或數學；與原JSON存取版本逐欄比對486手通過。每靴清空狀態，old/new使用同一組已洗牌結果，推薦後才揭露當手結果。

輸出`REPORT.zh-TW.md`、`statistics.json`及`streak_distribution.png`。統計同時列：全部機會分母、非和推薦分母、平注1單位扣佣EV、最大推薦連勝／連錯、≥3/5/7極大段次數與平均長度、重疊窗口與每千次推薦頻率。冷卻不清零的推薦序列是主指標；遇Skip中斷的時鐘連敗只是輔助，不用來掩蓋連錯。

## 回滾與晉升

將`physics_priority_v1.enabled`設為`false`並重載，即走原推薦、馬可夫、EMA與顯示；選填點數UI隱藏，原PF不受影響。已累積的資料不會因切旗標變成「從未執行過新模式」；要做完全相同狀態的比較，應新靴開始或用獨立測試儲存空間。原開通碼與30分鐘授權不得清除或改寫。

功能通過不代表統計通過。晉升需檢查出手率、條件hit的整靴配對區間、相同時機固定莊對照、每推薦EV、推薦序列最大連錯與每千次三連錯。若不足，保持候選分支，不自動合併。獨立模擬通過後仍需真實前瞻驗證，不保證未來勝率或最大連敗上限。
