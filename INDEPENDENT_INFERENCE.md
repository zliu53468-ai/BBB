# 每局獨立推論 v1（保留 B/P/T 特徵）

依使用者確認：只能提供B/P/T，保留現有特徵並做每局獨立推論。本模式不是每局完整重訓、不是無歷史算牌，也不保證勝率或EV提升。

## 保留

- app256forward.js、app256continuation.js完全不改；原本就是frozen prior、update_ab=false，沒有可供清空的逐局學習A/b。
- Physics 213→64→32→48、57D特徵順序/定義、直接P=0/B=1分類、兩個正式模型權重完全不變。
- 當下完整B/P/T歷史仍用來重建Core、Physics與PF特徵。PF每次重建500粒子，沒有沿用上一手粒子狀態。
- Dynamic Clip、noise/uncertainty gating、三階段EV及最高下注限制不變。2pp預設方向顯示與不觀望測試按鈕保留；測試不新增正式下注。
- 開通驗證不變。auth-client.js只改final runtime的快取版本查詢字串。

## 每次推論不使用的狀態

cardcount_profile.json的inference.mode=independent_v1（version=1）預設啟用：
1. 決策EMA直接採本手已Clip機率，alpha=1，不讀上一手快照。
2. Macro只保留觀測/診斷接口，不對機率產生方向修正。
3. 本機校準不套用、不建立新校準ticket、不沿用殘差修正。
4. Volume Guard保留接口，但不讀歷史出手率、沒有基於過往結果的band/EV放寬；仍由本手EV/noise/band資格決定B/P/Skip。
5. 不刪除B/P/T觀測、訓練匯出或結算紀錄，這些作稽核用途；它們不再作為EMA/Guard的推論先驗。

碼中「// 【極致勝率優化：單局獨立】」只作使用者指定標記，不是成效宣稱。每局重新推論與模型權重重訓不同；完整重訓仍需訓練資料與holdout驗證。

## 首局與Skip的定義

沒有固定等待7/12局或先累積下注紀錄才能通過的新增條件。第一次可分析時即可依當手機率與原有EV資格出手。
原本UI的歷史輸入要求不變；沒有實際A-K時不能憑空產生真實牌組狀態。API可對空歷史求模型先驗，但不是知道第一手結果。
未達EV/機率差/noise條件仍可Skip，沒有強制最高下注或把Skip改成B/P；不觀望測試依原設定只作畫面測試。

## 診斷與回滾

finalProbability.inference_mode記錄independent_v1或legacy_stateful。Macro/EMA/校準等模組沒有刪除。
將inference.mode改為legacy_stateful即可恢復原路徑；或把validation/cardcount_middle_late_v2_baseline.json完整還原為cardcount_profile.json。回滾時以新靴比較，避免混用不同模式下的舊決策EMA快照；不要清除授權或真實觀測資料。完整程式回滾用git revert本次提交，不force-push。

## 驗證與限制

node test_independent_inference.js
node test_final_probability_runtime.js
node test_macro_runtime.js
node test_physics_forecast_first.js
node test_cardcount_profile.js
node test_middle_late_upgrade.js

新測試將同靴舊快照、Macro紀錄與校準狀態注入，確認相同B/P/T歷史得到逐值相同的機率、EV、方向與57D；重試及上一手結算也不改變當手結果。舊EMA/Volume Guard測試明確使用legacy_stateful，以保留回滾路徑的數學回歸。
未做真實獨立holdout，因此不能宣稱命中率、EV提升或Skip下降。停用歷史Guard的放寬可能使某些邊界局Skip增加；這是選擇不依賴過往出手率的取捨。
舊完整重訓/參數工具刻意凍結舊production hash，不可把舊成績當成此模式驗證。其CI用舊提交重建凍結來源檢查；若要重訓/晉升新版，需另凍結此模式的production對照與未使用過的完整牌靴。

本次更新目標只有BBB；不會自動同步到BGS0.3V.1或其Render服務。
