# 最新單局重建粒子／Physics 輕量輔助

基準：main c0811e25（PR75）。這是未驗證命中率的推論模式，不是完整模型重訓。

## 資料流

最新一局 B/P/T → 全新 10,000 粒子 → 符合該勝負的合法四／五／六張抽補牌假設 → 扣除假設用牌 → 模擬下一局 → Physics 小幅同向輔助 → 既有 Clip／EV／信心與品質資格 → B/P/Skip。

PF 僅取最後一局；Physics 的 213D 編碼也僅接收這一局，不重播整靴。Physics 拓撲、權重、Frozen Core 檔案及 57D 模型均不改。Core／XGB 保留診斷，不參與方向。馬可夫、Macro 方向修正、跨局決策與顯示 EMA、本機校準輸出、Volume 歷史放寬在此模式旁路，避免舊紀錄反向影響推論。原有結果紀錄可繼續供匯出，不代表它參與新模式運算。

## 設定

`cardcount_profile.json.latest_hand_pf_v1`：enabled=true；version=LATEST_HAND_PF_PHYSICS_LIGHT_V1；physics_weight=0.08；max_physics_weight=0.12；max_delta=0.004。Physics 只有同向且品質通過才提供小修正；可信強反向只會禁止正式下注，不翻轉 PF。

先驗明確採「最新一局之前為全新八副牌」：不假裝知道整靴剩餘牌數或深度。這個先驗不能當成真實牌靴重建。只輸入 B/P/T 時僅三類觀測，重複相同結果將得到相同估計；可能長期偏莊，不隨機翻向，也不保證提高命中率。

## 點數與診斷

既有按鈕介面不新增欄位，不把勝負編碼 1/0 當真實點數。若外部呼叫確實取得最終點數，可用 `applyFinalPrediction(history, null, {player:3, banker:7})`；只接受整數0–9，須與最新勝負相符，無有效條件樣本時失敗觀望。目前網頁按鈕未傳入實際點數。

粒子診斷新增 observation_scope、observed_points、prior、latest_hand_reconstruction。後者記錄最新一局假設的張數、雙方點數分布與預期牌階消耗；既有 point_card_forecast 是下一局分布，兩者不可混用。匯出快照新增 latest_hand_pf 與既有 particleDiagnostics。所有未知點數／牌階都是條件假設。EV 仍為莊 .95×pB−pP、閒 pP−pB，和局退回；沒有改補牌規則或佣金。

## 驗證與回滾

執行 `node test_latest_hand_pf.js`。可設定 BBB_BASELINE_ROOT 指向PR75原版資料夾，額外逐項比對停用後的機率、方向、EV及受保護檔案。測試涵蓋 B/P/T 前綴隔離、重算、空輸入、實際點數條件、矛盾點數失敗觀望、跨局紀錄與不觀望按鈕不污染機率。

既有正式機率與按鈕測試明確使用原版關閉模式，新測試驗證新模式。原版 test_first_layer_ev.js 與 test_post_draw_ev.js 已在基準本身失敗（缺少舊介面），本次不修改無關舊介面。

回滾只需設 latest_hand_pf_v1.enabled=false 並重新載入；恢復PR75原有預設，也保留 ?pf_primary=1 入口。未修改開通碼、裝置綁定、30分鐘限制、畫面2pp門檻與模型權重。未進行命中率晉升驗證，不宣稱優於原版。
