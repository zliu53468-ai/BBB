# 牌路主導／算牌微調實驗版

依本次授權將BBB預設從Physics主導改成牌路／馬可夫主導；這是實驗部署，不宣稱命中率提升。基準main為PR75 c0811e25。未合併PR76，也沒有搬入獨立Sites面板的HMM或Logistic。

## 運算順序

本次同靴B/P/T → 牌路二階馬可夫候選（不足回退一階）→ 原Direct Physics213→48＋PF10,000重建 → 算牌有界微調 → 原Clip／Macro／決策EMA → Core／57D XGB診斷 → 原最終EV與一致性／噪音／Volume資格 → B/P/Skip。

牌路上下文不足12次時，明確以原Physics＋PF暫代，畫面標示「算牌暫代／牌路樣本不足」。計算失敗仍走既有失敗觀望邏輯，不以缺失輸出冒稱成功。只有第一層主要機率來源與先後權限調整，並非把幾個script重新排序就冒稱升級。

## 設定及權限

cardcount_profile.json新增road_primary_v1：enabled=true、order=2、window=200、min_observations=12、prior_strength=24、prior={B:0.45,P:0.45,T:0.10}、physics_weight=0.10、max_physics_weight=0.15、max_delta=0.004、max_physics_uncertainty=0.70、accuracy_validated=false。

prior是明確設定的對稱莊閒參考先驗，沿用既有中性介面比例，不是經訓練或真實牌局校準的機率。牌路機率由本次上下文轉移計數加上此先驗取得；不把「總莊閒比例」直接當下一局機率。Physics／PF只對條件P(B)作10%校正，硬上限15%、單次最多0.004（0.4個百分點），且不能單獨翻轉牌路方向。Physics不確定性>0.70不修正；牌路Balanced不被物理強制推成方向。這些是候選設定，未證明最佳。

保留Physics＋PF的和局質量；同一組校正後無條件機率計算EV：EV_B=.95×pB−pP、EV_P=pP−pB。原手續費、和局退回、補牌規則不改。Core/XGB僅診斷，57D診斷仍用原Direct Physics分布。後段Clip、Macro、EMA或校準仍可能改變幅度或讓最終資格觀望，不能把第一層10%稱為最終輸出的全流程固定佔比。

原馬可夫8%加成在新路徑不再執行，避免對同一序列重複加權。Frozen Core、Physics／XGB模型檔、PF核心、Macro程式、本機校準程式均未改。授權與30分鐘邏輯不變；auth-client僅新增模組載入與版本字串，index僅更新auth-client快取版本，版面不變。一般顯示仍為2pp。

## 診斷與隔離

新增road_primary至預測與匯出快照，包含road候選機率、上下文計數、先驗、物理機率、有效修正權重、delta與fallback原因。execution_order先記road_primary_candidate，再direct_physics／particle_filter／physics_pf_adjustment，沒有第二次markov_prior。

新舊模式的EMA、Volume與校準以road設定版本tag隔離，避免舊模型的本機校準／下注紀錄混用。資料仍會使用本次已輸入的完整B/P/T，每次PF從零重建，不沿用上輪粒子。開始新靴請依原結束／清空流程，不假稱只用最新一局。

## 回滾

將road_primary_v1.enabled設false並重新載入，即回到PR75的Physics主導＋原馬可夫；測試數值與保護檔案hash已和c0811e25核對。明確網址?pf_primary=1仍可進原PF主導試用入口，不與牌路路徑疊加。完整程式回滾可對本次合併提交執行git revert -m 1 <merge_sha>，再部署。

## 驗證

新預設用node test_road_primary.js、node test_road_primary_runtime.js測試：先牌路後算牌、方向主權、限幅、不重複馬可夫、XGB隔離、和局與EV、缺模組觀望、匯出、fallback、跨呼叫重建，以及停用後回到原版。

既有PR75測試以明確的test-only legacy_profile_fixture.cjs關閉road旗標來驗證回滾路徑；新預設有獨立CI工作，不靠舊模式測試代替。fixture不載入正式網頁，也不修改任何模型權重。validation/road_primary_rollback_pr75.json保存由原c0811e25實際產生的回滾數值及保護檔案SHA。

這次沒有完整真實hold-out或大規模策略比較。功能正確不代表下注命中率或EV提高；牌路模型可能更受序列外觀影響，也可能更敢出手，均需事前紀錄驗證。
