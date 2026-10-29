# 正式網頁雙核心實驗啟用

依使用者明確要求啟用未驗證版本。此次已接入正式applyFinalPrediction，不再只是Shadow檔案。

獨立Core A=Direct Physics＋sequential PF500（重用當靴後驗、撤銷/新靴重建），Core B=PR83 Markov＋一階輔助，融合使用現有fusion_candidate.json（Train選road_weight=.5、Validation temperature=1.2）。品質動態混合，沒有再跑原stateless PF10000或road.combine二次混合。融合B/P/T寫入主候選機率，繼續原Macro/Clip/EMA、EV/Volume Guard、方向一致性等資格。57D XGB維持診斷、不否決翻向。0.5pp及開通/30分鐘限制不改。

未重訓Physics/Core/XGB；模型promotion_approved仍false，實驗啟用不偽造已通過驗證。原7靴小樣本不足，未證實勝率提高，不能宣稱模型正式晉升成功。UI標記雙核心實驗預測。

cardcount_profile.json：road_primary_v1.vnext_live_v1.enabled=true。設false回PR83舊實際計算，原PF10000與road.combine重新生效；策略tag隔離新舊歷史。缺新模組/無效融合模型時Skip，不默默假裝仍跑雙核心。新模型資料採目前BPT輸入；實際牌值入口未新增至原UI。

node vnext/test_live_runtime.cjs驗證融合實際影響主機率、舊PF不重複執行、因果/撤銷、同靴狀態、EV/和局、快照及精確PR83回滾。未完成新正式整條流程大規模holdout。
