# 回復PR78＋無先驗的一階輕量輔助

正式主路徑回復PR78：3→2→1階支持數回退、原固定先驗24、Physics+PF有限校正、原EMA/Macro/Clip、EV與Volume Guard。PR79～82的分段/整靴自適應額外先驗、可靠度收縮、EV加嚴與五階/五級調整已從正式路徑移除。模型與授權/30分鐘邏輯不改。

新增first_order_aux_v1：以最新B/P/T為一階上下文，僅計同靴已公布轉移，各結果機率=count/n，不加先驗/偽樣本。這是未收縮的頻率估計，並不是另一套有資訊先驗的貝葉斯模型；不能把勝方按鈕當真實點數。

僅PR78主候選可用且一階支持>=12筆、B/P質量>0時，給2%輕量混合，條件莊率delta上限0.001（0.1個百分點）；不能翻轉主候選方向，保留和局質量。再走原Physics校正及正式風控，不用輔助單獨救活樣本不足的主路徑。

設定road_primary_v1.first_order_aux_v1：enabled=true、min_observations=12、mix_weight=.02、max_delta=.001；硬上限mix_weight .05、max_delta .002，min至少12。既有road.first_order_aux診斷隨快照匯出，含counts、無先驗機率、delta與原始候選機率。

關閉first_order_aux_v1.enabled即可精確回PR78計算；version策略tag僅為隔離舊狀態。全回PR78配置可使用764cdb8617c0bcb7253321c74041f6a037f2c25c中的配置。git revert本次合併會回PR82，不等於PR78。

node test_pr78_first_order.js 驗證一階count/n、同靴因果、無先驗、低樣本禁用、混合上限/不翻向、和局、PR78模組/runtime數值回滾及匯出；另跑原road、授權、結束分析與EV測試。未重訓，未證明命中率提升。
