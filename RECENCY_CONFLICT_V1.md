# 舊證據淡化＋近期分歧閘門測試版

基準PR86。只改上下文加權牌路的歷史證據權重，新增正式出手分歧veto；Physics/PF核心、Frozen Core/模型權重、原Clip/Macro/EMA、佣金/和局、EV/Volume Guard公式、0.5pp方向顯示與開通/30分鐘限制不變。未新增模型、未重訓。

## 固定起始參數（未依報告調優）

road_primary_v1.history_adaptation_v1.enabled=true；count_decay=.97（每手保留97%，半衰約23手）、score_decay=.96（半衰約17手）；short_window=16；short_min_context=6；min_extra_history=8；short_prior_strength=6；direction_margin=.015；disagreement_gap=.06。count/score decay硬範圍.90～1。

每個上下文保留raw支持數，原12筆主候選門檻不降低。B/P/T轉移counts用lambda^(觀測年齡)加權；固定先驗強度仍24，所以舊證據較弱時相對先驗影響較大，未另加自適應先驗。權重支持項使用weighted n，raw n只作足量閘門。模型評分逐局淡化，仍先用與上一刻實際發出相同的事前機率評分，再納入新結果；惰性時間戳衰減與重播可重建相同狀態。

## 分歧閘門

取得最後16手內、以上一手為上下文的一階轉移；至少6筆非和轉移且整體prefix至少多8手才評估。短窗口用既有B/P參考先驗、強度6作保守診斷，不作新主候選。與當前最終（Clip/EMA後、含Physics/PF）條件莊率比較。雙方在.5兩側且各離.5至少.015，彼此差>=.06時，正式下注強制Skip；不把機率/方向翻向，不假定落後方必然補償。

這個6筆門檻僅能阻止下注，不能讓主候選提早通過12筆門檻。近期與整體資料重疊，並非兩份獨立證據或統計顯著轉折證明。近.5、窗口不足或沒有實質反向時，交回原有EV/風控。Volume Guard不能繞過此veto。畫面保留0.5pp機率方向，額外標示「正式下注觀望：近期與整體訊號分歧」，不混淆方向與資格。

## 診斷與回滾

context_mixture.experts的counts現在是加權計數，另有raw_observations/weighted_observations；context_mixture.history_adaptation記錄衰減設定。road_primary.history_adaptation記錄近期/最終機率、支持數、差距、reason、forced_skip，沿既有快照匯出。沒有跨桌持久新狀態。

history_adaptation_v1.enabled=false：候選與runtime精確回PR86；context_mixture_v1.enabled=false仍可回PR83，新增閘門也隨之關閉。git revert本次合併可完整回PR86。原授權與結束分析清除/重新導向不改，入口/模組快取版本更新。

## 500靴序列代理（不是正式下注回測）

新seed6182026，8副牌/burn14/cut14/最多70局；五批各100靴共35000局，無訓練/調參。此代理的分歧比較使用序列候選，未執行Physics/PF、EMA、0.5pp或正式EV/Volume Guard；沒有序列候選時用同一固定先驗計Brier。不能將下表稱為正式下注中獎率/EV。

| 模式 | 三類Brier | 非和局參考命中 | 最大參考連錯 |
|---|---:|---:|---:|
| pr86 | 0.58504368 | 49.6863% | 11 |
| adapted | 0.58376854 | 49.8801% | 13 |
| always_b | 0.58184000 | 50.9129% | 14 |

| 局段 | PR86參考命中/次數 | 候選參考命中/次數 |
|---|---:|---:|
| 1_20 | 45.8333% / 57 | 45.8333% / 57 |
| 21_45 | 49.9687% / 7035 | 49.9574% / 7768 |
| 46_70 | 49.5163% / 10536 | 49.8451% / 11083 |

Brier/logloss改善，但最大參考連錯更高、未優於永遠莊；前20局序列候選少，這份代理無法判斷早期Physics fallback的正式表現。序列代理中參考方向次數增加，不能宣稱出手率一定下降。依使用者要求作可回滾測試部署，沒有完整下注穩定度/真實勝率提升證據。詳細五批/各局段數據見validation/recency_conflict_smoke.json。

測試：node test_recency_conflict.js；node test_context_mixture.js；node test_road_primary_runtime.js；node test_end_analysis_reload.js。測試涵蓋原raw門檻、精確PR86回滾、與已發出機率一致的score-before-observe、淡化/重播/隔離、分歧只veto、不改final_p或EV、匯出及既有模型/授權。重現代理：node tools/evaluate_recency_conflict.cjs（需原Physics Python模擬依賴）。
