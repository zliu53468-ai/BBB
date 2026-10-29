# 整條流程共用可靠度的自適應協調

保留PR80順序：貝葉斯+馬可夫自動選階→Physics+PF校正→原Macro/Clip/EMA→最終EV→原Volume Guard。沒有新增局數分段，也沒有重訓。

road_primary_v1.adaptive_pipeline_v1：enabled=true、ema_min_multiplier=.65、max_ev_uplift=.004。關閉即可回PR80數值路徑；策略tag版本更新以隔離舊狀態。

共用r=min(1-Physics uncertainty,1-noise,particle_quality,recent_ess_ratio)，各項限定0～1；缺少或非有限的粒子品質/ESS取0。ESS只是一個保守限制，不作預測準確率或真實樣本數。沒有實測殘差方差或多批抽樣置信區間，不能宣稱已使用它們。

馬可夫可用時：既有Physics校正權重乘r，保留原delta上限與不能翻向限制；不再把整個馬可夫機率縮向.5。馬可夫不足時：僅將Physics+PF條件莊率的偏移乘r，保留原和局質量。一次機率收縮後重算相同EV頭。

Macro已有的delta乘r（未校準係數仍維持0）。決策EMA沿原曲線，最新權重乘(.65+.35r)，最低.20；沒有前值時直接用本局值，不虛構舊值。這是調整最新權重而非再次向.5壓縮。Clip硬界限不放寬，Core/XGB不恢復影響力；本機校準仍維持原顯示閘門。

EV使用Clip/EMA後同一組機率、Physics保留的和局質量重算：EV_B=.95*pB-pP，EV_P=pP-pB。EV本身不再次乘r；最低EV資格在原有計算上加.004*(1-r)，Volume Guard原邏輯保留，沒有降低原門檻。這不是保證所有新舊預測的出手集合互為子集，因為機率與EMA也變動。

診斷road_primary.adaptive含r、各項來源、原始/校正後機率、EMA實際權重、最終EV及門檻加成，隨既有快照匯出。所有數值只使用本局前資料。點結束分析仍先清除當桌狀態，再依原有防快取網址重新導向；入口與受保護模組帶新版本參數，開通碼/30分鐘限制不改。

測試：node test_adaptive_pipeline.js、node test_road_primary_runtime.js、node test_context_bayesian.js、node test_end_analysis_reload.js。沒有獨立真實牌靴的命中率提升證據；機制與回歸通過不等於勝率提升。
