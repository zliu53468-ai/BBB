# 弱訊號相對基準輔助方向

本次只處理顯示/測試方向，不修改Physics、PF、融合、Macro/Clip/EMA、任何原始或校準後百分比、正式B/P/Skip資格、佣金/和局與Volume Guard。

基準沿用vnext/replay.cjs既有八副牌標準理論比較值B=.458597、P=.446247、T=.095156（近似參考，不是對實際牌桌已校準的模型）。以排除和局後q0=B/(B+P)約.5068比較。raw_gap=P(B|非和)-P(P|非和)；relative_gap=2*(P(B|非和)-q0)。只在|raw_gap|<=.02（莊閒相差2pp內）啟用。relative_gap>.001偏B、<-.001偏P、否則Balanced。不是固定把莊機率挪給閒，不刻意讓方向輪流。

正常顯示仍須原始莊閒差>=.005（0.5pp）；相對無明顯優勢時顯示觀望。不觀望模式同樣可顯示相對方向，無相對優勢時均衡。紙上實驗結算方向與實際顯示使用相同處理，配置標籤隔離舊實驗紀錄。正式下注仍使用原來的機率/EV，所以輔助閒可能同時顯示莊機率較高或正式下注觀望；文字明示「不改機率及正式下注」，不能把它當新正EV證據。

設定relative_baseline_v1.enabled=true，僅雙核心live啟用時生效。max_raw_gap=.02、min_relative_gap=.001可調，硬界限.03/.005。無效基準或缺模組則維持原方向；超出弱訊號帶維持原方向。關閉此flag即回PR90顯示；關閉dual live也自動取消輔助，回PR83。

raw_next_hand_forecast保留原始方向。next_hand_forecast.direction與forecast_direction為畫面輔助，relative_baseline欄位記錄raw_direction、advisory_direction、原始/相對差距、基準與display_only，進匯出快照。所有機率值及正式direction/bet_direction/EV保持原樣。基準是固定參考，不用當局尚未公布結果、不跨靴累積校正。

node vnext/test_relative_baseline.cjs驗證微弱B可變P、強訊號不改、0.5pp不放寬、原始百分比/正式出手/EV/guard完全一致、無修改輸入、關閉精確回PR90、匯出與均衡處理。未證實提高命中率，這是比較基準的可回滾實驗，不是機率校準或模型晉升。
