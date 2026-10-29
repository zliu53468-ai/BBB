# PR78 基準：整靴貝葉斯收縮＋馬可夫自動選階

整桌50～70局使用同一個規則，沒有前/中/後階段或第20局開關。選階仍完全沿用PR78：3→2→1階，最高階上下文達12筆已觀測轉移即採用，全部不足時照舊回Physics+PF。未降低樣本門檻，已移除PR79的6/10/14門檻與局數漸變。

所選上下文 n 筆，先驗強度 s=24+24*12/(n+12)。P(t)=(count[t]+s*prior[t])/(n+s)。少樣本收縮較強，多樣本漸近原24；貝葉斯不另選一個階，而是對馬可夫實際選定的階做後驗估計。prior沿用B=.45/P=.45/T=.10，這是既有對稱參考，不是已校準真實勝率。僅依已公布同靴上下文，沒有以未來結果挑階，也不把贏輸比例當實際牌值。

設定：road_primary_v1.adaptive_bayesian_v1（enabled=true、extra_prior_strength=24、support_scale=12）。原order=3、min_observations=12與prior_strength=24保留。extra硬範圍0～100，support_scale為1～100，非有限值回預設。

診斷：road.backoff維持各階樣本/資格；road.bayesian新增selected_order、observations、effective_prior_strength、data_weight與prior_weight；既有匯出快照會包含這些欄位。

PR78的分段EMA曲線、0.5pp畫面門檻、PF/Physics/Core/XGB、EV/Volume Guard、開通與30分鐘限制皆不動。本次僅取消「貝葉斯」的手數分段，並未取消PR78既有EMA。策略tag含新配置，可隔離舊決策/校準狀態。

回滾：adaptive_bayesian_v1.enabled=false，候選數值與選階完全回PR78；其餘版本tag僅為狀態隔離。完整回復PR78可將road_primary_v1.version改回ROAD_PRIMARY_ADAPTIVE_3_2_1_V2並刪除此區塊（或使用PR78提交）。git revert本次合併則回到父版本PR79，並非PR78。

驗證命令：node test_context_bayesian.js；node test_adaptive_road_v2.js；node test_road_primary_runtime.js。測試涵蓋全靴一致規則、各階回退、樣本越多收縮越少、關閉精確回PR78、因果重建及快照匯出。未重訓，未有獨立真實牌靴證明命中率提升。
