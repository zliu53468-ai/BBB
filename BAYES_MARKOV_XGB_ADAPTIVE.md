# 貝葉斯／馬可夫＋自適應 XGB

在PR95的最終機率供應層恢復既有57D直接分類XGB。沒有重訓或修改權重、57D定義、Core、Physics、PF；原先Macro／Clip／EMA／EV／Volume Guard／1pp顯示門檻不調整。來源雜湊回歸鎖定Clip、EMA、EV policy與Volume Guard的函式內容。

流程：Core＋Physics → 原貝葉斯馬可夫機率 q_base 與57D XGB機率 q_xgb → 因果自適應凸組合 → 原Macro／Clip／EMA → 原EV／Volume Guard。

每次事前預測完整記錄兩個未混合的機率，不用已經混合的機率當基準。下局結果公布後才評分；同靴、模型指紋與設定吻合、前綴與實際歷史一致、已結算且非和局的預測才能進入評分。Skip的已登記機率也可評分，避免只挑下注樣本。匯入歷史不反造未存在的預測；重試同一局不重複計分。只統計最近24筆，至少12筆。

Brier gain = mean((q_base−y)²−(q_xgb−y)²)。gain>0.0005才可介入。reliability=n/(n+12)；w=0.25×reliability×clamp((gain−0.0005)/0.02,0,1)×noise_gate。noise由0.55到0.85時gate從1降至0；未知noise時gate=0。單次delta=clamp(w×(q_xgb−q_base),−0.01,+0.01)，q_mix=q_base+delta。硬上限：w≤25%、|delta|≤1個百分點。max_delta限制後記錄實際有效權重。這不是保證下一局更準的統計檢定。

樣本不足、前向Brier無改善、高noise、XGB不可用或模組缺失時使用原q_base。重新建立機率後才執行原平滑與EV，不再額外做第二次平滑／校準。和局質量、佣金與正式資格公式保持原樣。

cardcount_profile.json的bayes_markov_final_v1.xgb_adaptive_v1包含enabled、min_samples=12、window=24、prior_samples=12、max_weight=0.25、max_delta=0.01、min_brier_gain=0.0005、target_brier_gain=0.02、noise_start=0.55、noise_stop=0.85。這些是保守候選初始值，沒有聲稱最適。原先貝葉斯strength=12、Physics先驗75%、階數支持4/5/6與其他參數不改。

診斷／匯出：bayes_markov.xgb_adaptive的base_pb、xgb_pb、model_tag、samples、base_brier、xgb_brier、brier_gain、noise_gate、proposed_weight、weight、delta、reason。finalProbability.xgb_influence為實際權重，不是固定25%。模型更新、換桌／靴與撤銷有隔離；EMA的pipeline tag包含配置與模型指紋，更新後重新累積，避免把舊模型的平滑結果混用。

回滾：只將xgb_adaptive_v1.enabled=false，回PR95貝葉斯馬可夫模式，其他參數不變；將整個bayes_markov_final_v1.enabled=false則回PR94。開通碼／30分鐘／版面與模型檔都沒有修改。

測試：node test_xgb_adaptive.js；node test_bayes_markov_final.js；node tools/smoke_xgb_adaptive.cjs 200。新種子20261101的200靴成對功能測試見validation/xgb_adaptive_smoke.json；每手僅使用已公布前綴，沒有依結果回頭搜尋參數。這是合成煙霧測試，並非真實獨立hold-out、不是勝率提升保證。
