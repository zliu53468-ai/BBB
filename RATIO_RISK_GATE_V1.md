# 比例失衡風險閘門（候選，未上線）

只收緊正式推薦，不修改 Physics 機率、不引入馬可夫、不翻向。最近16局中至少12個有效莊閒結果；和局不算莊閒分母。莊占比偏離50%超過0.20，線性增加 abs(PB−0.5) 門檻，最大+0.012（莊閒差額外+2.4pp）；偏離達0.35直接Skip。這是保守風險啟發式，並非證明比例失衡代表下一手較難預測。

保留 early_mid_conservative_v2：前20局原12pp門檻不放寬，21–30局漸變至5.5pp，31–45局漸變至3.8pp；前30局PF≤8%、EMA≤40%、Residual≤±1.5pp、校準停用；之後至少30個有效莊閒樣本。路單外層修正權重為0，XGB/馬可夫不參與正式方向；Physics本身仍使用既有B/P/T輸入特徵，未改特徵或重訓，不能宣稱其內部序列影響被量化為5%。

回滾本次閘門：physics_priority_v1.ratio_imbalance_guard.enabled=false。上一版500靴統計不代表本次新閘門已回測；本次只做單元測試與真實runtime三種歷史煙霧測試，不宣稱命中率提高。

修改：physics_priority_runtime.js、cardcount_profile.json；auth-client.js與index.html僅更新快取版本。新增test_ratio_risk_guard.js。未修改模型、Core、開通碼、使用期限或重置流程。
