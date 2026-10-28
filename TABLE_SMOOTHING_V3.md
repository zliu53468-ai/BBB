# 50–70局分段平滑 v3

使用者明確要求將候選一併上線。這是實驗配置，尚未證明提高命中率；先前500靴對照的命中率下降結果仍保留，不能當成v3驗證。

只改Physics外層決策EMA曲線：前1–30局base alpha=0.32、上限0.40；45局base=0.34、上限0.48；50局base=0.35、上限0.49；70局base=0.36、上限0.50。節點間線性插值。alpha代表最新值權重，中後段比上一候選更平滑；實際值仍由原signal_gain與noise_penalty計算並受上限限制。Macro、Frozen Core及模型權重不變。版本EARLY_MID_TABLE_SMOOTH_V3隔離舊平滑記憶。

包含前中段保守門檻、比例失衡風險閘門與前30局4批獨立PF重建EV穩定性閘門；這些均非已驗證勝率優勢。開通碼與30分鐘邏輯不變。

回滾整個候選：physics_priority_v1.enabled=false；僅回滾本次平滑：恢復上一提交的early_mid_conservative_v2.version/curves.alpha/curves.alpha_max。正式整包可revert合併提交（merge方式需git revert -m 1 MERGE_SHA）。
