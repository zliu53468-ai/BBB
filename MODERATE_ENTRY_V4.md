# 適度增加出手 v4

修正前期多批EV檢查口徑：先前批次用未混合PF後驗，正式推薦卻為Physics+最高8%PF。新增evaluation_mode=official_blend，4批仍各自重建不同可能牌組，但以同一Physics混合口徑計算EV下界；預設不填仍是舊posterior_only，方便回滾。保留4批同向與EV下界高於原門檻，沒有降低EV底線。

前1–30局莊閒差門檻5.5pp（原前20局12pp），40–45局3.5pp，50–70局2.8pp，節點漸變。保留上一版平滑、不確定性、ESS、品質、比例失衡、3/5連錯保護、前期Residual及校準限制。版本PHYSICS_MODERATE_ENTRY_V4隔離舊決策EMA。

快速檢查：350組人工機率/階段組合，舊門檻合格222、新門檻276；此為功能驗證，非牌靴回測或命中率提升證據。實際runtime煙霧測試與單元/整合/回滾測試通過。未訓練模型、未改Core/授權/30分鐘。依使用者要求實驗上線，不能保證較多出手仍維持命中率。

回滾本次：恢復上一版early_mid_conservative_v2.version及probability_band曲線，early_batch_ev_gate.evaluation_mode改posterior_only；或revert本次合併。全體新外層關閉physics_priority_v1.enabled=false。
