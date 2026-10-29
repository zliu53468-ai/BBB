# EMA與PF混合上限由低到高 v4

依使用者要求重新調整兩條曲線；不是獨立回測找到的最優值，不保證提高命中率。基底2c721d0，點數/張數條件抽樣修正保留。

|節點|EMA最新值基礎權重|PF混合上限|
|---|---|---|
|1–20局|42%|10%|
|30局|48%|12%|
|45局|54%|14%|
|50局|57%|14.4%|
|70局|62%|16%|

節點線性插值，兩者均單調不減。alpha越高，最新機率權重越高、平滑越輕；不是後段平滑更強。噪音/uncertainty原減項與alpha35%–75%界限保留，第一個沒有前值的預測仍直接用raw。PF表格是上限，實際由品質/同向閘門決定，最低6%與硬上限16%不變；不代表實際PF權重每手一定上升。

正式決策與不觀望參考模式共用曲線，各自分支平滑，不重複套用。參考平滑仍最多±0.006、不可翻向、五五波保護、撤銷/新靴重建。版本PF10000_ASCENDING_WEIGHT_EMA_POINT_CARD_V4隔離舊曲線的平滑與校準狀態。原10000粒子、點數/4–6張聯合分布、不匹配樣本排除、每正權重粒子5次forecast、馬可夫、EV/Volume/連錯相關既有條件未改。沒有重訓模型或改Frozen Core，授權與30分鐘不變。

通過：test_pf_weight_smoothing_runtime.js（含1–70雙曲線單調、舊設定回滾、正式EV與參考顯示隔離）、test_cardcount_profile.js（Python/瀏覽器一致）、test_pf_display_smoothing.js、test_browser_calibration.js、test_end_analysis_reload.js、test_license_storage.mjs、Python的分段連續性測試。設定檔除新版本及兩條曲線外與2c721d0完全相同。

回滾此次：以新提交恢復2c721d0檔案樹，或恢復該版pf_weight_smoothing_v1.version與anchors.decision_alpha/anchors.pf_max。關閉整個pf_weight_smoothing_v1會回到更早期PF10000原設定，不等同上一版本。無新增命中率回測，不將功能測試當作勝率提升證據。
