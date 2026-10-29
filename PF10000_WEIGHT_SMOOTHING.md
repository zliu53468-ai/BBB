# PF10,000權重與適中平滑實驗設定

這是使用者指定直接上線的人工參數組，不是經獨立回測找到的最優參數，不保證提高命中率。基底dd66e35，沒有重訓任何模型。

|局數|PF混合上限|決策/參考EMA基礎最新值權重|
|---|---|---|
|1–20|12%（原10%）|42%（原45%）|
|30|14%|48%|
|45|16%（原15%）|54%（原60%）|
|50|16%|57%|
|70|16%|62%（原72%）|

節點線性插值。PF實際權重仍由原品質及同向signal gate決定，最低6%、硬上限16%，不是每局直接混滿12%/16%；粒子多不代表自動更可靠。噪音與uncertainty原減項及alpha硬界限35%–75%不變。alpha越低，舊值權重越高、平滑越強。Macro參數與方向係數不變。

原不觀望參考方向不經決策EMA，因此新增獨立pf_display_smoothing.js，只在不觀望模式使用同一組alpha，讀取本局之前已存在的同靴預測。每次顯示概率最多偏離原值±0.006（莊條件機率0.6pp），不能翻轉當前原始概率符號，和局機率不變；不改正式概率、EV、Volume或下注資格。正常決策與參考顯示是分開分支，不是對同一結果連續做兩次EMA。缺少有效上一手預測即使用raw，不回填未知預測。

紀錄使用bgs_pf10000_display_ema_v1，受既有授權隔離與完整結束分析清除管理；新靴、版本變更、撤銷/改牌可重建。錯誤回退raw並輸出display_smoothing.error。本機校準tag納入本設定；正式EMA只取同設定版本的紀錄，避免沿用舊權重下的平滑值。舊500粒子專用實驗校準維持停用。

設定區塊pf_weight_smoothing_v1.enabled=false可停用新增權重曲線和顯示平滑，回到原PF10000參數；原本anchors完整保留。需要完全恢復檔案可建立新提交指向dd66e35的檔案樹。未改開通碼、30分鐘、Frozen Core、Physics拓撲、57D、模型權重、補牌/佣金/和局規則及EV門檻。

驗證：test_pf_display_smoothing.js（因果、冪等、撤銷、新靴、版本、符號、和局、配額錯誤）、test_pf_weight_smoothing_runtime.js（1–70參數、對照dd66e35 golden rollback、實際參考模式、正式EV隔離、重置）、test_browser_calibration.js、test_end_analysis_reload.js、test_license_storage.mjs均通過。未完成新牌靴命中率回測，不能把功能測試當勝率驗證。10,000粒子效能限制沿用上版。

Python cardcount_profile.interpolate/decision_alpha同步讀取同一設定並驗證界限；瀏覽器/Python逐節點與噪音alpha一致。舊中後段早期鎖定測試明確關閉新實驗組，保留原設定回歸；新的實驗曲線另行驗證。
