# 原A回復與A+逐局參考建議

基底為原A提交c6f4a0a，移除PR73/74新增閘門與平滑，保留原畫面、開通碼、30分鐘及完整結束分析清除流程。採使用者選定的獨立面板A+候選A_m4_r0.0001_d0.01：馬可夫4%、有界單調Platt slope=0.5、intercept=0.01879682266346735、每次條件機率校準上限1pp；本機校準沿用原A閘門。Physics/PF/Core/XGB權重檔均未改。

預設啟動原有不觀望參考模式，開始分析也保留此預設；每次報B/P/T即更新下一局，參考方向與正式下注資格分開。正式EV/Volume保護仍可Skip，不將參考紀錄列為正式下注。模型未就緒不虛構方向。保留切換按鈕。新版本/模型指紋隔離原校準狀態。

此為既有模擬候選上線，沒有重新訓練、沒有新10,000靴回測，先前測試未證實優於原A。不保證勝率/獲利。

驗證：test_aplus_release.js、test_browser_calibration.js（明確切換正常模式驗證）、test_end_analysis_reload.js、test_license_storage.mjs、test_no_skip_calibration_candidate.js。新測試驗證指紋有效、0/1/5/30/48手逐局方向、校準上限、機率總和、切換顯示不改正式EV資格。

回到原A：部署c6f4a0a檔案樹。回到本次改動前：部署5be10a5檔案樹，保留提交歷史。停止預設逐局參考：a_plus_release.enabled=false並重新載入；馬可夫與Platt回原A需一併恢復原profile及對應指紋，不能只修改其中一個。
