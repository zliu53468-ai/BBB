# PR83基準稽核

基準32fe3c001911998d299c9ad13a133aa40f971064。原檔案不改。

BPT prefix進predictPhysics（固定213D causal history編碼與既有MLP 48D）；原stateless PF10000每次重播同靴prefix，不是逐局重訓。原按鈕介面沒有點數/張數欄位；recordCompletedHand可存Macro觀測，但PF estimate只取BPT，不代表真實牌面已條件化。vnext明確另提供部分/完整觀測入口，不修改原輸入流程。

主路徑：3→2→1 Markov（各上下文>=12，固定prior .45/.45/.10強度24）→無先驗一階輔助2%、delta<=.001、不翻向→Physics/PF有限校正→Macro/Clip/causal EMA→最終EV→Volume Guard。馬可夫不足時Physics/PF暫代。Core/XGB仍計算，active road模式強制xgb_zero_influence，不混入最終機率。

Physics輸出包含4/5/6張、點數、winner BPT與uncertainty等48D；沒有真實牌階時，PF殘牌只是假設後驗。原57D ABI及XGB binary目標不變。原fixed先驗與一階輔助使用重疊BPT資料，不應當成獨立證據；VNext核心間同樣相關，因此不作獨立likelihood相乘。

正式EMA只讀同靴且round_index較早、同策略tag的快照；本機校準為顯示層且有閘門。未發現本次新模組把當局結果當事前輸入：回放先預測、事後另寫actual。這不是對所有歷史研究程式的安全證明。原API與權重不動，採完全獨立shadow loader，所以不存在新增雙重正式平滑。
