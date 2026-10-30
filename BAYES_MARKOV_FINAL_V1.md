# Core＋Physics → 貝葉斯／馬可夫 1–3階最終機率

> 本文件描述PR95基準。後續啟用xgb_adaptive_v1時，僅以有界自適應XGB取代下述「不執行XGB」的設定；其他流程不變，見BAYES_MARKOV_XGB_ADAPTIVE.md。

依 BGS0.3V.1 的 fbe3f88（原 Final57 XGB 最終輸出流程）恢復機率供應器的位置，改為貝葉斯馬可夫上下文估計。不是整庫退版，不搬回舊LINE／授權程式，也不是完整逐值複製當年引擎。正式數學模型權重與57D特徵定義不改。

## 現行執行鏈
B/P/T同靴前綴 → Frozen Core機率＋Direct Physics 213→48 → 貝葉斯馬可夫1–3階最終P(B|非和) → 原Macro／Dynamic Clip／決策EMA → 原EV policy／Volume Guard → 正式B/P/Skip。

原57D向量保留作匯出診斷，不執行XGB。PR94的雙核心融合、PF混合、獨立牌路主導、弱訊號相對基準、顯示端雙向校正與不觀望Platt實驗不疊加到新機率。這些舊流程檔案與旗標都保留供回滾，啟用新模組時由單一入口隔離。新模式一般畫面仍使用既有1pp門檻，不觀望按鈕保留；畫面方向不等於通過正式EV。

## 統計與配置
bayes_markov_final_v1.enabled=true；version=BAYES_MARKOV_1_3_V1。全靴50–70局使用同一套規則，不按前中後段切換模型。EV／平滑既有分段保留。

每個order=1/2/3，以最新order個B/P/T為上下文，僅統計已公布歷史中同上下文之後的B/P/T。B/P是二元預測目標；T保留在上下文，但不計入B/P有效支持。每階輸出原始Markov頻率與Beta後驗，這是同一份證據的一次估計，不將貝葉斯與馬可夫當獨立模型重複乘權。

prior_strength=12；physics_prior_weight=0.75。先驗由75% Physics非和莊機率＋25% Core莊機率構成，再依Physics uncertainty向標準八副牌非和莊基準收縮。各階後驗 (B+12×prior)/(B+P+12)。min_support：一階4、二階5、三階6；採最高滿足支持的階，缺樣本回低階，無階可用則回Core／Physics先驗。以上參數是未驗證候選起點，非最適或勝率保證。

每次依傳入的同靴前綴重建計數，沒有跨靴模型記憶；撤銷重算一致。核心A/b與權重不改、不重訓。既有決策EMA／Volume Guard仍使用已結算本靴快照，pipeline tag含新配置，不能沿用舊模型快照。

## EV口徑與安全
保留目前和局退回與莊5%佣金：EV_B=(1-pT)×(0.95q−(1−q))，EV_P=(1-pT)×(1−2q)。不恢復舊Python T=0 的輸出缺陷。顯示B/P是排除和局的條件機率，B/P/T診斷保持和局質量且總和1。正式EV門檻及Volume Guard設定不變，不以低支持自動強制下注。缺新模組、Physics或無效資料則Skip，不能靜默假裝成功。

## 測試／回滾
node test_bayes_markov_final.js：上下文計數、1/2/3階、自動回退、因果與撤銷、XGB/PF/舊融合隔離、48D／57D、和局／佣金EV、診斷匯出、失效Skip與PR94精確關閉回滾。
node tools/smoke_bayes_markov_final.cjs 200：固定種子八副牌模擬、50–70局、僅歷史B/P/T輸入，產出validation/bayes_markov_smoke.json。此為功能煙霧，不是真實hold-out，也不以測試結果調參。

只將bayes_markov_final_v1.enabled=false，即回PR94原預測；完整撤回可revert本次合併提交。沒有更動開通碼、30分鐘限制、UI版面、Physics／Core／XGB模型檔。當前main以此新模組為預設候選，尚未驗證優於舊版。
