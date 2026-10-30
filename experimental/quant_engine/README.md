# BBBQuantEngine：規格驗證候選，尚未接入正式預測

完整單檔：bayes_markov_runtime.js；Node測試：node experimental/quant_engine/test_engine.cjs。瀏覽器載入此檔後為window.BBBQuantEngine，Node為require(...).BBBQuantEngine。

本分支不取代正式PR98、不修改Core或授權。原因：
1. B/P/T不能識別真實殘牌、對子或幸運六機率。扣牌40/45/15等比例與RC/L6都是使用者指定的未校準啟發式，不是真實Card Counting。未知對子不冒充沒開對子；updateRound可額外提供已觀測pair:true/false，否則pairCount不動。
2. 固定80/20與TC限制±8、最大順龍±.005，最後莊機率最大只改±.0034。保持B/P/T總和1、莊佣金5%：EV_B最高−.0039，EV_P最高−.0056，主線永遠Skip。固定T=.0952且8:1的EV為−.1432，也不下注。不能同時照此公式又宣稱有正EV出手。
3. 現有app256forward.js的前128D並非全部1.0：只有其中10個槽固定1，其餘有進度、和局、平衡、熵等既定定義。新128D是BBB_QUANT_HEURISTIC_128_V1，不相容舊Frozen Core權重。generateShoeContext可供新模型研究；injectShoeContext預設拒絕注入，顯式allowUncalibrated=true才可作離線實驗，不把長度相同當模型相容。

退火只將條件組成影響降到70%，剩餘30%按中性當前密度扣除，單局總耗牌仍4.94/4.5；分群耗盡會按剩餘容量重分配，避免負牌或帳面總量不一致。<=60的洗牌是估計值觸發重置，不代表知道真實賭桌已洗牌。

對子／幸運六指數僅提示；無校準機率與實際賠率則不發出可下注建議。即使提供機率，也不自動替邊注分配本金。主線Kelly考慮和局退回；若正EV才按0.25 Kelly後Clip5–30%，此5%下限可能高於原1/4 Kelly，輸出minimum_clip_applied以明示。當前指定公式的主線不會進入正EV分支。

128D映射：0–15主狀態；16–55五分群×8個派生值；56–79 TC區域基底；80–103進度基底；104–127八個窗口的B/P/T頻率。cardRatios=估計剩餘群張數/初始群張數；cardDensities=估計剩餘群張數/剩餘總張數。二者不混稱真實殘牌。

上線前需決定：接受上述必然觀望的獨立模擬模式，或重新定義有校準依據的機率／邊注輸入與128D特徵版本。未做勝率提升宣稱，也未擅自調高機率或沿用不相容權重。
