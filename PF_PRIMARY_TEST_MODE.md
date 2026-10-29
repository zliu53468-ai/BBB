# PF主導／Physics第二層測試模式

基底1eb641e。測試網址：https://zliu53468-ai.github.io/BBB/?pf_primary=1 。不加參數維持原Physics主導。需同樣有效開通碼，授權、裝置綁定與30分鐘不變。畫面modePill顯示「PF主導／Physics輔助測試」才是新模式；原獨立acompare面板不受這次更新影響。

## 真正的資料流

同靴B/P/T → PF先重建10000粒子並取得未混合posterior → Physics213→48 → 有界同向校準 → 既有Markov/Clip/EMA → Core/XGB診斷 → 同一組無條件機率的EV → 原資格/Volume → B/P/Skip。

PF呼叫只用中性48D介面適配向量，不先執行Physics，也不使用舊estimate.physics混合輸出。該中性向量不是觀測資料，其校正結果完全不作主要機率使用；實際PF posterior依原抽樣器的同靴歷史獨立生成。抽樣器、點數/張數修正及模型權重未改。

第一層方向取raw PF pB/(pB+pP)；不沿用舊PF必須跟隨Physics的clamp。第二層Physics只在同向、noise≤.70、uncertainty≤.70時，以設定8%做校準（硬上限12%），單次條件PB改動最多0.004、硬上限0.006；保持PF和局質量與方向。Physics可信且反向機率差≥4pp時，只強制正式Skip，不能翻向。弱衝突忽略Physics。以上權重指第二層當下校準，不能誤稱為經Markov與EMA後的最終總佔比。

其後Markov原8%同向輔助、EMA42%→54%→62%、EV/Volume條件維持；正式最終EV從本流程校準、Clip及平滑後的相同B/P/T機率重新算，不混用舊Physics勝率。佣金與和局退回原樣。Core/XGB不能否決/翻向；57D診斷仍使用原Direct Physics特徵分布。PF或必要模組失敗則新模式不可用/Skip，不冒稱已成功用PF預測。

## 開關與隔離

cardcount_profile.json新增pf_primary_v1：enabled=false、allow_query_test=true。query只供使用者明確開啟測試，預設不更動。移除?pf_primary=1即可回原模式；allow_query_test=false可遠端停用測試入口。未經大規模驗證不設為全站預設。

PF/Physics模式各自的校準tag、參考EMA、正式EMA與Volume歷史以pf_primary_policy_tag區隔，避免互相沿用預測。匯出pf_primary記錄raw PF/Physics方向、各機率、校準幅度、有效第二層權重、衝突及forced Skip。B/P/T只是可能牌組的條件資訊，不是實際牌階重建。

## 測試與限制

test_pf_primary.js：方向所有權、同向限幅、弱/強衝突、和局與EV。test_pf_primary_runtime.js：實際PF先於Physics、raw PF與抽樣joint winner一致、EV口徑、失敗觀望、XGB隔離與匯出。test_pf_primary_rollback.js：關閉模式與原1eb641e的機率/EV/方向/48D/57D完全一致。其他平滑/校準/授權/結束分析測試通過。Frozen Core、兩個權重檔、PF抽樣器SHA比較完全未變。

5靴全新模擬、每靴10/30/50/70四個預先指定的冷啟動局面，共20案例；不是完整逐局回測、不能評估連錯或持續校準效益，也不構成晉升依據。兩版用完全相同已知前綴與下一局結果；沒有依結果選參。

|指標|原預設|PF主導|
|---|---:|---:|
|參考方向B/P|12/8|20/0|
|非和局參考正確數|8/19|8/19|
|正式出手數|2|0|
|條件Brier|0.2454|0.2508|

未見優勢，PF參考方向偏莊，不能用隨機翻向掩蓋，也不能將原版僅2次出手的勝負當成穩定命中率。資料與原始統計：validation/pf_primary_smoke_shoes.json、pf_primary_checkpoint_smoke_report.json；重現node pf_primary_checkpoint_smoke.js。promotion_allowed=false。

若要評估能否替代正式版，仍需完整chronological whole-shoe資料及獨立hold-out比較命中率、扣佣EV、Brier、出手率及連錯。這次僅上線可選試用功能，不宣稱提高命中率。
