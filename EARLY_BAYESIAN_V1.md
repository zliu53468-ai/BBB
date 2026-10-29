# PR78 架構內的前期貝葉斯收縮與上下文選階

保留 3→2→1 階上下文選階，再對所選階數的 B/P/T 計數使用 Dirichlet 先驗收縮。僅影響預測第1～20局（已知歷史長度+1）；第21局起與PR78完全相同。無新增模型、無重訓、無儲存跨桌狀態。

## 可調參數

road_primary_v1.early_bayesian_v1：enabled=true；end_round=20（硬上限20）；taper_start=10；一/二/三階樣本門檻6/10/14；extra_prior_strength=24；support_scale=12。

前10局 phase=1；第11～20局 phase=(21-預測局數)/11，21局起為0。各階門檻 ceil(12+phase*(該階配置-12))。最高階有足夠已公布轉移才採用，否則退階；全部不足照舊回Physics+PF。這比PR78一律12筆允許一階更早就緒，同時增加收縮，並不保證較準。

實際先驗強度 = 原24 + phase*24*12/(上下文樣本數+12)。P(t)=(觀測次數+實際強度*prior[t])/(樣本數+實際強度)。先驗沿用PR78的B=.45/P=.45/T=.10；不是已校準真實勝率。樣本越少越收縮，不用全靴比例或未來標籤選階。

## 診斷與邊界

road.backoff記錄各階observations、eligible、minimum。可用候選新增road.bayesian：prediction_round、phase、effective_prior_strength、data_weight、prior_weight。原先road快照嵌套匯出自然保留這些欄位。

不調整PR78的EMA、0.5pp畫面門檻、PF、Physics、Core、XGB、EV/Volume Guard或開通/30分鐘限制。不同配置的原有策略tag會隔離歷史決策狀態。

## 驗證與回滾

node test_early_bayesian.js；node test_road_primary_runtime.js；node test_adaptive_road_v2.js。

early_bayesian_v1.enabled=false即可回PR78數值與選階；本次也改了策略version用來隔離狀態。完整回復可revert合併提交。

僅通過因果與機制回歸不代表命中率提升；目前沒有新獨立真實牌靴證明前期收益。
