# 各路估計 → 單次融合 V1

以使用者確認的架構替代PR98逐層校正。新路徑：同靴B/P/T → 純序列貝葉斯馬可夫1–3階；Core、Direct Physics、PF原始後驗、57D XGB各自提供候選 → 一次有界凸融合 → 原Macro／Clip → 原決策機率EMA一次 → 原EV／Volume Guard。

## 分工與不重複修正
新sequence estimator只使用B/P/T上下文與固定非和莊基準0.458597/(0.458597+0.446247)，不再把Core／Physics先混入序列先驗。支持門檻仍1/2/3階=4/5/6，Beta先驗強度12，整靴同一規則，T保留上下文而不作B/P二元標籤。

Core輸出、Physics條件莊機率、原10000粒子posterior與XGB校準輸出都直接進融合器。PF沿用原抽樣、和局更新與相同前綴快照，不使用legacy blended Physics。Physics48D與XGB57D原始特徵／權重不改。新模式不呼叫舊Physics correction.apply、PF light.mix或XGB adaptive.mix；舊檔及設定保留供關閉回滾。

這是邏輯上的分路估計，不是新增多執行緒。各路仍共享B/P/T，XGB本身也使用Core／Physics特徵；不能當成統計獨立的多份新證據，不將機率相乘。

## 候選權重
single_fusion_v1.enabled=true，基礎權重sequence=0.50、core=0.10、physics=0.25、pf=0.05、xgb=0.10。這是待驗證起始值，未做勝率最佳化宣稱。
Physics gate：uncertainty0.35→0.65由1降至0。PF gate沿用quality0.55→0.80、ESS0.40→0.80、不確定性0.55→0.80的三閘門。XGB只使用本模式同靴已結算真實預測（含有登記的Skip）的非和樣本，至少12筆、最近24筆，與未融合sequence的事前Brier比較。gain>0.0005時，gate=n/(n+12)×clamp((gain−0.0005)/0.02,0,1)×noise_gate；noise0.55→0.85關閉。

configured_weight×gate後只正規化一次權重，再對PF實際權重上限0.06、XGB上限0.25作有界權重分配，最後只計算一次sum(weight×raw_probability)。權重總和1；來源缺失權重0，其餘重分配；全部權重為0時退回sequence。缺必要Physics或融合器則Skip。融合不是三個delta相加，因此舊±0.5pp／±0.25pp／±1pp逐層上限不適用於新模式；原Clip限制仍保留，沒有放寬EV／下注門檻。

更新後pipeline tag隔離舊EMA及舊Brier資料；需重新累積本模式配對預測。重試／撤銷按同靴前綴評估，模型／房間不混用。所有當局結果在預測之後才可結算評分。

## 下游與匯出
Macro特徵EMA保留，不另增機率平滑。applyDynamicSmoothing呼叫一次；既有Clip、EMA曲線、EV policy／Volume Guard函式源碼雜湊保持不變。和局質量沿用Direct Physics，EV_B=(1-pT)×(0.95q−(1−q))、EV_P=(1-pT)×(1−2q)，佣金與和局退回不變。一般畫面仍1pp門檻，不觀望按鈕保留。開通碼、30分鐘與版面不改。

bayes_markov.single_fusion匯出sources、configured_weights、gates、weights、weight_sum、xgb_evidence、model_tag、PF診斷、fusion_count=1及錯誤。finalProbability.rawPB是融合結果，finalPB是原平滑後機率；未融合序列raw_pb及全部來源另存，不混淆畫面與正式下注。

## 驗證與回滾
node test_single_fusion.js：計數／固定先驗、因果XGB評分、權重總和與上限、真10000 PF、一次融合／一次EMA、舊校正禁止呼叫、57D／EV／匯出、撤銷、缺模組Skip、PR98原始PR98雜湊驗證的逐欄基準回滾（跨平台浮點容忍1e-12，非數值／形狀一致）。
node test_xgb_adaptive.js：檢查後段函式雜湊不變。其餘既有重要回歸保持執行。
node tools/smoke_single_fusion.cjs：兩個完整模擬鞋（50與70手）配對煙霧；共用同一確定性PF原始快照僅節省重算，兩路仍各自平滑、評分與結算，不用未來結果。報告validation/single_fusion_smoke.json。不是大樣本／真實hold-out，也不保證提高命中率。

只將bayes_markov_final_v1.single_fusion_v1.enabled=false即可精確回PR98。保留原資料檔與授權；完整撤回可revert合併提交。
