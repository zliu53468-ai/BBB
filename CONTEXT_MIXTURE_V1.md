# PR83基準：因果多階上下文加權測試版

本次僅修改牌路候選來源，其他PR83的Physics/PF、Macro/EMA、Clip、EV/Volume Guard、0.5pp、Frozen Core及授權/30分鐘限制不變。沒有重訓原模型。

## 方法與參數

這是固定深度的因果專家混合，借鑑上下文加權思想，不宣稱是完整CTW演算法或具有其理論保證。一、二、三階各建B/P/T上下文計數，用原固定先驗B=.45/P=.45/T=.10及強度24。各階當前上下文仍至少12筆才可參與；全部不足照原Physics+PF暫代。新增候選不再呼叫PR83額外一階輔助，避免重複採信同份轉移資料；原一階配置保留作回滾。

每個已公布結果t，先用<t的計數預測並在支持數達標時累計 log(p_order(actual_t)/prior(actual_t))，然後才納入actual_t；這些評分僅用於下一手。分數逐步裁至[-8,8]，是保守工程限制，不是未裁切的貝葉斯證據。

當前log_weight=log(order_prior)+0.25*log_evidence+log(n/(n+24))；只在可用階數間softmax，得到一組混合B/P/T機率。order_prior預設[.6,.3,.1]，支持尺度24。這些是固定測試起點，未依此次測試結果挑最佳參數。未把不同階樣本數相加，observations為主權重階的支持數。

設定road_primary_v1.context_mixture_v1：enabled=true、order_priors=[.6,.3,.1]、learning_rate=.25、evidence_cap=8、support_strength=24。缺值/無效參數用保守預設或界限。所有計數與分數每次從同靴輸入重建；撤銷、跨桌與重播無隱藏持久狀態。

## 500靴序列代理檢查（非正式下注回測）

新模擬seed5182026、標準8副牌補牌、burn14/cut14、每靴最多70局；500完整靴，五個100靴批次，共35000局。無訓練或依報告調參。只跑同一前綴的PR83與加權序列候選；缺候選時用相同固定先驗計Brier/logloss。參考出手僅指候選有B/P，沒有Physics/PF、0.5pp、平滑、EV或Volume Guard，不能當正式下注命中率、出手率或EV。

| 候選 | 三類Brier | Log loss | 非和局參考命中 | 最大參考連錯 |
|---|---:|---:|---:|---:|
| pr83 | 0.58310202 | 0.94376796 | 50.1563% | 12 |
| mixture | 0.58324497 | 0.94397881 | 50.1986% | 13 |
| always_b | 0.58024000 | 0.93868771 | 50.4978% | 15 |

此次加權候選Brier與logloss略差，參考命中率差異很小且最大連錯更高，未證明改善穩定度，也未證明優於永遠莊。依使用者要求部署為可回滾實驗候選，不能宣稱已提升正式勝率。完整統計及五批差異見validation/context_mixture_smoke.json。

## 診斷與回滾

road.context_mixture.experts記錄每階context/counts/observations/eligible/probabilities/log_evidence/evaluated_predictions/support/weight，混合B/P/T與主權重order沿既有快照匯出。order現在是主權重階，不表示其他階被丟棄。

context_mixture_v1.enabled=false即可精確回PR83的硬回退與原無先驗一階輔助；策略version更新僅為隔離舊狀態。git revert本次合併亦可完整回復原PR83程式。入口與新模組有版本標籤，結束分析仍清除資料並重新導向。

驗證：node test_context_mixture.js；node test_pr78_first_order.js；node test_road_primary_runtime.js；node test_license_storage.mjs；node test_end_analysis_reload.js。重現序列代理：node tools/evaluate_context_mixture.cjs（需原Physics simulator的Python依賴）。機制測試含score-before-observe、支持門檻、正規化混合、無重複一階、因果重播、PR83精確回滾、EV/和局與匯出；另保留原模型/授權與平滑回歸。
