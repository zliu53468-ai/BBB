# 一階貝葉斯＋GRU自適應介入測試版

基準PR83。保留馬可夫3→2→1的12筆支持門檻、固定先驗、原Physics/PF/Clip/Macro/EMA/EV/Volume Guard；模型權重、佣金/和局與開通/30分鐘限制不改。

## 一階貝葉斯

sequence_aux_v1.enabled=true時，以bayes_first_order取代原無先驗一階輔助：prior B=.45/P=.45/T=.10，固定強度24，posterior=(count+24*prior)/(n+24)。至少12筆，原2%混合與0.001條件莊率上限不變；不能翻轉主方向。這是一組序列證據的替換，不把同一資料當成獨立投票加倍。

## 真正固定訓練的GRU

最近20手B/P/T，單層16單元，純NumPy實作GRU三閘BPTT/Adam訓練全部參數、瀏覽器只推論。每次從同靴prefix最後20手及零hidden重建，因此新靴、撤銷、重播不會繼承其他桌狀態。少於20手或模型/版本失敗時只有診斷，不改主候選。

新模擬1000靴：標準8副牌補牌規則，固定burn14、cut14、最多70手；600訓練/200驗證/200測試，按產生順序完整牌靴分割，再建立20手視窗。固定8 epochs，僅以validation logloss選epoch；test只評估選定模型，未依test重新搜尋。無實際牌面輸入，沒有真人驗證。

| 分割 | 模型 | Brier（三類） | Log loss | 全部非和局參考命中率 |
|---|---|---:|---:|---:|
| validation | gru | 0.58398752 | 0.94689327 | 51.3813% |
| validation | constant | 0.58398355 | 0.94686910 | 51.3813% |
| validation | markov_bayes_proxy | 0.58690256 | 0.95172559 | 50.6934% |
| test | gru | 0.58472748 | 0.94839102 | 50.4109% |
| test | constant | 0.58471605 | 0.94836614 | 50.4109% |
| test | markov_bayes_proxy | 0.58809875 | 0.95420637 | 49.5669% |

**這些是GRU及序列代理的獨立分類指標，不是完整PR83+Physics/PF的正式下注命中率/EV。** 每個驗證/測試分割各10000個20手之後的事前預測視窗。GRU未優於constant基準，standalone_gate_passed=false；full_runtime_validated=false，因此promotion_approved=false。沒有宣稱勝率提升。

## 自適應閘門

現行部署會執行GRU並顯示/匯出B/P/T及validation_gate_closed，實際介入權重0。要介入，模型必須同時獲得standalone_gate_passed、full_runtime_validated、promotion_approved；這次不偽造任何通過欄位。之後真正通過才按history/support/entropy閘門給最多5%權重、條件莊率delta最多.002，不翻轉主候選或改和局質量。熵低本身不等於準確；不能取代hold-out晉升。

原GRU權重候選保存在gru_aux_candidate.json，不替換任何Physics/XGB .json/.joblib。詳細報告gru_aux_report.json；重現：python tools/train_gru_aux.py --output . --shoes 1000 --epochs 8 --seed 20261030（需現有NumPy及Physics simulator依賴）。新權重須更新expected_model_version與策略version，不可直接打開晉升閘門。

## 回滾與驗證

road_primary_v1.sequence_aux_v1.enabled=false即可回PR83的無先驗一階與其餘原數值路徑；或revert本次合併。兩個子模組也可獨立關閉。既有first_order_aux_v1配置保留作回滾。

node test_sequence_aux.js：Python/JS權重推論golden一致、固定一階先驗公式、因果/20手context/新靴與撤銷重建、缺模型、實際失敗閘門0權重、僅測試fixture通過後5%/delta上限及不翻向、PR83精確回滾、EV/和局與快照；另保留原road/模型保護/授權/結束分析測試。所有fixture通過旗標僅在記憶體測試，不寫回候選權重。
