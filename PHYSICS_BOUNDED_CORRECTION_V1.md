# Physics 有界校正與和局檢查

> 此文件的PF未執行／和局檢查是PR96基準。後續pf_light_v1啟用後，PF會在Physics校正後小幅介入，見PF_LIGHT_INNOVATION_V1.md；PF核心與和局規則仍未修改。

新增位置：原貝葉斯／馬可夫機率 → Physics校正 → 原自適應XGB → 原Macro／Clip／EMA → 原EV／Volume Guard。其他模型、權重、57D特徵、平滑與出手設定不變。既有Physics已用於先驗，本層是明示且有界的第二次使用同一訊號，不將兩者當成獨立證據相乘，亦不宣稱新增了真實牌階資訊。

配置bayes_markov_final_v1.physics_correction_v1：enabled=true、min_weight=0.10、max_weight=0.20（硬上限）、max_delta=0.005（硬上限）、uncertainty_low=0.35、uncertainty_high=0.65、signal_scale=0.05。

q_phys=physics[23]/(physics[23]+physics[24])。signal=clamp(|q_phys−0.5|/0.05,0,1)，nominal=0.10+0.10×signal；uncertainty_gate=clamp((0.65−physics[42])/0.30,0,1)。delta=clamp(nominal×gate×(q_phys−q_base),−0.005,+0.005)。高不確定性gate為0；缺模組或無效Physics保留q_base。名目權重10–20%，受uncertainty及±0.5pp上限後實際權重可能更低，不強迫翻向。

XGB的比較基準改用已校正q_base，既有最少12筆、24筆視窗、權重25%與±1pp上限都不改。模型／配置tag包含本校正，不能重用PR96的舊配對評分或EMA快照。更新後須重新累積同靴已結算預測。匯出bayes_markov.physics_correction含原機率、Physics條件機率、校正後機率、uncertainty gate、名目／實際權重、delta與reason。

## 和局：僅檢查，不加翻邊規則
目前bayes_markov_final_v1啟用時，正式路徑不呼叫PF；PF script仍為回滾保留。只讀5組PR96前綴檢查，PF呼叫均為0，和局後Physics48D皆更新。T保留在1–3階上下文；若T相關上下文支持不足，模型回退到低階或Core/Physics先驗。T不是B/P二元標籤，因此不計入B/P支持數或XGB Brier樣本。原EMA會使用已結算的上一筆機率；測例最新值權重0.392，可能讓方向暫時留在舊側。

例BBPP重複前綴：和局前finalPB≈0.48508；和局後rawPB≈0.51499，但EMA後finalPB≈0.49680，畫面因1pp門檻觀望。這是有限前綴示例，不是使用者實際個案，不能聲稱所有和局都由同一原因造成。報告validation/tie_pr96_inspection.json。沒有更改T上下文、和局退回、補牌、PF或平滑規則；和局後不必然換邊，也沒有新增必翻邊條件。

## 驗證／回滾
node test_physics_correction.js：上下方向對稱、gate、硬上限、輸入失效／關閉回退、Physics→XGB接入、48D/57D與權重檔不變、匯出及PR96和局前綴重播。
node test_xgb_adaptive.js：鎖定既有Clip、EMA、EV及Volume Guard函式源碼雜湊。
node tools/smoke_physics_correction.cjs 200：新種子20261102同靴配對功能測試，結果見validation/physics_correction_smoke.json，未依測試結果調參，並非真實獨立hold-out。

physics_correction_v1.enabled=false即回PR96原路徑，其他設定不變。此配置是依使用者要求上線的可回滾實驗；無已驗證命中率改善保證。開通碼、30分鐘與原版面不變。
