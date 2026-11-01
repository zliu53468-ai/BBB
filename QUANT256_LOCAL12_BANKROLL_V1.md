# BBB 256D 局部校準與本金控制 V1（2026-11-01）

正式頁面新增 quant256_local_calibration.js，於既有量化主模型與物理融合模組之間載入。原本開通碼授權、牌局 B/P/T 儲存鍵、返回上一局、前端其他功能保留。既有 app256.js 的 Frozen Direct 是手工靜態先驗，不是已驗證勝率的機器學習權重。

## 近 12 局真正更新 A/b
每次輸入 B/P/T 前，保存當局開獎前的完整 256D context；輸入開獎後才將標籤寫入局部 FIFO。只有最後 12 個結果有效；和局計入窗口但不標 B/P。使用 42 個有效維度、42x42 完整 SPD ridge 矩陣，A=I+Σxxᵀ，b_B=Σ(開B時+x、開P時−x)，b_P=−b_B。Cholesky 求解預期分數與模型不確定度。
融合公式：blend_weight=0.35*min(1,effective_samples/max(4,12*0.67))；blended_ucb=(1-w)*base_ucb + w*local_ucb。這是局部研究分數，不是正確率。

## 第 129～256 維特徵
新增 road[24..29]，即 256D 的索引 152..157，含近兩局連同側/單跳結構與近五局轉向率、run、規律度、莊比例及熵差的二階交互。舊主模型 36D 有效索引不變，局部模型加用 6 維。

## 雙重門檻
現有物理門檻 0.008／連勝 0／失誤單次 0.035 保留。新增 UCB 雙臂差距與差距相對模型不確定度的兩項門檻；連敗一次時各乘 1.5，連敗兩次以上乘 2.0。只統計實際已提示且已結算的前推研究訊號，和局不算錯、跳過不假造結果。不硬翻莊/閒，不保證連敗最多兩次。

## 本金與曝險試算面板
本金花費選擇：NT$5,000／10,000／20,000／30,000／50,000；比例 10%～50%，預設 20%；狀態儲存在獨立 localStorage 鍵 bbb_stake_exposure_panel_v1。計算公式為本金×比例×模型相對信心權重（0.5～1.0），並限制在本金 10%～50%。未通過研究訊號即顯示 NT$0。這是資金曝險試算，並非真實勝率或經證實正 EV 的下注建議。每局投入 10%～50% 本金風險很高。

## 檔案與測試
quant256_online_runtime.js：新增六個二階路單特徵。
quant256_local_calibration.js：12-window A/b、融合 UCB、曝險試算。
physics_forecast_first_runtime.js：局部訓練、雙重門檻、連敗門檻調整。
quant256_online_panel.js、index.html、styles.css：本金 UI 與狀態顯示。
auth-client.js：授權後按順序載入新模組。
test_quant256_local_calibration.js、test_quant256_online.js、test_physics_forecast_first.js：整合測試。

模型僅透過 B/P/T 不能還原真實 A～K 耗牌，不可能保證四連勝或阻斷三連敗。應使用不重疊的獨立牌靴做前推回測，公開出手密度、命中率、連敗長度及實際期望值。