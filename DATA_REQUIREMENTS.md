# 真實資料需求

目前真人完整牌靴0靴。vnext/synthetic_rows.json為固定種子模擬，不能當真實賭局證據。格式見vnext/data_schema.json，範例[{"shoe_id":"table1_shoe1","round_index":1,"outcome":"B"}]。rank為1=A、11=J、12=Q、13=K，10/J/Q/K百家樂值0；勝方按鈕不是點數1/0。

按shoe_id分組再validateRows，局號必須從1連續、同組不混靴，結果和可選牌值/點數/補牌規則必須一致。事前snapshot與事後outcome分開儲存；未知牌值留空，禁止補成猜測真實牌值。真實資料由使用者匯出同靴逐局紀錄後匯入，再以整靴/時間排序分Train/Validation/Test，Test不得調參。

晉升至少先累積1000實際Standard出手、30真實靴，仍須配對CI/EV/Brier/logloss/連錯風險檢查；達量不自動晉升。PF500/2000/10000精度差異還需要多種種子/大量獨立樣本比較；目前僅有一觀測的運算/Monte Carlo對照。
