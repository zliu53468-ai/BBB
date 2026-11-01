# 256D Online LinUCB V1（2026-11-01）

## 適用方式
開通碼驗證後，原本 BBB 頁面的「莊／閒／和」按鈕直接啟用新版 LinUCB，歷史記錄沿用 \`bgs256d_frozen_6x15_forward_v18\`。每次點擊 B/P/T，先由原系統儲存結果，再由新版線上引擎更新並自動顯示下一局結果；開始分析也會同步。返回上一局重建模型，結束分析由既有授權清除流程處理；新模型的 A/b 僅存於記憶體，不額外保存私有資料。

加入 \`?quant256=frozen\` 可保留原始 Frozen Direct 介面作對照。既有 \`app256forward.js\`、授權驗證、Cloudflare Worker 與 Frozen Core 皆未變動。

## 核心參數
- \`lambda = 0.94\`（驗證限制 \`0.90 ~ 0.96\`），每一局 \`A = lambda * A + (1-lambda) * I\`、\`b = lambda * b\`。
- 已知本局結果為 B/P 時，使用上一局的 256D context 更新兩臂：\`A += xx^T\`，正確的 arm 更新 \`b += x\`，另一臂 reward=0。和局僅執行衰減及歷史更新，不會被錯誤標記 B/P。
- 嚴格觀望：\`minSamples = 12\`（非和局）、\`confidenceThreshold = 0.025\`，以 3/8 局 Entropy、8/16 局 turn-rate 波動拉高門檻。當 \`abs(UCB_B - UCB_P) < threshold\` 或樣本不足，強制 \`Skip\`。
- 256D 固定接口：前 128 維進程／模擬，後 128 維路單；實際使用 36 個特徵，矩陣在所有 active dimensions 上包含完整非對角協方差，停用超長窗 48/56/64/72 局隨機特徵。短窗 3 vs 8 Regularity 與 Entropy 差分加入後半部。
- 盲眼物理：24 個粒子分別使用標準第三張牌規則，依已知 B/P/T 結果做條件抽樣，估計可能的發牌張數、虛擬高／低點值殘量；結果標記 \`simulated_unobserved\`，不會填入任何臆造的真實 A~K 殘牌、已知補牌數或校準勝率。物理特徵在 context 中保守夾限。
- \`alpha = 0.15\` 僅為 UCB 的探索係數，模型 UCB 差距不是莊／閒真實機率，也不表示正期望值。

## 驗證
\`node test_quant256_online.js\` 涵蓋 A/b 更新公式、lambda 上下界、嚴格觀望、盲估不可辨識、256D 形狀、重播防重複、返回上一局、按鈕接管、防止 Frozen 重複判斷。GitHub Actions 工作流：\`.github/workflows/quant256-online.yml\`。

**重要限制**：標準百家樂中，只看歷史 B/P/T 並無已證實可預測獨立下一局的套利訊號；線上學習或模擬並不保證提高真實出手命中率。只有在具代表性的逐靴、前推式獨立留出資料驗證後，才能評估門檻是否真的有效。
