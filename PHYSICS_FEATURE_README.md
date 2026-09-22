# Physics 56D Production Runtime

## 正式架構

目前 production chain：

\`\`\`text
B/P/T History ──────────────┐
                            ├─ PhysicsFeatureExtractor (48D Multi-Task MLP)
                            │
256D / V23 Core ─> Core P(B)
                            │
                            ├─ 原固定 7D（值與順序完全不改）
                            │
                            └─ [1D Core P(B) + 7D + 48D] = 56D
                                             │
                                             v
                                      XGBoost Residual
                                             │
                                      Delta clip ±10%
                                             │
                                         Final P(B)
\`\`\`

\`app256forward.js\` 與 \`app256continuation.js\` 的 V23 Core 不被 Physics 模組修改。

## Physics Multi-Task MLP

Physics 層一次輸出 48 維。正式版使用輕量 MLP，避免 48 組 boosting tree 造成 bundle 過大與手機端延遲。

- input：213D B/P/T history encoding
- hidden：64 -> 32
- output：48D
- browser：matrix multiply + ReLU
- 最後 residual 修正層仍為 XGBoost

## 48D output

- 3D：下一局 4 / 5 / 6 張牌概率
- 10D：閒家 0-9 點概率
- 10D：莊家 0-9 點概率
- 3D：莊 / 閒 / 和概率
- 13D：A-K 預期消耗張數
- 4D：四花色預期消耗比例
- 3D：已消耗總張數（0-416）、剩餘 low-rank / high-rank density
- 2D：莊減閒點數差、絕對點數差

low-rank = A-5；high-rank = 9-K。

## 物理限制

只提供 B/P/T 路徑時，無法識別真實未翻開的牌。Physics 48D 是標準 8 副牌離線 simulation 下的 conditional expectation，不是實際殘牌重建。

## Browser runtime

正式入口為 \`physics_residual_runtime.js\`，\`index.html\` 已切換到此 runtime。

Fail-safe：

1. Physics 48D + 56D XGBoost residual
2. 新模型載入失敗 -> 舊 7D residual
3. 舊 residual 也不可用 -> 原 V23 Core

## Production model files

- \`physics_multitask_model.json\`：browser MLP weights / scaler
- \`physics_multitask_model.joblib\`：Python training copy
- \`residual_bias_physics_model.json\`：browser XGBoost trees

## 自動建模

\`.github/workflows/physics-production.yml\` 會：

1. 跑 Python unit tests
2. 模擬標準 8-deck baccarat
3. 訓練 Physics 48D MLP
4. 用獨立 simulation shoes 產生 residual dataset
5. 直接呼叫現有 \`app256forward.js + app256continuation.js\` 取得 Core P(B)
6. 建立 \`Residual = actual_B - Core P(B)\`
7. 訓練 56D XGBoost
8. 驗證 portable tree traversal
9. Node 模擬 browser 跑 production smoke test
10. 全部成功才 commit model bundles

## 線上資料

瀏覽器 console：

\`\`\`js
__BGS_PHYSICS56__.getModelStatus()
__BGS_PHYSICS56__.getTrainingCount()
__BGS_PHYSICS56__.downloadTrainingData()
\`\`\`

和局不會誤用成下一個 B/P label；T 會直接清除 pending prediction。

## 安全邊界

\`\`\`text
delta = clip(raw_delta, -0.10, +0.10)
final_p_b = clip(core_p_b + delta, 0.0, 1.0)

Final P(B) > 0.50 -> B
Final P(B) <= 0.50 -> P
\`\`\`

沒有 PASS。
