# BBB Card Marginal / Conditional SMC V4（2026-11-01）

本次僅調整「B/P/T 隱藏扣牌／物理 EV 估計」和最終方向的字樣。原有 **0.005 基礎研究門檻、連勝 0.0、錯局緩衝 0.035** 完整保持，不新增任何新的觀望門檻。保留 8 副牌、256D 物理主導與弱路單融合、授權／原始輸入歷史。

## 1. 真正的條件殘牌粒子重採樣，而非固定大小牌扣除

`quant256_online_runtime.js` 的 `BlindPhysicsEstimator`：
- 每個粒子以 8 副牌點數真實不放回的機率，依標準自然牌與第三張牌規則，抽取多筆合法發牌路徑。
- 已知本局結果 B/P/T（若另提供真實 4/5/6 張數，還同時符合張數）才會接受該樣本，估計 `P(outcome | hypothetical shoe)`。
- 各候選粒子以該結果的**條件似然**進行系統重採樣，每個父粒子保留最多 8 個 reservoir 樣本以保持多條可能合法出牌路徑，避免原本每個粒子都找一個匹配結果、忽略不同殘牌產生該結果機率的缺陷。
- `marginal_card_weights` 回傳 0～9 各點數的「條件局平均消耗 − 所有合法局平均消耗」正規化相對差；`small_1_4_excess` 與 `large_5_8_excess` 是**數據計算出的相對超額消耗**，不是硬性正負權重。
- 和局 T 的補牌及 0/4/8 等點數同樣由條件抽樣自然表現，**不會強制額外扣除**，否則會破壞實際 8 副牌張數守恆。
- 新增 `effective_particle_size`（ESS）、`conditional_outcome_likelihood`、逐點消耗矩陣與可重現隨機種子。只有 B/P/T 的條件下，這些資料**仍不能當作實際扣牌或證實 95% 殘牌一致**。

## 2. 從標準百家樂機率推導的單張牌邊際 EV 矩陣

`physics_forecast_first_runtime.js` 新增 `oneCardRemovalLikelihood(counts, hand, point)`、`accumulateMarginals` 與 `finalizeMarginals`，對點值 0～9 逐一估算「假設下一張額外消耗此點值」時的莊／閒 EV 變化。

對由殘牌 C 抽取的 m 張合法發牌路徑 h，移除一張點值 k 後的路徑機率比率為：

`P(h | C − e_k) / P(h | C) = ((C[k] − consumed_k(h)) / C[k]) × (N / (N − m))`

其中 N 是原始剩餘牌數。這個比率利用**同一批合法蒙地卡羅樣本做重要性重新加權**，不需另跑十份獨立的蒙地卡羅；支援自然牌、4/5/6 張合法路徑及非線性補牌。輸出每個 k 的 `delta_ev_banker`、`delta_ev_player`、`delta_ev_gap`，並彙整 A～4、5～8 的平均邊際量。正負號由實際分布與補牌規則算出，而非強行假定小牌消耗必定利莊、大牌消耗必定利閒。

**原有 `EV_B = 0.95P(B) − P(P)`、`EV_P = P(P) − P(B)` 不修改。** 若輸入整靴完整實際發牌值，來源標註已觀測；僅 B/P/T 則標註虛擬粒子 MC，不能偽稱真實機率優勢。

## 3. 靴尾放大保留，但不篡改 EV

`shoe_progress = clamp(current_hand / 70, 0, 1)`

`scale = 1 + shoe_progress² × 1.5`

`final_bias = (ucb_banker − ucb_player) × scale`

`final_bias` 現在保留在模型診斷輸出，**不作為真實 EV**。原本 V3 的 `research_score = |raw EV gap deviation × scale| × resonance` 繼續參與研究訊號，避免二次放大；真實 EV 和證據驗證閘門仍使用沒有放大過的原始物理概率。中後段倍率不是命中率保證。

## 4. 主介面最後方向

- 大字從「莊（研究）」／「閒（研究）」精簡成**「莊」／「閒」**。
- 一次性物理與路單衝突時仍顯示「觀望（微觀衝突）」。
- 下方維持 `SIMULATION ONLY` 或 `EV VERIFIED` 的證據狀態，以免將未觀測牌值的模型方向冒充已確認的實際套利機會。
- 提供大小牌邊際 EV 變化與粒子 ESS 診斷，不改動登入、授權、歷史、原始 B/P/T 按鈕。

## 5. 測試

`node test_quant256_online.js` + `node test_physics_forecast_first.js`，並由 `.github/workflows/quant256-online.yml` 執行。

測試涵蓋條件似然重採樣、逐點超額消耗、完整八副牌守恆、重要性比率與逐張機率直接相除的一致性、B/P/T 與真實牌值來源隔離、`0.005/0/0.035` 門檻不變、歷史重播、手機面板移除「研究」標籤以及版本載入。

**限制**：正統物理算牌需要真實發牌點數或可觀測殘牌。僅 B/P/T 不能辨認實際牌值；新增的條件化和邊際效應會改善模型計算定義與減少某些推估偏差，但不能證明提高命中率或阻斷連敗。
