# BBB LightGBM + XGBoost Residual Ensemble

這一層只掛在既有 256D/V23 Frozen Base 外部，不修改核心權重。

## 架構

1. Frozen Base 先輸出 `P_core`。
2. LightGBM 與 XGBoost 都預測 `actual_B - P_core`。
3. 兩個 residual 以 50/50 平均融合。
4. 最終 `Delta` 嚴格裁切在 `[-0.10, +0.10]`。
5. `final_p_B = clip(P_core + Delta, 0, 1)`。
6. `final_p_B > 0.50 -> B`，否則 `P`，沒有 PASS。

## 416 張牌物理特徵

固定以 8 副牌 / 416 張、最多 70 局為基準：

- `core_confidence`
- `current_hand`
- `avg_cards_per_hand = (416 - remaining_cards) / current_hand`
- `remaining_cards_ratio = remaining_cards / 416`
- `shoe_progress_delta = current_hand / 70 - (416 - remaining_cards) / 416`
- `sx_markov_p_same`
- `stage`
- `depth`

瀏覽器只有 B/P/T 時無法知道真正用了幾張牌，因此 runtime 支援外部提供實際 `remainingCards`。若未提供，只會使用明確標記為 `estimated` 的 deterministic fallback。

```js
window.__BGS_DUAL_RESIDUAL__.setRemainingCards(250)
window.__BGS_DUAL_RESIDUAL__.clearRemainingCards()
window.__BGS_DUAL_RESIDUAL__.setEstimatedCardsPerHand(4.9)
```

## 訓練

```bash
pip install -r requirements-ensemble.txt
python ensemble_residual_bias.py --input training.json --output ensemble_residual_model.json
```

訓練資料的 B/P 實際結果會轉成 `actual_B = 1/0`，兩個模型共同學習同一 residual target。未訓練前 `ensemble_residual_model.json` 為 `trained:false`，因此 `Delta=0`，網站只維持 Frozen Base 原始結果。
