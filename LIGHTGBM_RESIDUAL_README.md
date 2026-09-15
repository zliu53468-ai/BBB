# BBB LightGBM 雙層殘差修正層

本模組不修改 256D Frozen Base / V23 核心。第一層核心只提供 `P_core`；第二層 `LGBMRegressor` 學習 `actual_B - P_core` 的連續殘差，並將 Delta 限制在 `[-0.10, +0.10]`。

正式特徵：`core_p_b`、`round_index`、`estimated_total_hands`、`remaining_ratio`、`sx_markov_p_same`、`stage`、`depth`。

決策公式：`final_p_b = clip(core_p_b + delta, 0, 1)`；`final_p_b > 0.5` 輸出 B，否則輸出 P。沒有 Pass。

瀏覽器無法直接執行 Python，因此 `lightgbm_residual_bias.py` 負責訓練並匯出可攜式樹 JSON；`lightgbm_residual_runtime.js` 在 GitHub Pages 端以相同特徵順序執行樹推論。未訓練模型 `trained=false` 時 Delta 固定為 0，等同 Frozen Base 原結果。

訓練示例：

```bash
python lightgbm_residual_bias.py train --input training.json --output lightgbm_residual_model.json
```

訓練器使用固定 `random_state`、單執行緒、`deterministic=True` 與 `force_col_wise=True`，並以 `shoe_id` 雜湊固定切 validation，降低同一份資料重訓時的非必要差異。
