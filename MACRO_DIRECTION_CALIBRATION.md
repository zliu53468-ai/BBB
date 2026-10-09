# Physical EV 三項巨觀 EMA 方向係數校準（2026-10-09）

## 結論與安全狀態

- 修正前 main: `fbca203dee5e1b60901c04131f07989a1ad0620f`。
- Frozen Core、Direct Physics 213→48D MLP、Particle500、57D XGBoost 與權重完全不動。
- `final_probability_model.json` 的 `decision_policy.macro_ema.coefficients` 仍是 **0 / 0 / 0**：不能把合成噪音當成實盤方向優勢。
- 已新增 `coefficient_units:"probability_delta"` 及一個離線校準/回放能力。日後選擇係數 0.01 代表對 *單位化* EMA Spread 的原始機率敏感度 0.01，而非先乘以 0.03 的舊制強度。
- 宏觀機率偏移仍限制在 `±max_delta`（正式設定 `0.03`），**再乘以原先的進度與噪音 gate**，而且不得越過 B/P 的預期值選擇邊界；Physical EV 僅在 Clip+EMA 前修正，主流程不變。
- 未提供可證明策略具有增益的完整真實靴，因此**沒有啟用非零的正式方向偏移**。

## 方向相關性：模擬而非保證

利用 `calibrate_macro_direction.js` 的標準八副牌洗牌、補牌、抽牌進度與完整 B/P/T 記錄，可重現 3000 靴實驗。每局預測特徵只讀**前一局為止**的 EMA；此局結算後才寫入 EMA。和局更新 EMA，但不當作方向標籤。

三個特徵都先做 short(.20) − long(.05)：
- `Six_Card_Spread`：六張牌比例短長差
- `Low_Score_Spread`：低勝點比例短長差
- `Point_Diff_Spread / 9`：兩家最終點數差絕對值的短長差，正規化到近似 ±1

合成資料未發現跨訓練集／校準集／獨立測試集同時穩定的方向信號；小幅相關不代表真實收益。具體精確數值由下列命令輸出。**不可直接將其中任一樣本內的正負號視作正式校準。**

## 使用說明

Node.js 20+：

```bash
# 1. 固定種子 20261009；3000 靴；標準八副牌合成資料
node calibrate_macro_direction.js --synthetic 3000 validation-local-macro-report.json

# 2. 使用網頁「匯出開牌資料」的完整真實靴，包含和局與未分析局
node calibrate_macro_direction.js bgs_observed_hands.json macro_real_report.json

# 3. 同一個最新 runtime 中，對照正式零係數與候選係數；不用改模型
node evaluate_macro_replay.js bgs_observed_hands.json macro_compare.json \
  --candidate-coefficients macro_real_report.json

# 4. 探索 signed 小係數的控制組（僅實驗；未獲真實資料支持）
node evaluate_macro_replay.js bgs_observed_hands.json macro_probe.json \
  --candidate-coefficients '{"six_card":-0.01,"low_score":-0.01,"point_diff":0.01}'

# 5. 檢驗
node test_macro_runtime.js
node test_macro_direction_calibration.js
python -m unittest -v test_macro_ema.py
```

校準結果的 `candidate_coefficients` 僅從完整靴分組的前 60% 訓練與接著 20% 校準產生，後 20% 完整靴為**不許調參**的獨立監測組。選入條件：訓練與校準各至少 30 靴／2000 筆非和局、|Pearson r| ≥ 0.01、按靴聚類 bootstrap 95% CI 不跨 0、兩組正負方向一致。候選值依弱／較強證據限制在 ±0.01 或 ±0.02。測試集 **只用於回報**，不依測試集改係數。這只是前篩，不代表通過即值得啟用；實盤仍須獨立評估多重比較、回放 EV 與風險。

## 可直接套用的 live 設定

JSON 層：`final_probability_model.json → decision_policy → macro_ema`：

```json
{
  "enabled": true,
  "coefficient_units": "probability_delta",
  "max_delta": 0.03,
  "progress_start": 0.3,
  "progress_power": 2,
  "min_observations": 12,
  "max_age": 3,
  "coefficients": {
    "six_card": 0.0,
    "low_score": 0.0,
    "point_diff": 0.0
  }
}
```

只有在**額外未參與選係數的真實完整靴回放**證實不惡化時，才能把三個 `coefficients` 手動更新成報告候選值。演算法本身不會寫入正式模型，也沒有更動 trees、base_margin、feature_names、Physics weights。

## 回放驗證驗收

`evaluate_macro_replay.js` 會輸出原設定 `current` 與實驗係數 `candidate`，包含 `all/early/middle/late`：
- `hit_rate_on_bets`：勝筆 ÷ 非和局出手筆數
- `realized_ev_per_bet`：固定單位投注總淨利 ÷ 全部出手（莊 +0.95、閒 +1、輸 −1、和 0）；這是事後實現值，不是理論 EV。
- `skip_rate`：Skip ÷ 可分析局
- `non_tie_bets / bets / wins / losses / tie_pushes`：觀察樣本量與風險。
- `max_macro_delta` 必須 ≤ 0.03；`no_flip_violations` 必須 = 0。
- 舊版和新版對同一靴，不准將當局點數帶進當局 snapshot；不能把當局結算後才存的 EMA 當作預測依據。

一律以完整靴／時間切分。樣本不足、標籤不完整、Pearson 不穩定時，結論是證據不足，而不是輸出正向收益預測。出手命中率不保證是獲利率；需要考慮莊家抽水與投注頻率。

## 快速關閉與完整回滾

**最快停用（只影響巨觀結構偏移）**：把 `final_probability_model.json` 的 `decision_policy.macro_ema.enabled` 改為 `false`，或者維持 `enabled:true` 但三個 `coefficients` 設為 0。前者停掉整個 EMA 偏移；後者保留觀測與巨觀 gate。正常模型仍可載入。

**Git 安全完整回滾**（本次基準 `fbca203`；先備份現有主分支，不使用 force-push）：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git branch backup-macro-direction-before-revert
git log --oneline fbca203dee5e1b60901c04131f07989a1ad0620f..HEAD
# 僅在此區間全部都是本次校準提交、沒有別人的新增提交時：
git revert --no-edit fbca203dee5e1b60901c04131f07989a1ad0620f..HEAD
git push origin main
```

如果區間內有其他人的提交，不可以執行整段 revert；改成逐一 revert 本次明確提交，避免回滾別人的程式碼。不要用 `git reset --hard` 或強制推送 main。
