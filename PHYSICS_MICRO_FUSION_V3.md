# BBB Physics Micro-Fusion V3 — 2026-11-01

## 變更概要

延續已上線的 **BASE_EV_GATE=0.005、OVERDRIVE_GATE=0、ONE_HAND_CUSHION=0.035**，保留八副牌、物理 5× / 路單 0.1×、已驗證的完整逐局牌值 EV Gate、登入驗證及舊版歷史重播。增加物理盲估成對消耗監控、靴尾研究訊號平方敏感度和帶 Beta(4,4) 收縮的貝氏風格融合。所有高頻方向都是研究訊號，不保證可賭贏。

## 1. 和局／補牌的條件聯合抽樣（不偽造真實扣牌）

- `quant256_online_runtime.js` 的 `sampleHand()` 對 8 副牌點數 `0...9` 按實際牌數不放回抽取，遵循百家樂自然牌與第三張牌規則，回傳 `consumed[0..9]`、結果 B/P/T、4/5/6 張數。
- 只有當樣本符合輸入的 B/P/T（若有可驗證實際發牌張數也符合 `observedDraws:4|5|6`）才更新該粒子的虛擬殘牌。
- 和局 T 或模擬 5/6 張發牌時，記錄 `pairwise_depletion`：點數 0、1、4、7 的平均共同耗用量與共同出現比率。**僅供監控**，不會額外扣兩倍任何點數。
- 為何不硬性雙倍扣牌：和局與補牌**不能證明**特定牌值加倍消耗。硬扣會違反不放回抽牌的合法組成，造成假精準、假殘牌。無法僅靠 B/P/T 證實 95% 真實牌靴同步。

## 2. 靴尾非線性放大（只影響訊號，真實 EV 不變）

- `penetration_ratio = clamp(current_hand / 70, 0, 1)`，`phase_scale = 1 + penetration_ratio^2 * 1.5`。所以首局 1×、第 35 局 1.375×、70 局以上最多 2.5×。
- 這個 `current_hand` 是已輸入局數，**不是**真實耗牌深度。理論原始 EV `ev_banker`、`ev_player`、`ev_gap_deviation` 完全不修改。
- 僅 `research_score = abs(raw_ev_gap_deviation * phase_scale) * confidence_multiplier` 參與動態研究訊號門檻；不能把倍率當成下注期望值或真正勝率提升。

## 3. Beta-Binomial 路單傾向與弱權重共振

- 逐個歷史 B/P 非和局建立**只看上一局以前**的路況 key（近兩局同側／交替、長龍與否、近五局變向頻率），標籤為下一局是否延續。以 Beta(4,4) 收縮：`P(stay) = (4+continuations)/(8+observations)`。
- `road_bias = P(B | road regime) - 0.5`、`support=n/(n+16)`。樣本不足或 1.28 倍後驗標準差未偏離 0.5 的情況不視為可信衝突。
- 用 `logit(P(B|physical,B/P)) + 0.1 * support * clip(logit(P(B|road)),±0.6)` 當作 Bayesian-inspired **診斷後驗**。未經獨立校準，不能聲稱是真正的 Bayesian evidence 或勝率。
- 物理／路單方向可靠地一致：研究訊號強度增幅最多 10%；可靠地衝突：研究訊號強度降幅最多 10%，並對該 **regime 首次衝突**短暫標為觀望；同一 regime 連續衝突不會無限觀望。
- **路單不得改變物理 EV 方向，也不得將虛擬 B/P/T 模擬變成正式 BET 資格**。原本的動態 0.005／0／0.035 仍存在，短暫衝突觀望僅為附加研究策略。

## 測試與操作

- `node test_physics_forecast_first.js`：驗證靴尾 1/1.375/2.5 倍、原始 EV 不變、弱路單後驗、單次衝突觀望、重播與 EV 資料來源驗證。
- `node test_quant256_online.js`：驗證和局/補牌條件抽樣實際扣除 4–6 張、不同牌值不超出合理殘牌、面板顯示和舊版按鈕。
- 用同一條正式網站入口登入；若載入舊快取，按 Ctrl+Shift+R。完整真實發牌值匯入 API 沿用既有 `setObservedRounds(rows)`。

## 局限

沒有觀測 A～K 的實際出牌時，模擬 EV、路單後驗與共振數字都**不是實際可套利 EV**。連敗不可能由這些設定確定阻斷；應以留出靴的前推式回測同時觀察命中率、出手密度、連敗最長長度和負期望值。