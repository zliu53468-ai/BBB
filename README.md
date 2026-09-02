# BGS 128D Frozen V2 Multi-Expert 測試面板

目前 `main` 使用 **128D = 64D 牌靴 + 64D 牌路**，方向核心已由 Frozen Prior V1 升級成 **Frozen V2 Multi-Expert Contextual Arms**。

## 核心限制保持不變

正式操作期間全部關閉：

- 不 bootstrap
- 不 Walk-forward
- 不 replay
- 不結算上一筆 prediction
- 不更新 A/b
- 不 decay

`updates` 永遠維持 0。

## 為什麼升級到 V2

V1 的主要問題是把大量方向特徵一起乘上同一個 `directionalMode`。一旦 Trend / Reversal regime 判錯，整組方向訊號可能同時翻面，容易把單局判斷錯誤延續成連續錯誤。

另外 V1 同時使用大量高度相關的 Banker Ratio 視窗，容易讓同一種訊號被重複投票。

V2 將這兩個問題拆開處理。

## 128D 結構

- 001–064：牌靴／進程特徵
- 065–128：牌路／結構特徵

牌靴端保留 penetration、shoe phase、樣本支撐、和局比例、entropy、balance 等資訊；因按鈕版沒有真實 A～10/J/Q/K 殘牌資料，所以不虛構 exact composition 方向訊號。

牌路端保留 current side、run length、Banker ratio、Turn rate、hazard、HSMM stability、大眼仔、小路、蟑螂路、run 統計、交替與同邊結構。

## Frozen V2 四專家

### 1. Trend Expert

判斷目前方向是否具有延續條件，綜合：

- current side
- 8 / 16 / 32 局方向結構
- run hazard
- Turn Rate
- HSMM stability
- current run length

### 2. Reversal Expert

獨立判斷轉折，不再把全部特徵一起乘負號。主要使用：

- 4 / 8 局 Turn Rate
- run hazard
- alternating tail
- HSMM stability
- current side 的反向候選
- 短窗相對長窗的 shift

### 3. Derived-road Candidate Expert

對下一局分別模擬：

```text
History + B
History + P
```

再比較兩個候選對：

- 大眼仔
- 小路
- 蟑螂路

的結構符合程度，得到 `derived B support`、`derived P support` 與方向差值。

這只是 deterministic candidate simulation，不是 replay，也不會更新模型。

### 4. Multi-scale Divergence Expert

方向核心不再讓 2/3/4/5/6/8/10/12/16/20/24/32/48 等大量高度相關窗口重複投票，而聚焦：

- 4 局
- 8 局
- 16 局
- 32 局

並使用：

```text
ratio4 - ratio16
ratio8 - ratio32
```

捕捉短期與中長期方向分歧。

## Regime Consensus

V2 同時計算：

- short regime
- middle regime
- long regime

至少兩個尺度同方向時，Trend 或 Reversal expert 才會獲得較高權重。

如果短、中、長尺度衝突，V2 不會整組翻面，而是：

- 降低 Trend / Reversal 權重
- 提高 Derived-road Candidate / Divergence 權重
- 壓縮最終方向幅度

目標是降低錯誤 regime 連續支配多局的風險。

## Frozen Contextual Arms

V2 仍保留固定 B/P Contextual Arms 與 128D normalization，但 base prior 權重被刻意降低，只使用 4 / 8 / 16 / 32 等較少的方向窗口作弱基礎訊號。

主要方向由四專家 Soft Fusion 決定；uncertainty 僅保留在診斷與 confidence calibration，不再用探索項強行改變 B/P 方向。

## Storage / 舊版本

新 storage key：

`bgs128d_frozen_v2_multi_expert`

第一次載入 V2 時：

- 保留舊 B / P / T 歷史
- 舊 A/b 不沿用
- 不做 migration training
- 不結算任何舊 prediction

備份分支：

- `backup-frozen-prior-v1-before-v2`
- `backup-128d-blankbrain-before-frozen-prior-v1`
- `backup-64d-f6c24d3-before-128d`

## 測試重點

不要只比較整體命中率，也應比較：

- 最大連錯
- 3 連錯出現頻率
- 連勝長度分布
- Trend / Reversal 切換後前 3 局表現
- Derived candidate B/P support 是否穩定
- 64D、128D V1、128D V2 在同一批未參與設計資料上的差異

## 限制

目前 repository 仍沒有可供 Train / Validation / Test 使用的多靴歷史資料，因此這版是透明的 engineered Frozen V2，不宣稱是 dataset-pretrained 模型。

這是模型架構測試工具，不代表任何保證勝率；維度增加或多專家融合是否有效，仍需要用獨立牌靴資料驗證。