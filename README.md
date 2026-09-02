# BGS 128D Frozen Prior Contextual LinUCB 測試面板

目前 `main` 是在 128D（64D 牌靴 + 64D 牌路）基礎上，改成 **固定 Frozen Prior** 的測試版本。

## 這版的核心限制

正式操作期間全部保持關閉：

- 不 bootstrap
- 不 Walk-forward
- 不 replay
- 不結算上一筆 prediction
- 不更新 A/b
- 不 decay

`updates` 永遠維持 0。

## 為什麼不是直接稱為 pretrained A/b

目前 BBB repository 裡沒有可供 Train / Validation / Test 使用的歷史多靴資料，因此這一版不假裝已經做過資料預訓練。

目前使用的是：

- `engineered_static_prior_v1_no_training_dataset`
- 固定 B/P 雙臂 prior
- 固定 A（以 diagonal precision vector 表示）
- 固定 b
- 固定 normalization
- 固定 deterministic regime gate

之後若提供歷史多靴資料，可以保留同一套執行介面，只把固定 engineered prior 替換成真正離線訓練、驗證後凍結的 A/b。

## 128D 結構

- 001–064：牌靴／進程特徵
- 065–128：牌路／結構特徵

牌靴端保留進度、penetration、樣本支撐、和局比例、B/P entropy、B/P/T entropy、balance 與不同階段 basis。

牌路端保留 current side、run length、Banker ratio、Turn rate、hazard、HSMM stability、大眼仔、小路、蟑螂路、run 統計與交替／同邊結構。

## Frozen Prior V1 的設計

### 1. Fixed normalization

128D Context 在進入雙臂 prior 前會先轉為固定中心化尺度，不會在執行期間重新估計 mean/std，也不會隨使用者資料改變。

### 2. Fixed regime gate

根據目前歷史的：

- Turn rate
- run hazard
- HSMM stability
- alternating tail
- same-side tail
- entropy / volatility
- derived-road support

產生固定 deterministic 的 `trend / reversal / neutral` regime 與結構可信度。

這不是學習器，也不會修改任何參數。

### 3. Shoe / Road gating

牌靴資料主要控制目前樣本成熟度與結構可信度；因按鈕版沒有真實 A～10/J/Q/K 殘牌資料，所以不讓牌靴進度本身虛構成 B/P 方向訊號。

Road directional features 才負責提供 B/P 方向證據，並受到 Frozen regime gate 的固定縮放。

### 4. Fixed B/P A/b

B、P 兩個 arm 各自有固定 b；A 使用固定 diagonal precision representation。

程式啟動時由 source code 重建：

```text
A_B = fixed diagonal precision
b_B = A_B × theta_B

A_P = fixed diagonal precision
b_P = A_P × theta_P
```

正式預測時只做：

```text
128D raw context
→ fixed normalization
→ deterministic regime gate
→ frozen model context
→ fixed B/P LinUCB scores
→ argmax(B, P)
```

沒有任何 feedback update。

## Tie handling

已移除「上一把選 B，平手就切 P」這類依賴上一個 prediction 的交替規則。

只有在 B/P score 真正完全平手時，才使用固定 history hash 做 deterministic tie-break；同一段歷史永遠得到相同結果。

## 舊資料處理

新 storage key：

`bgs128d_frozen_prior_static_v1`

首次載入時如果偵測到舊 128D / 64D / 32D localStorage：

- 只保留 B / P / T 歷史
- 舊 A/b 完全不沿用
- 不做 migration training
- Frozen Prior 直接由目前 source code 重建

舊版本備份：

- `backup-128d-blankbrain-before-frozen-prior-v1`
- `backup-64d-f6c24d3-before-128d`

## 真正 pretrained 的下一步

如果要把這版升級成真正 Frozen pretrained A/b，需要另外準備多靴歷史資料，依整靴分成 Train / Validation / Test，離線建立 A/b、選 ridge / alpha，再把最終參數寫入前端。

正式前端仍然可以維持本版所有 Frozen 限制，不需要加入線上學習。

## 注意

這是模型架構測試工具，不代表任何保證勝率；128D 或 Frozen prior 是否有效，仍應使用未參與設計的獨立測試資料比較。