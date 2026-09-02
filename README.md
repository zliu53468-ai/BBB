# BGS 256D（128D＋128D）Frozen Direct 測試面板

目前 `main` 是以使用者指定的基準 commit `c3420c274bd7acf1096e44178de482297ae542e4` 為底，將原本 128D（64D 牌靴 + 64D 牌路）對稱擴充成：

- 128 維牌靴／進程特徵
- 128 維牌路／結構特徵
- 合計 256D Context
- two-arm LinUCB：B / P

## Frozen Direct 限制

正式操作流程完全維持原基準版：

- 不 bootstrap
- 不 Walk-forward
- 不 replay
- 不結算上一筆 prediction
- 不更新 A/b
- 不 decay

莊／閒／和按鈕只加入歷史；只有按下「開始分析」才用目前完整歷史重新計算 256D Context 並直接預測下一局。

## 256D 結構

### 001–128：牌靴／進程

前 64 維延續 c3420c2 的 128D 版本原有牌靴語意，再增加 64 維多尺度特徵，包括：

- 2 / 3 / 5 / 7 / 10 / 14 / 20 / 28 / 40 / 48 / 56 / 64 局 Tie ratio
- 同窗口 B/P entropy
- 同窗口 B/P/T entropy
- 同窗口 B/P balance
- penetration / remaining cubic 與 quarter-root
- 更細四分位 shoe phase basis
- 4 / 12 / 20 / 32 局 sample support
- 短長窗 tie / entropy / balance delta

按鈕版仍沒有真正的 A～K 牌值輸入，因此既有 rank relative ratio 維持 neutral，不虛構實際殘牌組成。

### 129–256：牌路／結構

前 64 維延續 c3420c2 的原有路單特徵，再增加 64 維，包括：

- 更多 7～72 局 Banker ratio 與 Turn rate 多尺度
- 大眼仔 / 小路 / 蟑螂路 6 / 12 / 24 / 32 窗口 regularity
- 更長的 previous run 與 6 / 12 run 統計
- 3 / 5 / 8 / 10 局 alternating 結構
- 6 / 7 / 8 / 10 局 same-side 結構
- Banker ratio 的 4–16、8–32、16–64 短長窗差分
- Turn rate 的同尺度差分
- hazard / continue / HSMM / derived consensus 非線性特徵

## 舊資料相容性

新 storage key：

`bgs256d_128plus128_frozen_direct_tech_panel_v1`

如果瀏覽器只有舊 128D / 64D / 32D localStorage：

- 保留 B / P / T 歷史
- 不沿用尺寸不相容的舊 A/b
- 自動建立新的 256×256 A 與 256 維 b

## 備份

升級前的目前版本已保留：

`backup-before-reset-to-c3420c2-256d-test`

指定的 128D 基準仍可由 commit：

`c3420c274bd7acf1096e44178de482297ae542e4`

精準恢復。

## 測試定位

256D 只是受控的維度擴充實驗，不代表勝率一定會提高。建議使用與 128D 相同的完整牌局序列，比較：

- 總命中率
- 最大連錯
- 三連錯發生次數
- 長龍 / 單跳 / 雙跳 / 轉折段表現
- 256D 是否因新增多尺度特徵變得過度敏感

這是模型架構測試工具，不代表任何保證勝率。