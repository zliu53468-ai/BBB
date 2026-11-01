> 目前預設為「牌路主導／Physics＋PF微調」實驗版；上下文不足時由原算牌暫代。未驗證命中率提升，可一鍵回滾。見 [ROAD_PRIMARY_TEST.md](ROAD_PRIMARY_TEST.md)。

> 可選PF主導測試：網址加 `?pf_primary=1`；原PR75曾預設Physics主導；目前預設改為牌路主導，可用road_primary_v1.enabled=false回復。PF小樣本未見改善，不自動晉升。見 [PF_PRIMARY_TEST_MODE.md](PF_PRIMARY_TEST_MODE.md)。

> 目前EMA與PF混合上限均由低到高：EMA 42%→54%→62%、PF上限10%→14%→16%；點數張數修正保留。見 [ASCENDING_WEIGHT_EMA_V4.md](ASCENDING_WEIGHT_EMA_V4.md)。

> 新增點數／4–6張聯合模擬與不匹配樣本排除；仍為未驗證勝率的PF10,000實驗版。見 [POINT_CARD_SAMPLING_V2.md](POINT_CARD_SAMPLING_V2.md)。

> EMA已改前高後低：1–20局60%、45局52%、70局44%，PF權重不變。見 [EMA_DESCENDING_V2.md](EMA_DESCENDING_V2.md)。

> 已啟用可回滾的PF權重／平滑實驗組：前期PF上限12%、最高16%；詳見 [PF10000_WEIGHT_SMOOTHING.md](PF10000_WEIGHT_SMOOTHING.md)，未驗證勝率提升。

> 目前PF10,000粒子測試版：其他出手條件不放寬，手機耗時需實測，沒有命中率提升保證。見 [PF10000_UPGRADE.md](PF10000_UPGRADE.md)。

> PF1000測試版：每次獨立重建1,000粒子；原500粒子專用實驗校準暫停，未證明命中率提升。見 [PF1000_UPGRADE.md](PF1000_UPGRADE.md)。

> 最新：中後段分段平滑、PF 抽樣與本機校準升級，前 1～20 局對照鎖定；完整重訓另有 middle_late_focus 候選設定，正式權重未替換。見 [MIDDLE_LATE_UPGRADE.md](MIDDLE_LATE_UPGRADE.md)。

> 最新：所有開通成功的瀏覽器自動逐局累積本機校準，資料留在各瀏覽器；這不是完整模型重訓。詳見 [BROWSER_LOCAL_CALIBRATION.md](BROWSER_LOCAL_CALIBRATION.md)。

> 2026-10-27：新增閒／莊最終點數輸入與十類點數條件粒子後驗；B/P/T 模式仍可切回，瀏覽器只做逐局後驗更新，不重訓模型。詳見 [POINT_CONDITIONED_PF.md](POINT_CONDITIONED_PF.md)。

> 最新：新增分段平滑、PF 同向品質閘門、2 個百分點適中觀望與激進重訓候選。詳見 [CARDCOUNT_MODERATE_PROFILE.md](CARDCOUNT_MODERATE_PROFILE.md)。正式模型權重未更換。

> 最新：PF500 已回復，每次分析獨立重建 500 粒子；保留觀望切換。見 [PF500_RESTORED.md](PF500_RESTORED.md)。

> 本次修改：算牌機率優先、莊閒方向獨立顯示、牌路規律僅輔助。詳見 [PHYSICS_FORECAST_FIRST.md](PHYSICS_FORECAST_FIRST.md)。模型與開通碼邏輯未更換；預測方向與下注資格分開。

> 2026-10-09：Physics-primary 巨觀 EMA 與動態門檻更新，完整程式碼、參數、驗證結果與回滾方式見 [MACRO_EMA_UPGRADE.md](MACRO_EMA_UPGRADE.md)。EMA 方向係數尚未校準，預設零偏移；下方保留歷史架構說明。

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
