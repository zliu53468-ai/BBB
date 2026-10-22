# 更敢出手｜平滑跟牌｜重訓尖銳｜再降觀望｜算牌優先規律 soft

本版以前一提交 `7b33570` 為修改基準；最初 main 為 `0bbd1f7db966be5c67f00809fa4db1822386ac21`。本文件取代舊版對平滑、重訓與出手 profile 的數值描述。

## 平滑表

共用設定檔為 `cardcount_profiles.json`。Macro JS 內有相同的離線安全預設，測試逐項核對；Python 直接讀取 JSON。

| Macro profile | short α | long α | max_delta | min_obs | max_age | progress_start | power | noise low／high |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| conservative | .20 | .05 | .030 | 12 | 3 | .30 | 2.0 | .60／.75 |
| **responsive_v1（預設）** | **.32** | **.11** | **.040** | **7** | **5** | **.22** | **1.6** | **.55／.72** |
| aggressive | .35 | .12 | .045 | 6 | 5 | .20 | 1.5 | .55／.72 |

`max_delta` 硬上限 .045。缺測略過、不當零；新靴清空；本手寫入後立即更新，下一手讀取已更新狀態；復原、重新分析、重新載入會依觀測 journal 重建，不重複累加。和局可更新 Low_Score=0，未知張數／點數不補值。

正式方向係數仍為零，沒有假裝已校準。`observations_ready` 表示資料較早就緒，不代表零係數已可輸出方向加成。B/P/T-only 無法反推出真實張數／點數，因此只有結果按鈕時，不能宣稱 Macro 正在估出未觀測牌值。

既有最終機率 EMA 沿用，沒有新增一層。舊 production 配置 off 原本會走隱含 fallback；現改為明確的 responsive_v1 fallback。進程 0／.20／.35／.50／.65／.80／1 的 α 分別為 .52／.55／.60／.65／.70／.73／.75。清楚物理訊號（noise<.60、uncertainty<.45、距離 .5 至少 .03）可再增加 .06，最終 α 仍限制 .35～.75。這會減少歷史值拖慢新訊號，不額外把機率拉回 .5。

新訓練 bundle 以 `runtime_config_schema=1` 標示選定平滑，瀏覽器尊重該選擇，包含明確停用；不覆蓋已經驗證過的模型選擇。舊 bundle 權重與 JSON 沒有更換。

## 決策表：aggressive_entry_v2

此 profile 已實作且可切換。預設出手政策仍為 production，避免在尚無真實 holdout 證據時直接換成較積極政策。Python 和瀏覽器都讀同一張 JSON 表，測試涵蓋 EV／noise／dynamic band／EMA alpha 的數值一致性。Python 訓練端的 XGB surrogate 不取代完整 Physics-primary 瀏覽器 pipeline；晉升以實際 runtime 回放為準。

| 參數 | early 1–40 | middle 41–50 | late 51+ |
|---|---:|---:|---:|
| 現行 production EV | .002 | .002 | .000 |
| aggressive EV min（非負截斷） | .000 | .000 | .000 |
| 要求降幅／截斷後實降 | .008／.002 | .006／.002 | .005／0 |
| clean relief | 0 | .0035 | .004 |
| clean soft band | 0 | .003 | .004 |
| confidence band（動態調整前） | .005 | .004 | .003 |
| strong margin | .002 | .002 | .001 |

共同參數：noise_threshold=.75、max_noise_ev_penalty=.0035、min_confidence=.0001、極端 noise≥.90 不出手、距離 .5<.002 不出手。原本 min_confidence 是 .0003 的機率點數尺度，不能機械再減 .02～.04；本版降至非負的 .0001。表中 soft band 是數值，不把 boolean true 當成 1。

Volume Guard 保留現行 window=12、min_history=5、expected_correct_floor=.93、max_relief_fraction=.25；目標出手率僅由 .58／.60／.66 微調為 .60／.62／.68，band relief=.010、EV relief=.006 不加大。Dynamic Band 保留進程／noise gate，minimum 依階段為 .0025／.0015／.0015；極端 noise 與距離 .5<.002 的硬性限制不可被 Volume Guard 繞過。原有 Dynamic Clip、Physics candidate、三階段 EV、noise penalty、Core／57D no-flip 都保留。

**目前 production bundle 的 EV 門檻已是 .002／.002／0，低於 hard_ev 參考值。** 因此本版按現行基準減去要求幅度後截斷為 0，而非把舊 hard_ev 減成 .012／.004 後反而變保守；最終預測 EV 仍必須嚴格大於非負閘門。出手率與品質仍須看完整物理、confidence、平滑及 Volume Guard 的逐靴回放。控制案例已測出對 hard_ev／strict 參考可增加出手，不代表真實命中率／EV 提升。

啟用方式：分析頁模型載入完成後呼叫 `window.__BGS_FINAL56__.setEntryProfile("aggressive_entry_v2")`，只影響本次頁面；或在載入前指定 `window.BBB_ENTRY_PROFILE`。要改成部署預設，先在候選分支將 JSON 的 `default_entry_profile` 設為該名稱，再做完整真實 holdout 驗收。回復用 `setEntryProfile("production")`。Macro 可用 `setMacroProfile("conservative")` 或 `setMacroProfile("responsive_v1")` 切換，會重建該靴觀測 EMA。

## 規律不得帶偏

先以物理／Core／57D 的 final_p、noise、EV、confidence 計算出手資格；Volume Guard 只能在既有限度內調整資格，極端 noise 不可繞過。將結果固定為 `base_eligible` 後，才計算 pattern soft bonus。

預設 pattern_extra_ev_relief=.003、pattern_only_if_base_eligible=true。base 不合格、noise>.60、距離 .5<.02、EV／信心非正時，加成強制零。加成只記錄已合格出手的 soft 門檻，不回流到 B/P/Skip、機率或注碼。因此規律本身不會減少 Skip，不能翻向、不能使漂亮路單硬過關。長龍／單跳／雙跳／轉折只作報告及最多 +10% 的保守 sample weight。

## 重訓表

| 項目 | 本版建議與預設 |
|---|---|
| Physics | 5000～8000 靴；預設 5000；validation .20、calibration .10、augment .25 |
| 主任務權重 | Winner 2.80、點數 1.20、張數 1.30 |
| 次要任務 | Rank .55；composition .45；沒有新增 Suit head 或輸入 |
| 校準 | 保留 affine／temperature／isotonic uncertainty |
| XGB | depth 2～4、400～800 trees、lr .015～.030、child 6～16、subsample .85～.95、colsample .75～.95、L1 .10～.60、L2 8～18 |
| sample weight | 41–50 局 1.05～1.10；51–70 局 1.20～1.30；低 uncertainty 最高 +18%；規律上限 +10% 且高 uncertainty 關閉 |
| 選模排序 | Brier → hit/bet → EV/bet → Skip delta；pattern hit 只報告 |

原比較模型 320 trees 只作 baseline，不計入新搜尋區間。樣本仍依原 class balancing 處理，整體權重正規化均值 1 並限制 .35～1.50；物理 actionability 只用當時模型預測，不使用本手標籤。Identity／Platt／Isotonic 依 chronological calibration probe 的 overall Brier 選擇，尾端 Brier／hit／EV 另報告，不拿 holdout 調參。

本版只更新訓練預設與搜尋程式，未重訓或更換 production 模型。指令示例見 `RETRAINING_TOP_PLAN.md`。

## 驗收與報告

```
node evaluate_macro_replay.js real_holdout.json report.json PRODUCTION_DIRECTORY --entry-profile aggressive_entry_v2
node tools/no_pf_acceptance.js report.json
```

可加 `--candidate-model candidate_final.json`、`--candidate-physics candidate_physics.json`，不覆蓋正式檔即可回放；`--macro-profile` 可切換平滑。輸入必須完整、連續、chronological whole-shoe；回頭插入已結束的靴會拒絕。先預測再結算／更新觀測。

報告包含 Skip、下注數、hit/bet、EV/bet、EV/row、Brier、最大連敗、分階段及長龍／單跳／雙跳等。最大連敗按下注結果計算，Skip／和局不算輸贏，每靴重新計數。

安全驗收採較嚴格的 hit／EV 各自不退步，不只禁止雙降；Brier 容差 .0015、Skip 上升最多 +5pp。合成資料不允許發布晉升。更換模型還必須以 `tools/verify_candidate_promotion.js` 核對已測 candidate、當前 production 模型及 runtime/config 的 SHA-256，避免把別份模型的好成績套到新權重。CI 缺真實 production 對照報告即不提交生成模型。人工仍須確認 holdout 從未用於調參，工具不會把自填標籤當成資料真實性的證明。

`validation/responsive_default_smoke.json`、`validation/aggressive_entry_v2_smoke.json` 僅為固定 seed 的合成機制回放，不能當真實提升或獲利證據。當下沒有真實訓練／holdout 資料，所以不啟用新權重，也不宣稱已達成實際低 Skip 且命中率／EV 不退步。

## 架構及回滾

Frozen Core 兩支程式、Physics 213→64→32→48 拓撲、57D 名稱／順序／定義、P=0／B=1 目標、production JSON／joblib 權重及開通碼邏輯不修改。PF 仍不參與；相容欄名 `particle_uncertainty` 仍只是 direct uncertainty ABI。

只回滾本版：revert 本次繁體中文提交，回到 `7b33570`，PF 仍移除。要比較最初版本可在獨立 checkout 使用 `0bbd1f7`；整體 PF 回滾見 `PF_REMOVAL.md`。不要把新 config、舊 runtime、候選權重混合部署。Macro 新舊速度切換會由 journal 重建，不需遷移會員授權資料。

## 本次完成紀錄

- PF 不載入、不呼叫，無 PF 不會強制 Skip。
- Macro 預設 responsive_v1，三組 profile 可配置，方向係數仍零；新增早就緒、切換跟隨、缺測／和局／重設及 JS／Python parity 測試。
- aggressive_entry_v2 可切換，基本資格先固定，pattern 不能破 Skip 或翻向；極端 noise、接近 .5 仍觀望。
- 新搜尋／權重／校準建議與發布閘門已完成；production 模型權重沒有替換。
- 60 項 Python 測試與 runtime／按鈕／規律安全／promotion 測試通過。控制案例證明 Skip 可下降，只驗證機制，不代表真實成效。

12 靴／840 手固定 seed 合成回放（828 次分析）：

| 版本 | wagers | Skip | hit/bet | EV/bet | 最大連敗 |
|---|---:|---:|---:|---:|---:|
| 原 main | 220 | 73.43% | 50.50% | -0.00114 | 6 |
| responsive 預設＋production entry | 239 | 71.14% | 47.71% | -0.05063 | 6 |
| responsive＋aggressive_entry_v2 | 250 | 69.81% | 48.25% | -0.04100 | 6 |

雖然合成回放出手增加，但命中率與每注實現 EV 下降，**未通過品質晉升要求**。這是回報失敗結果，不在這批資料上反覆挑選較好成績的參數。aggressive 保持可切換候選，不把它自動設成 production；沒有實際重訓／獨立真實 holdout，不宣稱成效目標已達成。
