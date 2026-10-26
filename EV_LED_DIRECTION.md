# 補牌後 EV 主導莊閒方向｜2026-10-26

本次為決策選向升級，沒有重訓或替換模型。版本為 PHYSICS_57D_V14_8_EV_LED_DIRECTION。

## 計算與資料流

B/P/T → Direct Physics 與原有 PF 補牌模擬 → Clip / Macro EMA → Core / 57D XGB 與原有決策平滑 → 補牌後莊閒和機率 → 莊閒淨 EV 比較選向 → EV / confidence / noise / Volume Guard → 正式 B/P/Skip。

PF 的抽牌與百家樂補牌規則完全保留，EV 在每手補牌完成的結果分布上評價押莊與押閒；不以 EV 挑選有利抽牌或重新加權模擬樣本，以免扭曲機率。PF 只在原有位置混合一次。

設無條件機率為 B、P、T，且 B+P+T=1，單位投注：

- EV(B) = 0.95 × B − P（預設莊佣金 5%）。
- EV(P) = P − B（預設閒佣金 0%）。
- 和局退回本金，淨收益為 0。

費率由 cardcount_profile.json 的 settlement 設定讀取。最終方向取兩者 EV 較高者；等值視為 Balanced。正式下注須另過原有門檻，兩邊負 EV 時不會強制下注。

舊流程先選 Physics candidate，再由最終 EV 判斷能否下注；新流程由最終 EV 直接產生 candidate，因此舊候選方向不會否決 EV 所選方向。牌路與 Core/XGB 輔助條件不得翻轉 EV 選向。

## 畫面與機率

畫面方向採同一最終 EV 口徑，機率維持模型原值；不把 EV 正規化成勝率。例：排除和局後莊 50.4%／閒 49.6%，扣莊佣金後可能是閒 EV 較高；這不代表把閒勝率改成超過 50%。EV 與命中率是不同指標。

一般方向顯示仍保留既有 2 個百分點門檻；「不觀望測試」保留並採 EV 較高方，只是顯示測試，不會繞過正式下注風控。EV 完全相同時測試畫面採莊，正式選向仍為 Balanced。

raw_next_hand_forecast 保留原始 winner-head 機率供診斷。本機逐局校準仍在背景評估，EV 選向模式下不改寫方向或正式資格，避免畫面與正式 EV 使用不同機率。

## 設定與回滾

cardcount_profile.json 新增 direction_selection：schema_version=1、mode=post_draw_ev_argmax_v1、ev_tie_tolerance=1e-12。

將 mode 改成 physics_candidate_v1，重新載入即可退回舊候選與顯示流程。完整回滾可回復本 PR 的 merge commit；保留先前補牌後 EV 一致化提交 e8454fdda9457661a77d24db421ba79f47838817。變更設定時應更新 auth-client.js 的 runtime 快取版本。

## 驗證

執行 node test_ev_led_direction.js、node test_post_draw_ev.js、node test_physics_forecast_first.js，以及既有 runtime / Macro / profile / browser calibration / button / persistence / middle-late 測試與 Python 報告測試。

新測試覆蓋：扣佣前後方向不同、負 EV 不強制下注、和局退回、EV 等值、正常 2pp 門檻、不觀望按鈕、不同前局 EMA、舊候選不再否決最終 EV，以及 Core / PF / 模型檔雜湊不變。

validation/ev_led_direction_smoke.json 是固定種子的 4 靴合成流程測試，274 筆：下注 83→115，Skip 69.71%→58.03%，最終機率差 0，EV 選向不一致 0，負預測 EV 下注 0。命中率 44.16%→48.57%，實現 EV/注 −0.11988→−0.03739，仍為虧損。這不是獨立真實 hold-out，不能據此宣稱準度、獲利或模型晉升；尚需完整 chronological 牌靴資料評估。

## 未變更範圍

Frozen Core 兩檔、213→48 拓撲、57D 定義與順序、XGB binary 目標、正式模型權重、PF 核心、開通驗證與 30 分鐘限制均不變。auth-client.js 僅更新預測 runtime 快取版本。本次只發布 BBB，不同步 BGS 或 Render。
