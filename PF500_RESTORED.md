# PF500 回復｜每局獨立抽算｜保留觀望切換

目前路徑：B/P/T 歷史 → Direct Physics（213→48）→ PF500 → Physical EV / candidate → Clip / Macro EMA → Core / 57D XGB 輔助 → Final EV Guard → Volume Guard。

每次分析建立全新的 500 個八副牌粒子，從該次完整歷史重新回放；不沿用前次粒子、權重或重採樣狀態。同一歷史使用固定種子，結果可重現；「每局獨立」不表示忽略已輸入的歷史，也不表示反覆按同一局會抽出不同方向。

恢復原有 quality 混合（6%～16%），保留 PF 不可單獨翻轉 Direct Physics 方向的限制。57D XGB 仍吃原始 Direct Physics 特徵，維持訓練分布。畫面與不觀望測試均顯示校正後 Physics＋PF500 機率。PF 診斷會記錄粒子數、完整歷史指紋與 rebuild_from_scratch；故障時沿用現有 Direct Physics 備援並記錄錯誤。

保留「全程不觀望測試／停止不觀望測試」按鈕，自動跟隨每次報結果；「開始分析」切回原本有觀望的正式分析。測試方向不新增正式下注紀錄。

未修改 Frozen Core 數學、Physics 拓撲、57D 特徵定義、模型權重、Macro EMA 參數、EV／Volume Guard 門檻或開通碼驗證。PF 原始演算法未改動。

PF 依 B/P/T 對隱藏牌值做條件模擬，不代表知道真實已出牌；沒有新增命中率或獲利保證。

驗證：四組 JavaScript 回歸、不同歷史交錯後同一歷史重算一致、500 粒子重建、UI 模式切換與正式紀錄隔離。回滾本次提交即可回到不使用 PF 的前一版。
