# 算牌優先修正（2026-10-22）

本文件及 `COUNTING_PRIORITY_REVIEW.md` 取代上一版 `NO_PF_UPGRADE.md` 的規律權重、選模與 Skip 驗收設定。原始 main 基準為 `0bbd1f7db966be5c67f00809fa4db1822386ac21`，上一版修改為 `7f44d37`。

## PF 已移除

B/P/T 歷史 → Direct Physics 213→48D → Physical EV／candidate → Dynamic Clip＋Macro EMA（依既有配置）→ Frozen Core＋57D XGBoost → Final EV Guard → Volume Guard → B/P/Skip。

實際載入器 `auth-client.js` 不再載入 PF；`index.html` 不直接載入 PF。`particle_filter_runtime.js` 只留 deprecated 空檔。主 runtime 不讀取 PF 物件、不呼叫 PF，也不要求 500 粒子診斷。Physics 或相容的 57D 模型未就緒會 Skip；其餘 EV、noise、confidence、Volume Guard 與錯誤保護仍保留。

48D slot 42 的 `particle_uncertainty` 只保留作為既有模型欄名，其值來自 Direct Physics uncertainty。改名或刪欄會破壞已訓練權重與 57D ABI，因此不改。歷史 PF 報告／測試及 shadow 工具不是正式載入路徑。比較原始 main 時，回放工具只會在基準目錄載入其原始 PF。

## 原始架構保護

- `app256forward.js`、`app256continuation.js` 完全不修改。
- Physics 的 213 輸入、64／32 隱藏層與 48 正式輸出不修改；既有訓練輔助 heads 不增不減。
- 57D 特徵名稱、順序、定義與直接二元目標 P=0／B=1 不修改。
- 現有 Physics JSON／joblib 與 57D JSON 權重不修改；本次新增的是可驗證的訓練配置，不宣稱已重訓或已提高命中率。

## 回滾

只撤銷本次算牌優先修正：revert 本次修正提交，會回到 `7f44d37`，PF 仍然移除。要完整回到最初 main，先在獨立分支檢視 `git switch --detach 0bbd1f7db966be5c67f00809fa4db1822386ac21` 的整套檔案，再按原部署程序發布該版本；不混用舊 loader、新 runtime 與舊模型。不要只恢復 PF script，因為新 runtime 已不讀取它。

回滾不是 reset 遠端 main；任何部署前都應先驗證完整組合。現行模型權重沒有更換，因此本次不需要模型轉檔或特徵遷移。
