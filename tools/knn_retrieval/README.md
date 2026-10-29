# 相似局面檢索：階段一完整合成管線

本次補正沿用 `calib/pr72-20261029-knn-retrieval`，基線固定 PR72 head
`42106a9adf082630357b3ae7ccd1c422c1add489`。main、PR72、Physics EMA、PF、256D Context、LinUCB 與模型檔均不修改。

## 最新產出

- [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)：欄位表、事前組成的最低資料、存放位置、完整重現命令及限制。
- [data.schema.json](data.schema.json)：實際由 pipeline 驗證的 JSON Schema。
- [SYNTHETIC_minimal.json](examples/SYNTHETIC_minimal.json)：一靴完整合成範例，檔名和每行都有 SYNTHETIC。
- [REPORT.md](../../validation/knn_retrieval/SYNTHETIC_pipeline/REPORT.md)：三組回放統計、逐靴表、切分和 bootstrap。
- [report.json](../../validation/knn_retrieval/SYNTHETIC_pipeline/report.json)：同格式機器報告，每個數值均附 data_kind。
- `generate_synthetic.py`、`pipeline.py`、`replay.js`：建庫、檢索、離線融合、獨立 runtime 回放的完整可執行流程。

**合成資料僅用於管線驗證，不得用於宣稱真實增量。合成資料的結果不能用來判斷真實增量，只能用來證明管線可運行。**

## 實驗界線

固定前七成完整靴建庫、後三成內再按時間等分 fitting/test。
事前組成鍵僅剩餘總張數及十類點值比例，排除所有路單、結果序列和核心特徵。
建庫區擬合標準化歐氏距離，k 預設三十，可配置二十至五十。
真實庫限五萬至五十萬筆；本次合成機制小庫不冒稱通過規模門檻。

融合為離線擬合的非負 logistic stacking。距離／熵門檻只在 fitting 區內以前半訓練、後半選擇，
再以 fitting 區重新擬合。最終 test 不調參。不得為使檢索生效而把擬合所得零權重改成非零。
固定正則化強度為 0.001；沒有從最終測試選出該值。

standard 僅包含 PR72 原生 strong 且 stake_multiplier=1，weak 全排除；這是本實驗明確映射。
三組均套用額外的離線兩連敗永久停手限制，不解除 runtime 原有 Skip。
每靴每組獨立 VM，核心先預測再讀當局結果、結算、更新；拒絕的出手以 Skip 登記，
不能從基線輸出事後截斷冒充候選回放。標準化器、檢索鍵、原始模型機率與融合機率分開保留。
實驗基線是「PR72 strong + 相同離線兩敗限制」，不是未加限制的線上指標。

四連勝在非和 standard 結算序列計算，和局/Skip 不重置；五/六窗口按原始連續局號滾動、
包含第一局無出手位置，窗口不得跨靴。四筆連勝須全部落在窗口內。
命中率合計勝數／非和 standard 出手數；窗口與四連勝為逐靴等權平均。
兩敗熔斷須逐靴實測通過；不允許把 light 計入補樣本。

合成資料不產生真實 INCREMENT 或真實 NO_INCREMENT 判斷，永遠 expansion_allowed=false。
任一組最終 standard 少於一百次，標記 INSUFFICIENT_SAMPLE，仍保留實測數字。
真實驗證還要求至少三十靴及每組五百筆非和 standard；即使符合最低門檻仍不等於功效充分。
配對整靴 bootstrap 一萬次，報告描述性區間；兩候選組×三個共同主要終點使用 Bonferroni 同時區間。
真實足夠樣本若命中率或五／六窗口任一未顯著增加，標 NO_INCREMENT 並停止，不擴庫。

## 歷史前置審計

`preflight.py` 和 `validation/knn_retrieval/preflight.json` 保留上一提交的「無真實輸入」前置檢查。
`source_audit.json` 保留 PR72 原倉庫資料盤點。它們不是這次 SYNTHETIC 完整管線的結果；
最新狀態以 `SYNTHETIC_pipeline/report.json` 為準。

## 驗證與回滾

執行方式見 DATA_REQUIREMENTS。`pr72_blob_manifest.json` 鎖定十個基線檔案的 Git blob SHA1，
每次回放先核對位元內容，報告另保存 SHA256。只新增離線工具與報告，無 UI/部署/workflow 變更。
本次報告包含輸入 hash、來源、時間範圍、種子、切分、融合權重及逐靴風控斷言。

撤銷補正提交時，僅在實驗分支執行 `git revert <補正提交SHA>`。
若要連同上一輪審計一起撤銷，再 `git revert 3d7468122a5f4898f1339aa7c36237e8bbcce17e`。
所有結論僅限回放統計，不提供未來勝率保證。
