# 相似局面檢索資料需求規格書

## 資料位置與格式

真實資料放在倉庫根目錄下 `research/data/knn_retrieval/real/complete_hands.json`。
這是本機資料位置，不提交真實紀錄；既有 `research/.gitignore` 是否覆蓋此路徑應先核對。
合成輸入放 `/tmp/BBB_SYNTHETIC_20.json`；最小完整靴範例為
`tools/knn_retrieval/examples/SYNTHETIC_minimal.json`。
正式 JSON Schema 為同目錄 `data.schema.json`（JSON Schema Draft 2020-12）。
pipeline 使用同一輸入契約與輸出格式處理真實/合成資料，不接受混合來源。

| 欄位 | 型別 | 範例／必要性 |
|---|---|---|
| schema_version | integer | `1`；頂層必填 |
| data_kind | string | `SYNTHETIC` 或 `real_observations`；頂層及每行必填 |
| seed | integer | `20261029`；頂層必填，控制重現性，不控制真實發牌 |
| source | string | 來源系統、匯出批次或合成器說明；頂層必填 |
| chronological_order | string | `oldest_to_newest`；頂層必填 |
| rows | array of objects | 完整逐局，包含和局、Skip、未下注局 |
| shoe_id | string | `SYNTHETIC-00000`；每行必填，同靴不可交錯 |
| shoe_start | ISO-8601 string | `2026-01-01T00:00:00+00:00`；有時區，同靴一致 |
| round_index | integer | `1` 起逐局連續，含和局；每行必填 |
| outcome | string enum | `B`、`P`、`T`；事後標籤，預測前不讀取 |
| initial_counts_after_burn | integer[10] | 燒牌後可觀察的初始點值計數；同靴固定 |
| remaining_counts_before | integer[10] | 本局發牌前的剩餘計數，只能從前綴累計 |
| consumed_counts | integer[10] | 本局實際消耗各點值張數；本局結束後才可用 |
| composition_source | string enum | 真實 `observed_prefix`；合成 `synthetic_fully_observed_prefix` |
| composition_through_round | integer | 必須為 `round_index-1` |
| composition_evidence | string | 初始牌組／燒牌及逐手牌值的原始可稽核紀錄位置或索引 |
| card_count | integer | 選填，4–6；來自本局結算 |
| player_score / banker_score | integer | 選填，0–9；來自本局結算 |

每個長度為十的向量依序表示：點值 `0,1,2,3,4,5,6,7,8,9`。
點值 0 合併 10/J/Q/K，點值 1 為 A；不能宣稱保留完整牌階。

## 每靴做事前牌組成檢索的最少資料

必須具備：靴識別、開始時間、每手局號及事後 B/P/T；燒牌後初始十類點值數量；
每手公開消耗的十類點值數量；來源證據；組成可得的時間邊界。
可直接記錄每手 `remaining_counts_before`，但仍須用初始組成和已消耗紀錄核對。
`remaining_before[t+1] = remaining_before[t] - consumed[t]`，不得使用本局或未來的消耗來建本局鍵。
若燒牌未公開而無法知道初始組成，就不符合「精確已觀察組成」契約。
單純 B/P/T、雙方最終點數或總張數不夠；不能由這些欄位捏造十類牌計數。
真實來源還需人工確認桌規、牌副數、燒牌、重洗和完整性；Schema 合格不代表證據真實。

合成器保留洗牌模擬器中的十類計數，明確假設所有已消耗牌都可觀察。
這是管線測試場景；不代表正式 B/P/T 手機輸入已取得這些資訊。
它使用既有 `research.baccarat.burn/deal`，八副牌、不放回發牌與現有補牌規則；
每靴固定觀察前六十手，非自然切牌結束靴，不做任何有利結果挑選。

## 用合成資料跑通完整管線

在與 PR72 相同的倉庫根目錄執行（Python 3.12（本次驗證版本）、Node 18+）：

```bash
python -m pip install -r tools/knn_retrieval/requirements.txt
python -m unittest discover -s tools/knn_retrieval -p 'test_*.py' -v
python tools/knn_retrieval/generate_synthetic.py --shoes 20 --seed 20261029 --output /tmp/BBB_SYNTHETIC_20.json
python tools/knn_retrieval/pipeline.py --input /tmp/BBB_SYNTHETIC_20.json --output /tmp/BBB_SYNTHETIC_run --k 30 --seed 20261029 --bootstrap 10000
```

輸入檔與輸出目錄均須為新的路徑，工具拒絕覆蓋。單靴範例只示範輸入契約，
不足以做三方切分；完整管線至少需要可形成建庫／內部前推擬合／最終測試的多靴。
真實資料使用同一 pipeline 命令，換成真實輸入與新輸出路徑；真實建庫不得低於五萬筆，
不得超過五十萬筆。小型 SYNTHETIC 模式明確豁免下限，用於機制測試，不能冒稱規模驗收。

預期輸出：

- `library.json`：建庫筆數、十一維名稱、只從庫擬合的標準化參數及 k。
- `replay_input.json`：僅 validation_fit／validation_test 的完整原始輸入。
- `signals_and_fit.json`：鄰居 ID、B/P/T 分布、條件 P(B)、距離均值／方差、熵及離線擬合權重／閘門。
- `baseline_trace.json`、`ungated_trace.json`、`gated_trace.json`：三個獨立 VM 回放。
  每行包含當局事前 runtime 輸出、原始 tier、實際執行方向、熔斷狀態；候選另含相對原始候選方向的鄰居 W/L 計數。
- `report.json`：三組六項指標、逐靴表、切分表、配對 bootstrap、來源、seed 和 runtime SHA256。
  每個數值均以 `{ "value": 0.5, "data_kind": "SYNTHETIC" }` 明確標註；真實資料用相同包裝。
- `REPORT.md`：可直接閱讀的同內容報告，每個數字旁都有 SYNTHETIC。

所有原始機器輸出頂層及逐筆紀錄均帶 data_kind，保留可直接計算的數值；
對外統計報告則逐值標籤。鄰居 W/L 指標排除和局，另列 tie_count，避免把和局當輸。

## 風控、分類與結果解讀

PR72 的原生 tier 為 strong/weak/skip。本離線實驗 standard 明確定義為：
`entry_tier == strong AND stake_multiplier == 1 AND direction IN (B,P)`。
weak 半注視為 light，全部排除，不升級為 standard；這是明示的實驗映射，非冒稱原碼已有 standard 名稱。
該 runtime 沒有可直接呼叫的兩連敗停手狀態，因此三組套用同一額外離線硬限制：
連續兩筆 standard 失敗後該靴永久停手；和局與 Skip 不重置連敗；新靴才重置。
保留既有 runtime Skip／強弱分類；檢索只可再拒絕，不能翻方向、恢復弱注或解除熔斷。
這些結果屬「PR72 強注 + 相同離線兩敗限制」的回放統計，不能冒稱未加限制的線上基線。

融合為非負 logistic stacking，權重由 validation_fit 離線最小化二元 log loss，含固定 L2 正則化。
閘門只在該區域內按靴前後分半：前半擬合各候選、後半以 log loss 選擇，再以整個 fitting 區重擬合選定模型。
距離與熵候選為前半 fitting 查詢的中位數、75% 與最大值；不在最終測試靴調參。
不改 raw P(B)；另存 fused_p_b，僅在融合方向與原始 standard 方向衝突時拒絕原出手。
有閘門組不通過距離／熵條件就回退基線；無閘門組所有機會都可用檢索投票。

任一組最終測試 standard 出手少於一百次，標 INSUFFICIENT_SAMPLE，但照實輸出所有可算數字。
零出手靴的窗口覆蓋、四連勝比例為零；命中率無分母時以零作格式慣例，
同時 `hit_rate_estimable=false`，不能解讀為已估出零命中率。
整靴 bootstrap 對各候選與基線同步重抽，給描述性 95% CI 及六項主要比較的 Bonferroni 同時區間；
樣本太小時這些區間僅供診斷，不作有效性認定。合成模式永遠不判定真實 NO_INCREMENT／真實增量，且禁止擴庫。

**合成資料僅用於管線驗證，不得用於宣稱真實增量。合成資料的結果不能用來判斷真實增量，只能用來證明管線可運行。**
