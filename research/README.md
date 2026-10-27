# BBB 百家樂「資訊上限、扣佣 EV 與實務命中率」獨立研究

> 研究流程說明：這裡的模擬與模型結果**不是已完成 20 萬靴的數值報告**，所有新訓練候選預設為 review-only。正式瀏覽器點數 PF 的設計與開關見 [../POINT_CONDITIONED_PF.md](../POINT_CONDITIONED_PF.md)。

## 環境與重現

建議 Python 3.11/3.12，在倉庫根目錄執行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r research/requirements.txt
python -m unittest discover -s research/tests -v

# 效能/統計煙霧測試：僅驗證流程，絕不可解讀為理論 edge
python -m research.run generate --shoes 40 --shard-size 20 --out research/data
python -m research.pipeline --data research/data --reports research/reports

# 正式預註冊研究：全新空資料夾，至少 200,000 靴完整記錄
python -m research.run generate --shoes 200000 --shard-size 250 --seed 20261027 --cut 14 --out research/data_200k
python -m research.pipeline --data research/data_200k --real-csv research/real_shoes.csv --reports research/reports_200k
```

### 由本機點數紀錄訓練離線候選

瀏覽器的「匯出開牌資料」會下載各本機牌靴逐局 B/P/T、閒莊最終點數與選填總張數。將該檔案移至離線研究環境後，可建立一份整靴 holdout 的輔助模型候選：

```bash
python -m research.train_point_auxiliary \
  --input bgs_observed_hands_export.json \
  --output research/candidates/point_auxiliary_candidate.json
```

需要安裝 `research/requirements.txt` 中的 scikit-learn。輸出包含特徵 ABI、整靴 80/20 holdout、準確率、log loss 與正 EV 篩選命中率；它只是一份 review-only 候選，不會替換正式模型。每次訓練請使用新輸出檔，避免覆蓋既有實驗。使用者歷史資料應依組織的隱私與存取規範，在本機離線環境處理。

**預設 20 萬靴**，不會默默減為小樣本。每一局均以 10 個點數類別**不放回排列枚舉**當下真實的 `P(B/P/T | composition)`，不是對 GT 使用蒙地卡羅。純 Python 在此規模可能極慢；安裝 numba 後首次呼叫需要 JIT 編譯，大規模完整枚舉仍屬高運算成本。依機器預算分批執行並保留分片，不能因測試時間不夠就聲稱已完成。

分批處理時：`--start-shoe` 與 `--shoes` 指定互不重複範圍，所有分片必須放同一目錄、共用同 seed/cut 參數；不可覆蓋已存在的檔案。建議記錄 Python、NumPy、Numba、sklearn、Torch 版本、CPU 與 Git commit；研究輸出包含 seed 與切分規則。若途中中斷，只在新輸出範圍重新執行。

## 精確發牌與環境假設

- 8 副牌，共 416 張：點數 0 有 128 張，1–9 各 32 張；點數 0 對應 10/J/Q/K；A=1。
- 發牌順序 P1、B1、P2、B2，8/9 天牌直接停止；其他依完整閒/莊第三張規則表；回傳 4/5/6 張機率、閒莊最終點數分布、期望點差等。
- 燒牌：預設揭面首張後，再燒等於牌面值數量，0 點牌視為 10；不同館別可調整 `burn(..., additional=False)` 對照。切牌自靴底往上 14 張，當完整一手跨過 cut 標記才結束該靴；此係明確研究假設，應用真實每靴局數驗證。
- 階段二額外存放事前剩餘牌組的十種點數類別比例與下一張牌的**期望百家樂點值**；這不等於確切 A–K 牌階期望：僅有點值 0 的合併計數無法辨別 10/J/Q/K 的實際剩餘牌階，絕不假冒真實 rank-level 標籤。
- `exact_distribution(tuple(counts))` 是**精確組成枚舉**，且由 `@lru_cache` 快取完全相同的組成；`simulate_shoe()` 依可重現 RNG 實際洗牌／發牌，其結果分布自然符合機率，不能拿模擬實現結果冒充 GT。
- 實測公認基準：8 副牌莊約 45.86%、閒約 44.62%、和約 9.52%。`unittest` 會檢查精確初始組成的概率接近此值。固定莊莊家抽 5% 佣金、閒無佣金、和局退注。

## ABI／預知未來洩漏檢查

已逐行對齊 `final_probability_runtime.js` 的 `historyVector(seq)`：最近 64 手右對齊 one-hot (B,P,T) 192 維，加上 21 維長度、連續莊閒、轉折、8/16/32/全靴比例、entropy 與前次非和局方向，恰好 213 維。

`history` 僅包含 **t-1 及以前**已公布 B/P/T；`ground_truth` 與輔助標籤來自 **第 t 局發牌前**尚未公開的牌組快照，**僅用作離線標籤／oracle**，不可輸入預測器。真實資料必須依整靴、按時間切分（先前 85% 靴，最後 15% 靴測試）。模擬資料使用 `SHA256(seed:shoe_id)` 固定分配 train 70%、valid 15%、test 15%；同靴永不跨組。

測試資料不得做 scaling 擬合、超參數搜尋、閾值選擇、模型選擇或任何事後篩選。閾值只從 valid 選出，再在 test 單次評估。配對 bootstrap／EV/CI 抽樣單位均是**整靴**。PF 回放只採用當前觀測之前的 B/P/T；不�