# BBB 百家樂「資訊上限、扣佣 EV 與實務命中率」獨立研究

> 2026-10-27：僅新增 `research/`；**完全不修改**正式 JavaScript/JSON、權重、閾值、授權、會員、UI 或 PF500。這是可執行研究程式，**不是已完成 20 萬靴的數值報告**。所有出手建議預設為 shadow-only。

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

**預設 20 萬靴**，不會默默減為小樣本。每一局均以 10 個點數類別**不放回排列枚舉**當下真實的 `P(B/P/T | composition)`，不是對 GT 使用蒙地卡羅。純 Python 在此規模可能極慢；安裝 numba 後首次呼叫需要 JIT 編譯，大規模完整枚舉仍屬高運算成本。依機器預算分批執行並保留分片，不能因測試時間不夠就聲稱已完成。

分批處理時：`--start-shoe` 與 `--shoes` 指定互不重複範圍，所有分片必須放同一目錄、共用同 seed/cut 參數；不可覆蓋已存在的檔案。建議記錄 Python、NumPy、Numba、sklearn、Torch 版本、CPU 與 Git commit；研究輸出包含 seed 與切分規則。若途中中斷，只在新輸出範圍重新執行。

## 精確發牌與環境假設

- 8 副牌，共 416 張：點數 0 有 128 張，1–9 各 32 張；點數 0 對應 10/J/Q/K；A=1。
- 發牌順序 P1、B1、P2、B2，8/9 天牌直接停止；其他依完整閒/莊第三張規則表；回傳 4/5/6 張機率、閒莊最終點數分布、期望點差等。
- 燒牌：預設揭面首張後，再燒等於牌面值數量，0 點牌視為 10；不同館別可調整 `burn(..., additional=False)` 對照。切牌自靴底往上 14 張，當完整一手跨過 cut 標記才結束該靴；此係明確研究假設，應用真實每靴局數驗證。
- `exact_distribution(tuple(counts))` 是**精確組成枚舉**，且由 `@lru_cache` 快取完全相同的組成；`simulate_shoe()` 依可重現 RNG 實際洗牌／發牌，其結果分布自然符合機率，不能拿模擬實現結果冒充 GT。
- 實測公認基準：8 副牌莊約 45.86%、閒約 44.62%、和約 9.52%。`unittest` 會檢查精確初始組成的概率接近此值。固定莊莊家抽 5% 佣金、閒無佣金、和局退注。

## ABI／預知未來洩漏檢查

已逐行對齊 `final_probability_runtime.js` 的 `historyVector(seq)`：最近 64 手右對齊 one-hot (B,P,T) 192 維，加上 21 維長度、連續莊閒、轉折、8/16/32/全靴比例、entropy 與前次非和局方向，恰好 213 維。

`history` 僅包含 **t-1 及以前**已公布 B/P/T；`ground_truth` 與輔助標籤來自 **第 t 局發牌前**尚未公開的牌組快照，**僅用作離線標籤／oracle**，不可輸入預測器。真實資料必須依整靴、按時間切分（先前 85% 靴，最後 15% 靴測試）。模擬資料使用 `SHA256(seed:shoe_id)` 固定分配 train 70%、valid 15%、test 15%；同靴永不跨組。

測試資料不得做 scaling 擬合、超參數搜尋、閾值選擇、模型選擇或任何事後篩選。閾值只從 valid 選出，再在 test 單次評估。配對 bootstrap／EV/CI 抽樣單位均是**整靴**。PF 回放只採用當前觀測之前的 B/P/T；不能使用已抽出的真實隱藏牌。第一層因果平滑只能沿正向時間累積；校準與選擇性出手的參數若有擬合，只能在 train/valid。

## 階段與閘門

| 階段 | 程式 | 必要輸出與限制 |
|---|---|---|
| 1 精確模擬 + 預測可達基準 | `baccarat.py`, `features.py`, `run.py` | `stage1.json`, `stage1.md`, `ev_curve.png`；先驗／結果模型／完全知道剩餘組成 oracle，log loss、Brier、每注與每機會扣佣 EV、整靴 95% CI、頻率與房屋優勢 |
| 閘門 1 | `run.py` | 結果模型的測試靴 EV 95% CI **下界>0** 才允許 2/3；否則 2/3 `SKIPPED_GATE_1`，但 4/5 照跑 |
| 2 軟標籤 | `stage2.py`, `reference.py` | 213→64→32→48，固定種子；軟 vs 硬模型與只讀正式 Physics direct 比較；校準圖 `calibration.png` 和配對靴 bootstrap。需 Gate 1 通過 |
| 3 組成 PF | `pf.py` | 精確結果似然、觀測結果條件下拒絕採樣隱藏牌、ESS < N/2 才重抽樣，最後一局條件更新 rejuvenation。500/2000/10000 試驗，含 10% PF 混合 vs direct Physics 關閉 PF 消融；`ess_curve.png`。需 Gate 1 通過 |
| 4 真實資料 | `real_audit.py` | **無論 Gate 1 均執行**；卡方、離散 run lengths 探索性 KS、runs、靴內 lag-1 相關、前中後與每靴局數、時間順序 holdout、基準線與現有下注日誌（若提供） |
| 5 命中率與選擇性方案 | `hitrate.py`, `upgrade_spec.md` | **無論 Gate 1 均執行**；依已有事前下注日誌衡量弱 EV、方向衝突、noise/ESS、分段勝率；A/B/C 提案、研究開關與回滾 |

**注意**：固定架構 MLP 在 holdout 的成果是「僅結果可達的**模型近似實證基準**」，**絕不是數學上已證明的資訊理論上界**。即使其 CI 下界 ≤0，也只能下結論「目前沒有統計上可證明的 edge」；不能據此嚴格證明任何可能的結果紀錄策略都無法超越佣金。反過來，單次 CI 下界>0 仍需要獨立重複研究排除選擇偏誤。

## 真實資料格式（CSV，一列一局）

最低必要欄位：`shoe_id,round,outcome,shoe_start`。 `shoe_start` 為每靴固定 ISO-8601 起始時間，`round` 從 1 連續遞增；`outcome` 僅 B/P/T。不完整或無時間戳靴不能假裝連續發牌。

為評估**實際已下注命中率**，加入事前記錄的 `direction`（B/P/SKIP）、`first_layer_direction`、`final_direction`、`final_ev`、`physics_noise`、`pf_ess_ratio`。真實 `outcome` 只在結束後回填，另建議記錄預測與開獎時戳。正式模型完整對比需要官方事前下注日誌；僅拿 Direct Physics 權重回放不是正式完整系統。

## 目前可以／不可以回答什麼

- **(a)** 只靠 B/P/T 能否超越佣金：需 Gate 1 holdout 結果，目前不聲稱有 edge；模型實驗不是嚴格理論上界。
- **(b)** 現行最終系統距天花板：需完整事前最終出手日誌與相同靴對照；目前不可估。
- **(c)** PF 與軟標籤改進：Gate 1 失敗必須停，不准虛構改善。
- **(d)** 真實消耗效應：必須有真實逐靴資料；若各檢定未顯著，可寫「未觀察到牌組消耗效應」，但不是證明不存在。
- **(e)** 模組保留／簡化：若無 edge，建議保留 EV 正確結算與必要風控／診斷，停用沒有增益的複雜影子模組；**此專案不會替你改正式程式**。
- **(f)** 命中率：適度縮小出手集合值得檢查，但必須接受出手次數下降；未證實可提升正期望。見 `upgrade_spec.md`。

## 程式邊界與未完成事項

除 `research/` 以外的正式程式未被修改。研究產出的 `data/`、`reports/` 均被 gitignore，避免上傳真實玩家紀錄。

目前未提供真實逐靴 CSV，也未執行 20 萬靴完整精確運算；因此第一、四階段數字與後續 Gate 結果 **待算**，不能交付已驗證的信賴區間。研究用 PF 含非常高成本的精確枚舉，1 萬粒子的 pilot 不代表全量靴已評估。真實環境若為連續洗牌機／不同牌副數／提早重洗，模擬數據不一定適用。
