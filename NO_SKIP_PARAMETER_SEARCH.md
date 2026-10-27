# 零觀望參數搜尋與輕量模型：尚未達到 52–53%
> 後續更新：使用者已明確接受未驗證風險，此候選僅接入「不觀望實驗」顯示，未更改正式下注。原統計未過的結論保留；部署與回滾見 [EXPERIMENTAL_NO_SKIP_RELEASE.md](EXPERIMENTAL_NO_SKIP_RELEASE.md)。

本次在目前工作環境實際執行程式回測，並非由 GPT 口頭推估命中率。**結果沒有證明平均維持 52–53%，因此不替換正式預測。** 只新增離線搜尋、校正候選、測試與報告；Physics、PF500、Core、XGB 權重、正式閘門、網頁、開通碼與時間限制不動。

## 一、固定搜尋空間，整靴分割

第一批使用全新固定 seed `2026102802`，共 **300 靴、20,973 局**：

- 前 180 靴／12,584 局：估計訓練集先驗。
- 接續 60 靴／4,193 局：選參。
- 最後 60 靴／4,196 局：候選鎖定後才做測試，不參與選參。

共比較 **24 組參數**：Direct Physics（僅消融對照）、Physics+PF、馬可夫 4%／8%／12%，三種單層因果 EMA，以及 0%／4% 訓練集先驗。馬可夫與額外先驗共用不超過 12% 的預算；額外先驗只同向使用。Direct Physics 單獨路徑不得作為部署候選，保留 Physics+PF 核心。

EMA 最新權重：不平滑 `[1,1,1]`、responsive `[0.48,0.58,0.65]`、fast `[0.65,0.75,0.84]`，分段 ≤20／21–45／≥46。這是在原始預測上比較一層 EMA，不是對已平滑的 finalPB 再做 EMA。這些只是離線候選，沒有改正式平滑設定。

每靴重置；只以當手之前的 B/P/T 產生特徵，結果公布後才更新既有本機校準。保持既有校準的 noise／樣本／改善閘門，不用模擬器的當局點數、牌值或張數作輸入。快取只節省重複 PF 運算，不把未來結果送回預測器。所有手都計分，包含第一手；和局計退回。

選參以選參段的中獎率、EV/bet、Brier 排序，最後偏好較簡單設定。最後測試段不影響排序。生產模型及資料雜湊固定，四個計算 worker 載入相同版本；重建現行策略的 **20,973 次逐局方向全部一致**。

### 第一次未接觸測試段的結果

| 路徑 | 排除和局中獎率 | EV/每次出手 |
|---|---:|---:|
| Direct Physics（消融） | 49.47% | −0.02358 |
| Physics+PF | 49.87% | −0.01770 |
| Physics+PF+馬可夫8% | 49.87% | −0.01770 |
| 選出的參數：馬可夫8%、先驗4%、不額外平滑 | **49.87%** | **−0.01770** |
| 只選訓練集較常見方向（控制組，不部署） | 50.79% | −0.00860 |

此測試段選出候選為 1,887 勝、1,897 負、412 和，100% 給 B/P。整靴 bootstrap 95% 中獎率區間約 48.31–51.40%。

**發現：同向的馬可夫混合能改機率大小，卻可能完全不改 B/P 方向，因此不代表中獎率會提高。正式 EV 門檻也不會直接提高「完全不觀望」畫面每局選向的命中率。** 本次同向先驗稍改 Brier，但沒有增加測試段正確方向數。

## 二、另外訓練兩個輕量校正模型

不是重訓 Physics/PF，也不是恢復 XGB 影響。使用原始 Physics+PF+馬可夫機率，額外訓練兩參數的單調 Platt 校正：

`q = sigmoid(slope * logit(p) + intercept)`

- 僅前 180 靴的 B/P 結果參與擬合；和局不當作 B/P 標籤。
- 斜率固定限制 0.5–1.5、截距限制 ±0.08、L2 身分映射正則 0.01。
- 比較兩個最大修正幅度：±0.005 與 ±0.010。
- 僅使用接續 60 靴選擇；不使用前一輪最後 60 靴做此模型的擬合或選擇。
- 使用相同既有本機校準規則，不改 57D 定義、模型權重或 PF 核心抽樣。
- 回測當時校正模型僅保存在 `candidates/no_skip_calibrator_v1.json`，未載入網站；後續使用者接受的實驗顯示版本見上述部署文件。

選出斜率約 0.922443、截距約 0.006789、最大修正 0.01。它在原選參段中獎率 49.82%，比現行對照 50.73% 更差，因此穩定性閘門已不通過。仍另外做獨立研究測試，不能用一次稍好的結果救回先前失敗。

候選參數先鎖定，再以另一個固定 seed `2026102803` 產生 **100 靴、6,989 局**；這批僅快取，不進行任何選參，之後一次驗證已鎖定模型。

### 第二批全新測試（相同 100 靴比較）

| 指標 | 現行零觀望比較組 | 輕量校正候選 |
|---|---:|---:|
| 勝／負／和 | 3,248／3,071／670 | 3,264／3,055／670 |
| 排除和局中獎率 | **51.40%** | **51.65%** |
| EV/出手 | +0.00951 | +0.01205 |
| B／P 次數 | 4,680／2,309 | 5,284／1,705 |
| 最大連敗 | 12 | 11 |

候選的整靴 bootstrap 95% 中獎率區間約 **50.45–52.86%**；EV 區間約 **−0.00944 至 +0.03352**。同鞋配對 bootstrap 的中獎率改變區間約 **−0.51 至 +0.92 個百分點**，不能確認有穩定改善。

**51.65% 不可四捨五入宣稱已驗證達到 52%，更不等於未來能平均維持 52–53%。** 初次 60 靴與新的 100 靴來自不同資料，不可拿 49.87% 與 51.65% 相減宣傳改善；正確同資料比較是 51.40% →51.65%。兩批都只是模擬，尚無真實 hold-out。

## 三、為何不部署

兩輪均沒有達到驗證目標；輕量模型還有選參段退步。缺少真實獨立牌靴，絕對命中率及配對改善的區間也未過門檻。所有報告 `production_promotion_allowed=false`。不會反覆換 seed、調完參再偷看同一測試段，直到挑到一批剛好超過 52%。

「中獎率」統一使用 勝／(勝＋負)，和局退回另列；另報 wins/all hands、覆蓋率、佣金後 EV 與 Brier。莊贏 +0.95、閒贏 +1、輸 −1、和 0。零觀望候選在完全均衡時選 B 作為已知先驗的研究比較，不捏造高於 50% 的信心，也不刻意莊閒交替。這個候選沒有取代正式畫面。

## 四、重現命令

在倉庫根目錄執行，每次使用新的輸出目錄。Python 3.10–3.12；Node.js 22 以上。四個 worker 只平行處理獨立牌靴，不更動 chronological split。

```powershell
node tools/no_skip_parameter_search.js --synthetic 300 --seed 2026102802 --workers 4 --output-dir ../no-skip-search-v1
python -m venv .venv-no-skip-calib
& ./.venv-no-skip-calib/Scripts/python.exe -m pip install -r requirements-no-skip-calibration.txt
& ./.venv-no-skip-calib/Scripts/python.exe tools/fit_no_skip_calibrator.py --cache ../no-skip-search-v1/cache.json --training-report ../no-skip-search-v1/report.json --output ../no-skip-search-v1/fitted.json
node tools/no_skip_calibration_candidate.js choose --cache ../no-skip-search-v1/cache.json --fitted ../no-skip-search-v1/fitted.json --output ../no-skip-search-v1/frozen.json
node tools/no_skip_parameter_search.js --synthetic 100 --seed 2026102803 --workers 4 --cache-only --output-dir ../no-skip-fresh-holdout-v1
node tools/no_skip_calibration_candidate.js evaluate --cache ../no-skip-fresh-holdout-v1/cache.json --candidate ../no-skip-search-v1/frozen.json --output ../no-skip-fresh-holdout-v1/report.json
```

研究結果未過門檻時退出碼 2 是正常拒絕晉升，不是用 0 冒充成功。資料錯誤退出碼 1。`--input real.json` 可取代 `--synthetic`；完整真實 B/P/T、來源與 chronological 格式見 `NO_SKIP_VALIDATION.md`。未知牌值不必補齊，也不可用模型猜測填成實際觀察。

本次快取及完整逐局資料留在本機輸出目錄；倉庫只存摘要、候選參數與測試。真實資料不會自動上傳 GitHub。

## 五、驗證與回滾

```sh
node test_no_skip_validation.js
node test_no_skip_parameter_search.js
node test_no_skip_calibration_candidate.js
python -m unittest test_fit_no_skip_calibrator
node test_hitrate_max.js
node test_pr56_restore.js
```

測試涵蓋未來標籤不影響選參／擬合、同靴因果重建、先驗預算 ≤12%、有界單調模型、禁止使用舊輸入或同 seed 當新測試、整靴配對區間，以及禁止自動部署。快取 callback 僅提供預測前特徵，不提供當局結果或點數。

本次沒有修改正式預測或網頁。回滾研究工具提交即可，正式權重無需回復。後續若要驗證實際中獎率，需要完整真實 B/P/T 牌靴，不需要每張牌。即使某份真實測試過關，也不能保證未來勝率。
