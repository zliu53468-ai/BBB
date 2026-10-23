# 完整 middle_late_focus 重訓｜執行狀態與正式路徑驗證

日期2026-10-23；查核main `39d7a6102ace1da8f5cbd7d99f32a5e565ca99ee`。本次未替換正式權重，未變更正式平滑、決策階段或開通碼。腳本只產出隔離候選，沒有自動部署。

## 1. 正式模型確認

| 正式檔案 | SHA256 | middle_late_focus替換狀態 |
|---|---|---|
| physics_multitask_model.json | 3ec0d60251cb5b3fcb40bd67fec1334cb64d37becebbe0b040efbae871071138 | 尚未替換 |
| physics_multitask_model.joblib | 7eb7fc9b9fdfee00455385278805f09f2a405a0383c5cf76c0eb8516055ef554 | 尚未替換 |
| final_probability_model.json | 862347a6937b30ac9a8a3ece8f0eb9927569c97dee799e53cbb418e6c10be2f9 | 尚未替換 |

Physics瀏覽器拓撲213→64→32→48、ReLU、48D順序保持；正式Physics metadata仍是2500靴訓練，非本次6000靴middle_late_focus。57D順序、Core兩檔與直接P=0/B=1目標保持。上一筆PR55只加入文件和驗證工具，沒有新模型。

現況仍為cardcount_middle_late_v2：決策α平台.45/.60/.72、Macro short .28/.33/.35且係數0、PF抽樣3/4/5；本機校準只影響畫面。正式finalPB來自Physics平滑，Core/XGB為advisory。只訓練XGB不保證正式bet指標改變。

## 2. 實際執行結果

已執行：

```powershell
python tools/top_parameters/run_middle_late.py --work-dir ../../outputs/middle-late-retrain-20261023 --execute
```

實際狀態：`blocked_before_training`，`training_executed=false`，`promotion_allowed=false`。

阻擋原因：

1. **需要使用者提供完整 chronological 牌靴資料**；目前倉庫和本次工作輸出中只有模型、基準設定與少量合成機制資料，沒有使用者真實完整訓練/獨立holdout。
2. NumPy1.24.3低於requirements的1.26；scikit-learn1.3.0低於1.5；XGBoost未安裝。joblib1.3.2符合最低要求。

因此本次沒有啟動6000靴Physics fit，也沒有XGB fit或新的品質指標。這是「資料/環境前置條件未通過」，不能稱作候選已通過或模型品質測試失敗。`execution_report.json`保存版本、原正式檔hash、缺項與null holdout_metrics，不編造分段命中率。

重訓有機會改善估計，但不是精準度或正EV保證。只有B/P/T歷史時，不知道實際已出rank/點數牌值；模擬資料或更多平滑不能變成已觀測的真實剩餘牌資訊。

## 3. 使用者需提供的資料

建議交付單一JSON，完整逐局觀測、由舊靴到新靴排列，不交錯；每靴從round_index=1開始，包含和局及未出手局，不只匯出prediction log。真實來源及時間順序需能追溯；以下metadata是格式聲明，不能替代來源審核。

```json
{
  "data_kind": "real_observations",
  "chronological_order": "oldest_to_newest",
  "rows": [
    {"shoe_id":"shoe-0001", "round_index":1, "outcome":"B"},
    {"shoe_id":"shoe-0001", "round_index":2, "outcome":"T"},
    {"shoe_id":"shoe-0001", "round_index":3, "outcome":"P"}
  ]
}
```

可選實際card_count（4–6）、player_score/banker_score（0–9），有填時必須與結果一致；不要推測未提供的牌值。完整靴若有提早中止應保留原始長度與原因，不以補零或假局填到70。附實際靴開始/結束時間、來源、匯出時間，作外部順序審核；程式保留輸入順序，不按shoe_id字典重新排序。

腳本在最外層保留最新20%完整靴作**production獨立比較**；開發部分至少1000個有二元標籤的靴、12000個有效二元列；最外層holdout至少300靴。因此起始資料至少約1500靴，實際還受有效列與各階段wager數限制。可用`--min-holdout-shoes 1000`提高下限。沒有足夠晚段資料/出手時不晉升，不能只增加早段資料補晚段證據。

這個最外層holdout不會進XGB CLI input。CLI再於舊80%開發靴內依既有train/tune/calibration/EV-tune/internal-holdout切分。全部按完整靴保留時間順序，不使用row隨機切分。

## 4. 環境與完整執行命令

於倉庫根目錄建立獨立環境，不改系統Python或既有正式joblib執行環境：

```powershell
python -m venv ../BBB-middle-late-venv
../BBB-middle-late-venv/Scripts/python.exe -m pip install -r requirements-xgb.txt
../BBB-middle-late-venv/Scripts/python.exe tools/top_parameters/run_middle_late.py --hands C:/BBB-data/complete_chronological_hands.json --work-dir ../BBB-middle-late-run-01
../BBB-middle-late-venv/Scripts/python.exe tools/top_parameters/run_middle_late.py --hands C:/BBB-data/complete_chronological_hands.json --work-dir ../BBB-middle-late-run-01 --execute
```

前一行只做preflight；下一行才fit。兩次可用同一work-dir，但已存在baseline/candidate時會拒絕重跑，避免覆寫實驗；新一輪用新目錄。資料路徑必須換成實際提供的檔案。Node.js須在PATH。Linux環境的Python改用`../BBB-middle-late-venv/bin/python`。

Physics實際命令使用薄包裝呼叫**既有PhysicsFeatureExtractor及train_from_simulation**，相當於：

```powershell
python tools/top_parameters/train_physics_chronological.py train --training-profile middle_late_focus --shoes 6000 --validation-fraction 0.20 --calibration-fraction 0.10 --augment-ratio 0.25 --bootstrap-samples 2000 --random-state 20261023 --output physics_middle_late_candidate.joblib --browser-output physics_middle_late_candidate.json
```

唯一fit參數差異為`early_stopping=false, shuffle=false`，max_iter仍160，避免既有MLP內部的隨機row validation及epoch洗牌。外部整靴train/calibration/holdout、原拓撲、loss、affine/temperature/isotonic及augmentation不變。train內10%/20%有放回抽樣依需求保留；它不是驗證集隨機切分。固定迭代可能耗時較長/較易過擬合，仍需外部holdout擋下，不據此放寬門檻。

XGB執行前，`build_core_rows.js`用凍結兩個Core及既有buildOriginal7取得**當局之前**的Core P(B)與原6D；和局保留在歷史，僅排除其二元標籤。不接受使用者預填core_p_b/physics_48d/features_57d覆寫。讓既有Python trainer以新candidate.joblib重新抽取Direct Physics，避免舊snapshot繞過新權重。

```powershell
python xgb_final_probability.py train --training-profile middle_late_focus --input development_core_rows.json --physics-model physics_middle_late_candidate.joblib --min-samples 12000 --min-shoes 1000 --validation-fraction 0.20 --calibration-fraction 0.15 --tuning-fraction 0.15 --ev-tuning-fraction 0.30 --bootstrap-samples 2000 --random-state 20261023 --report-output middle_late_candidate_report.json --joblib-output final_probability_middle_late_candidate.joblib --output final_probability_middle_late_candidate.json
```

上述兩條由runner在隔離candidate目錄執行，實際argv和returncode寫進execution_report，各階段stdout/stderr寫入獨立log。禁止加`--ci-smoke`繞過品質失敗；CLI若失敗就停止，不生成可晉升結果。

使用production原profile：focus_training_weight平台1.10/1.25/1.35，Physics僅train額外中段10%/後段20%；不套用先前S/T實驗檔。所有原程式/設定逐檔hash確認，來源變動需重新審核。依賴版本記錄；完整重現另需保存pip freeze、Python/Node及系統/BLAS環境。

## 5. 正式路徑驗證與晉升門檻

模型生成後先檢查213→64→32→48、ReLU、48D順序、57D逐項相同、sigmoid classifier；保留原candidate JSON/joblib。僅在隔離candidate瀏覽器bundle中固定production的EV/decision_policy/smoothing、PF開關及runtime演算法版本，以便這一輪比較權重效果；保留候選的機率與noise校準。**特別避免trainer匯出的particle_filter_enabled=false悄悄停用正式PF。** 固定後的實際瀏覽器bundle才是外層回放候選，不拿原CLI內部XGB分數冒充正式Physics下注結果。

`replay.js`對baseline/candidate同一份獨立資料先預測、後結算，完整時間線更新，最後才切片，不分段重置Guard。第一局依現有UI無歷史不預測，1–20報告實際含第2–20局。包含和局的局數定義一致。

| 門檻 | runner/gate要求 | 本次狀態 |
|---|---|---|
| Hit-rate | ≥production，和局不列hit分母 | 未評估，缺資料 |
| EV/placed bet | ≥production；固定1單位、莊+.95/閒+1/輸-1/和0 | 未評估 |
| Absolute Correct Bets | ≥95% baseline | 未評估 |
| Brier | 整體退化≤.0015；非和條件P(B) | 未評估 |
| Skip | 增加≤5pp；`--strict`另要求至少+2pp | 未評估 |
| 前1–20、≤40 | hit/EV不回歸，Brier額外≤+.0005 | 未評估 |
| 中21–45、後46–70、>50、含50的50–70 | 各段hit/EV不回歸、至少其一嚴格改善、Brier≤+.0015 | 未評估 |
| 明確實證收益 | 2000次paired whole-shoe bootstrap，各指定中後段至少一個Δhit或ΔEV的95%CI下界>0；另項點估計仍須過gate | 未評估 |
| Core/PF核心/production完整性 | 訓練前後hash一致 | 本次已確認 |

另完整輸出41–50、wagers、non-tie wagers、正確注數、最大連敗及ΔBrier/ΔSkip信賴區間。所有比率從重抽的靴合計重新計算，不平均逐靴命中率。CI不足或任何分段缺有效出手，停止晉升；資料不足不能當成品質通過。

成功只得到`eligible_for_manual_release_review`，**promotion_allowed仍false**，沒有寫入production、GitHub push或自動部署。仍需審核資料來源、時間/靴隔離、實際commission、模型序列化/瀏覽器一致性、真機與開通/撤銷/新靴回歸。失敗會記錄`holdout_failed_no_promotion`或`execution_failed_no_promotion`及log路徑。

## 6. 目前維持兩套階段的理由

1–20/21–45/46–70是平滑/抽樣的反應速度區間；≤40/41–50/>50是既有風險門檻區間。同一局同時使用不同用途的區間不是必然矛盾；尚無資料證明它是退化根因。

這一輪完整重訓先固定兩套區間，以隔離權重效果，並輸出雙分段成績。不在模型驗收前一併改成20/45，避免前21–40局風險規則突然改變。模型通過後，另在新的開發集比較「只對齊階段」「只改EMA」「組合」，以新的獨立holdout驗收；不可用已開封的這輪holdout調參。

後段決策α=.72並不明顯保守；提高α不等於增準。Macro係數保持0；只有完整觀測特徵與獨立驗證支持非零係數才另案校準。本輪不改cardcount_profile版本，因正式參數未變；runner/報告schema_version=1記錄新執行流程。

## 7. 通過後的替換與回滾

先人工核對execution_report、production_gate、paired_bootstrap及凍結hash，確認候選的外層正式runtime報告全過。發佈同一個隔離candidate的**已固定policy/PF設定**瀏覽器bundle；不要改用未回放過的原始XGB export。

正式替換應成套包含physics_multitask_model.json及配對joblib、已驗收final_probability_model.json，以及需要提供離線用途的candidate XGB joblib；後者的離線decision設定若與瀏覽器固定設定不同，須清楚標記，不能當瀏覽器策略對照。57D、Core、PF核心及現行profile不變。記錄release manifest與回滾提交，先跑必要CI及瀏覽器一致性測試，再部署。若部署時又改policy、平滑或profile，就需重新驗收該完整bundle。

回滾取本次`baseline/`完整副本的Physics JSON/joblib、XGB JSON及profile，核對hash後成套還原；不混用新舊noise/calibration。Git使用`git revert <確認的模型升級提交SHA>`，不reset/force-push。Pages重新發佈與瀏覽器載入需要時間，現有架構不保證一秒全球生效。

## 8. 資料到位前的下一步

提供完整chronological資料及可追溯的靴順序；完成獨立環境安裝後執行上述runner。當前不能據缺失資料判定哪組參數最好，也不進行第四步平滑/階段調整。若未來gate失敗：先按失敗分段查Brier/校準/noise與有效樣本權重、維持production；下一輪只在開發集調整，使用新的獨立靴驗收，不放寬此次門檻。
