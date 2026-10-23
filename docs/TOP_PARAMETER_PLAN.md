# BBB 算牌優先｜前段保護、中後段穩定化參數方案

日期：2026-10-23。凍結 production：main `09166e1f7fc7b4e7c3bb4448af424a611b8f92ed`，tree `79371288fe590a543a918eaedac83d7489133c74`。本機 checkout tree 相同。已核對指定四份文件、現行設定、validation、runtime 與兩個 trainer。

**結論：目前不能誠實宣稱存在已達所有門檻的最佳參數。主要瓶頸包含驗證與正式機率路徑不同、狀態更新頻率不同，以及缺少目前 production 的真實獨立 holdout。先修驗證與凍結對照，再搜尋參數；不可直接放大 α、機率或中後段 loss 後部署。**

這次交付隔離候選與唯讀驗證工具，沒有修改 production、開通碼、Frozen Core、Physics/PF 核心、57D 或正式權重；沒有執行完整重訓。main 的上一筆合併也明確記錄「未替換正式權重」，不能把逐局本機校準稱為完整重訓。

## 1. 診斷：確定事實與待驗證假說

### A. EMA：中後段已敏感，先釐清哪一層真的生效

- 現行決策最新機率權重平台為早 .45、中 .60、後 .72，交界插值；noise/uncertainty 再扣，硬限 .35～.75。後段 .72 表示每次有效更新只保留 28% 前值，不是慢速平滑。固定 α 近似下，舊值兩次更新後僅餘 7.84%。進一步加速可能放大 PF 抽樣或模型估計抖動，實際是否如此要做消融。
- `applyDynamicSmoothing()` 讀取同靴已結算訓練快照的前值，不是獨立逐觀測 EMA。稀疏分析時以跨多局的舊值混合；不觀望自動顯示沒有正式 registerPrediction，可能無前值而 α=1。Python 則對每個二元資料列更新，和局先被訓練標籤過濾。兩者不是同一條時間線。
- 模型 bundle `smoothing.enabled=false` 不代表瀏覽器關閉決策 EMA；runtime 會回落至 Physics-primary EMA，再由 cardcount_profile anchors 控制。
- Macro 平台 short .28/.33/.35、long .08/.11/.12；方向係數全為 0，所以目前調快 Macro 本身不會提升正式方向或 EV。缺少真實張數、點數時，多個 Macro 特徵也未必達 ready；B/P/T-only 不能被當成真實牌值。
- **建議：前 20 局逐值保留；中後段同時比較原值與略降敏感度候選。Macro 係數維持 0。** 不在本次參數方案中改寫狀態機。

### B. 重訓：已有中後段監督，名目權重不等於有效權重

- middle_late_focus 的 XGB 平台是 1.10/1.25/1.35；Physics train-only 額外抽樣中段 10%、後段 20%。不能說程式完全沒加強中後段。
- 低 uncertainty 與 actionability 乘數上限約 1.18、1.08，後段名目乘積 1.35×1.18×1.08=1.72044；之後有類別平衡、全體均值正規化、.35～1.50 clipping，實際相對權重依資料分布而變。只看 1.35 看不出有效監督量。前段參數不變也不能保證重訓後前段輸出不變。
- Physics 現行 aggressive/focus loss 相同：winner3.50、points1.45、card_count1.65、rank.55；早已高於先前2.7～2.9建議。再加大未必有益。各 head 的總梯度還受維度與 loss 正規化影響。
- MLP 的外層 train/calibration/holdout 是 whole-shoe，但 sklearn `early_stopping=True` 的內部 validation_fraction=.15 並非 whole-shoe chronological。訓練集重複抽樣更可能讓近似/重複列跨內部停止集合。這不等於污染外部 holdout，卻使內部停止分數過度樂觀；正式決策只能依外部獨立評估。
- 必報每段：原始列數、獨立靴數、重抽列數、normalized weight 的均值/分位數/總和、觸及1.50比例、effective sample size=(Σw)²/Σw²。

### C. Guard/band：Volume Guard 本身是放寬，不是壓制

- 現行 EV 門檻≤40/41–50/>50=.002/.002/0；confidence base=.006/.005/.004；已是偏低門檻。
- Volume Guard target=.58/.60/.66、window12、min_history5。它不足出手時提供 relief，不是硬性限流；不能把所有 Skip 歸罪於它，也不能把 target 當實際出手率保證。
- Guard 讀正式快照；「預期正確數」是機率代理，不是真實命中數。冷啟動、稀疏分析與機率失準影響啟動與解讀。
- 最終 baseBand=max(policy band, Physics band)，candidate 已 Skip 時，調 Guard 不能救回。須按第一個失敗 gate 統計，不是一起降低所有門檻。
- dynamic_band.enabled=true 時，confidence_band 的 noise_gain / clean_late_relief 被動態計算結果取代，直接改這兩個舊鍵可能沒有作用。實際生效 noise_gain 是 dynamic_band.noise_gain=.015，進度 relief=.40（硬上限也是.40），noise gate=.60/.75。
- 動態 relief 又受 band/activation 的25%上限，現有 band_relief=.010、ev_relief=.006 常先撞 cap；再提高這兩值可能無效。

### D. 本機逐局校準與正式模型不是同一件事

- 本機校準只作用於畫面／不觀望測試，沒有改正式 finalPB、bet_direction、EV或Guard。調 prior/decay/window 不能宣稱提升正式 bet hit/EV。
- early/middle/late prior=100/90/80、decay=1/.98/.96、視窗32/24/20。最少24個二元樣本，再16個有效提案，最早約40個二元結果後才可能啟用，和局不算二元樣本。中段可學習時間有限。
- late .96 的固定 decay 幾何記憶尺度約25次更新，20筆比較窗又短，易受短期變異影響；是否過敏感須配對回放，不能推論已過擬合。
- config/model fingerprint 改變會重新累積，資料各瀏覽器隔離；沒有跨裝置完整重訓。校準 gate 用 forecast uncertainty，並非正式 mapped noise_score，兩者數字不可直接互換。

### E. 校準與正式 holdout 的 mismatch 是首要問題

- 正式 `finalPB=filteredPB=smoothedPB`，Core/XGB 是 advisory_only_no_veto_no_flip，未接管正式機率。XGB 機率變尖或校準 Brier 改善，不必然改變正式出手。
- Python 主評估使用 XGB 機率；模型搜尋基準為第一個重新訓練 candidate，最終 legacy 對照也重新訓練，並非凍結 production+PF+Guard。
- production bundle 的校準 probe：Identity Brier .24929247、Platt .24968157、Isotonic .24986642；目前選 Identity 有數據理由。這是歷史校準 probe，不是本次真實 holdout，也不是目前 Physics正式路徑的Brier。
- 校準選擇目標=.70整體Brier+.30有效進度≥.70的Brier，沒有前1–20/≤40硬保護。資料少於500列時不能完整比較，會走Platt預設；不要把小資料自動校準視為可靠。
- 帶57D snapshot 的資料優先使用snapshot；即使帶新Physics模型，也不會自動重建全部48D。若又附PF校正的 physics_48d，snapshot loader會以它重算進度/noise，而48D主塊仍可能是Direct Physics。重訓前必須驗證或用凍結Core與候選Direct Physics重新產生一致的57D，不能把舊snapshot誤當候選特徵。
- 現行部分 stage_decision_metrics 分段後重新跑決策，會重置Guard歷史；新增focus_stage_metrics才是完整時間線結果切片。舊 `late_50_70` mask 實為>50，不含50。本交付工具明確包含50。
- EV_CONSISTENCY_V1目前false，物理candidate EV含和局質量、最終EV使用條件B/P機率，單位不同；舊shadow未過門檻，所以本方案保持false，分開報條件機率和無條件三項機率，禁止盲開。

### F. noise/uncertainty：多重作用，尚無過重的實證

- noise同時影響Clip、EMA、dynamic band、EV與Macro gate；uncertainty另影響早期shrink、precision penalty、PF品質閘門。中後段有可能多次保守，但必須統計各階段截斷率才可下結論。
- 正式noise可從PF校正後物理向量計算，XGB仍以Direct Physics向量；訓練noise mapping與PF後分布的差異需實測。現有isotonic mapping並非新PF路徑的準確性保證。
- 先做高/中/低noise桶的Brier、reliability、EV及失敗gate統計；不要改noise_score特徵定義，也不要重新拉伸noise數值求出手。

### 現有驗證證據的界線

`validation/macro_ema_smoke.json`是舊版本8靴合成560局，不是目前production基準：整體hit52.85%→51.91%、EV .0397→.0236、Skip75.36%→73.91%；後段hit52.50%→48.94%、EV .0352→-.0284。它直接說明「更少Skip」不等於「更準」，但不能用來判定現在真實收益。3000靴Macro檢查也是合成，係數保持0。現有validation沒有足以完成此次晉升的production真實配對holdout。

## 2. 兩套候選：先穩定化，再訓練；均不是已勝出的最佳值

`prepare.py`會產生完整versioned JSON與隔離目錄。版本名稱寫入profile.name；不新增訓練CLI名稱，沿用middle_late_focus，避免改架構。

| 參數 | 現行早/中/後平台 | 僅平滑候選S | 完整重訓候選T |
|---|---|---|---|
| 決策最新p權重 | .45/.60/.72 | 前≤20逐值保留；中.58、後.66 | 同S |
| Macro short/long | .28/.08；.33/.11；.35/.12 | 保留，係數0 | 同S |
| PF每粒子抽樣 | 3/4/5 | 3/6/8 | 同S |
| PF混合cap | .10/.15/.16 | 前逐值保留；中.14、後.15 | 同S |
| local prior | 100/90/80 | 100/100/100 | 同S |
| local decay | 1/.98/.96 | 1/.99/.98 | 同S |
| local比較視窗 | 32/24/20 | 32/32/32 | 同S |
| XGB focus stage | 1.10/1.25/1.35 | 不訓練 | 前逐值保留；中1.22、後1.30 |
| Physics train額外抽樣 | 中10%、後20% | 不訓練 | 中10%、後15% |
| Physics winner/points/cards/rank | 3.50/1.45/1.65/.55 | 權重完全保留 | 3.40/1.40/1.60/.55 |
| 低u/actionability gain | .18/.08 | 不訓練 | .15/.06 |
| 規律額外權重 | 0 | 0 | 0 |

S是沿用原模型的平滑/PF抽樣參數消融，非單純只改一個EMA；如要嚴格EMA-only，先只套decision_alpha，其他鍵保留baseline。**逐組比較 EMA-only、PF-only、local-only，再比較組合，不能把組合收益歸因於某一項。** local-only正式下注結果理應完全相同，否則查實作或測試。

保留20局交界值；中後平台於24/43/49附近插值，PF抽樣於24/50附近整數化。前20局設定保留不代表≤40局結果不變，後者另設硬驗證。

### 搜尋空間：只在開發/校準集合搜尋

- 決策中α={.55,.58,.60,.63}、後α={.62,.66,.70,.72}；前1–20固定。Macro中short .30～.35、long .10～.12，後short .32～.36、long .11～.13；係數0時它們是狀態/就緒測試，不列成下注提升來源。
- PF中draws={4,6,8}、後={5,8,10}；混合cap中{.12,.14,.15}、後{.14,.15,.16}，仍受既有16%硬cap、同向與品質閘門。增加抽樣需要測低階手機耗時；不能保證按獨立抽樣理論等比例降低實際PF誤差。
- local中prior={90,100,120}、後{80,100,120}、decay中{.98,.99,1}、後{.96,.98,.99}、window{24,32,40}；不降低min_samples24、min_validation16、gain .00005，不提高delta .005。
- band先固定現行。第二輪才試confidence middle{.0045,.005,.0055}、late{.0035,.004,.0045}，early=.006固定；dynamic noise_gain={.012,.015,.018}。舊static noise_gain/clean_late_relief先不搜，因dynamic會覆蓋。progress_relief上限.40不突破。全域noise_gain會影響早段，必須過雙早段門檻。
- Guard target early=.58，中{.58,.60,.62}、後{.62,.66,.68}；band_relief{.006,.010}、ev_relief{.004,.006}；cap=.25、window12、min_history5固定。只接受實際有效relief有變化的組合，刪除撞cap的重複候選。
- XGB stage中{1.20,1.22,1.25}、後{1.25,1.30,1.35}；低u gain .10～.18；actionability .04～.08；pattern=0。Physics winner{3.25,3.40,3.50}、points{1.35,1.40,1.45}、cards{1.55,1.60,1.65}，rank .55不變，其餘head不變。這是對「現行已激進」的合理小幅搜索，不是盲加loss。
- XGB depth1～4，trees400～850、lr.015～.030、child8～20、alpha.10～.60、lambda10～20，subsample.85～.95、colsample.75～.95。lambda到20/child到20是為較深樹保留更強正則。已提供四組可執行額外候選；既有legacy/aggressive候選仍保留。
- 不跑龐大全排列。固定seed，逐項消融→少量組合→rolling whole-shoe開發驗證；一次凍結最終候選再開strict holdout。若反覆用同一holdout挑參，該集合必須退役。

## 3. 可執行命令（PowerShell；在倉庫根目錄執行）

建立隔離副本，不變更來源：

```powershell
python tools/top_parameters/prepare.py . ../BBB-top-experiment-01
```

比較正式runtime，輸入必須為完整觀測rows：shoe_id、連續round_index、outcome=B/P/T；可選實際card_count/player_score/banker_score。不可省略Skip或和局，不可把預測快照當完整觀測。

```powershell
node tools/top_parameters/replay.js ../BBB-top-experiment-01/baseline C:/BBB-data/strict_holdout_hands.json baseline_holdout.json
node tools/top_parameters/replay.js ../BBB-top-experiment-01/smooth C:/BBB-data/strict_holdout_hands.json smooth_holdout.json
python tools/top_parameters/gate.py baseline_holdout.json smooth_holdout.json smooth_gate.json
# strict策略才加 --strict，要求Skip增加至少2pp且最多5pp
```

`C:/BBB-data/...`為工程師實際獨立資料路徑，須先提供，非倉庫附帶檔案。工具要求每靴按順序且不交錯，先預測再讀本局結果。正式分析每局都執行，並settle/記錄前局再分析下一局；它不模擬所有UI操作頻率，也不測本機畫面校準。需要另外對稀疏分析、不觀望、撤銷、重載與真機做回歸。第一局無歷史依現有UI排除，報告應標示1–20實際從第2局開始，不能虛構第1局預測。

完整重訓在隔離retrain目錄；先安裝requirements-xgb.txt指定依賴。Physics使用既有模擬器；XGB資料需按下段規範事先製備。

```powershell
Push-Location ../BBB-top-experiment-01/retrain
python -m pip install -r requirements-xgb.txt
python physics_feature_extractor.py train --training-profile middle_late_focus --shoes 8000 --validation-fraction 0.20 --calibration-fraction 0.10 --augment-ratio 0.25 --random-state 20261023 --bootstrap-samples 2000 --output physics_top_candidate.joblib --browser-output physics_top_candidate.json
python xgb_final_probability.py train --training-profile middle_late_focus --input C:/BBB-data/development_rows_rebuilt_with_candidate_physics.json --physics-model physics_top_candidate.joblib --validation-fraction 0.20 --calibration-fraction 0.15 --tuning-fraction 0.15 --ev-tuning-fraction 0.30 --bootstrap-samples 2000 --random-state 20261023 --output final_probability_top_candidate.json --joblib-output final_probability_top_candidate.joblib --report-output top_training_report.json
Pop-Location
```

不可使用`--ci-smoke`晉升失敗候選。CLI的既有門檻保留，通過仍須另外跑凍結production正式路徑；失敗就停止，不把smoke產物部署。8000靴是計算建議，不是統計保證。

**資料製備前置條件不可略過：**使用候選Direct Physics重建48D與進度/noise、凍結Core取得pre-hand Core P(B)、原6D定義不變。不要傳舊features_57d/舊physics_48d後期望`--physics-model`自動覆寫。每列只見預測局之前歷史，不用未來牌值或未來uncertainty。Core、Physics及XGB的訓練/校準/strict靴不得交叉；最外層真實strict靴必須完全不進上述CLI input，CLI內部holdout只當開發期驗證。模擬Physics外部測試與真實strict報告分開。

T候選正式回放前，工程師將候選Physics JSON與XGB JSON複製到隔離retrain目錄對應的production檔名。確認Physics history_input_dim=213、physics_dim=48、activation=relu、coefs形狀與baseline相同；XGB feature_names與baseline逐項相同且57個、model_type仍binary classifier。為隔離「權重訓練」效果，先把XGB候選的decision_policy/smoothing還原成baseline相同內容（保留候選calibration與noise mapping）；原始候選完整留存。再跑：

```powershell
node tools/top_parameters/replay.js ../BBB-top-experiment-01/retrain C:/BBB-data/strict_holdout_hands.json retrain_holdout.json
python tools/top_parameters/gate.py baseline_holdout.json retrain_holdout.json retrain_gate.json
```

這段刻意不自動覆蓋模型：未經維度、語意與內部品質檢查的產物不可直接stage。若只重訓XGB，正式bet結果預期幾乎不變，因目前XGB只是advisory；不得為證明XGB有效而偷偷改其角色。

## 4. 晉升檢查清單：比對凍結production，不放寬門檻

1. 固定真實、完整、按時間最晚的strict靴；記錄資料hash、來源、時間、靴ID與所有模型/config/code hash。至少1000靴為初始樣本量下限，仍需依實際wager數及CI判斷精度，不是足夠證據的保證。data_kind須明確標記真實資料；此字串不能替代人工來源審核。
2. 相同時間線配對比較：hit=wins/非和出手；EV=固定1單位淨利/所有出手，莊勝+.95、閒勝+1、輸-1、和0；Skip=Skip/全部可分析局；Brier只在非和局計算條件P(B)。另報tie數、B/P wager分布、每靴/時期與最大連敗。若實際莊佣金不同，先鎖定相同實際賠付規則，不能混用。
3. 整體：hit≥baseline、EV/bet≥baseline、correct bets≥95% baseline、Brier差≤.0015、Skip差≤+.05；strict另要求Skip差≥+.02。不改成「hit/EV不得雙降」。
4. 早1–20及≤40：hit和EV皆不回歸；建議額外Brier差≤.0005（比整體更嚴）；任何指標無足夠樣本則不晉升。S前20局機率/方向還應逐列一致，T無法承諾逐列一致。
5. 中21–45與後46–70：各自hit/EV不回歸，至少一項嚴格改善；同時報Brier。報告另列≤40/41–50/>50及含50的50–70，全部取同一完整時間線切片，不能分段重跑Guard。
6. 以同一抽中的靴同時重抽baseline/candidate，至少2000次paired whole-shoe bootstrap，重新算比率而非平均逐局比率；報Δhit/ΔEV/ΔBrier/ΔSkip的95%CI。若要宣稱中後段有穩定實證收益，至少其一改善指標下界>0，且另一項的非劣性也有支持；CI跨0只稱觀察收益，延長shadow，不稱已證明。不能藉CI或樣本不足放寬點估計硬門檻。
7. 前綴標記長龍/單跳/雙跳，只做分組報告，不加未來標籤或pattern硬門檻。noise分桶、calibration曲線、prediction/EV前後一致性、逐gate Skip計數與手機p50/p95耗時一併檢查。
8. 規律不能改向或把Skip救成出手；開通碼、裝置隔離、無觀望切換、撤銷/新靴reset與鎖定檔hash通過。先shadow，再在人工核對後獨立部署。

`gate.py`自動檢查第2～5項的數值子集與鎖定程式hash，資料不足/合成資料/退化時exit2；**它永遠不自動授權部署**，bootstrap、資料真實性與模型形狀需另外完成。不能把exit0冒充全部驗收通過。

## 5. 最小diff與版本管理

- 第一批只改候選`cardcount_profile.json`，完整可讀JSON已由prepare產生；formal policy、正式權重、runtime與Core/PF核心皆不動。可用`git diff --no-index baseline/cardcount_profile.json smooth/cardcount_profile.json`取得完整diff。
- 第二批只在獨立候選套用上述training區塊、focus anchors與candidate lists；不新增模型，不改213/48/57及target，沿用middle_late_focus。
- 驗證修正建議位置：`xgb_final_probability.py::_snapshot_56d/make_training_dataset`做snapshot provenance檢查；`evaluate`的stage_report從完整realised/wagered切片、50–70改成>=50；部署比對改為讀凍結production回放報告而非legacy。這些是獨立驗證工具修正，不應混入參數收益歸因。
- `final_probability_runtime.js::applyDynamicSmoothing`的快照依賴、`dynamicBand/volumeGuardAdjustment`的有效cap與`auxiliaryCandidateFilter`的advisory角色寫入文件。本輪不改。若將來要統一逐觀測狀態，需另案，因它不再是純參數修改。
- `physics_feature_extractor.py`內部early-stop需另作whole-shoe停止集改善評估，維持拓撲；本輪不聲稱內部split已修復。外部strict必須獨立。
- 文件版本及manifest記錄baseline、候選JSON、資料hash、seed、選中calibrator、有效權重分布、CLI與完整指標。舊`validation/cardcount_moderate_v1_baseline.json`只供歷史消融，不能替代當前main基準。

## 6. 回滾

目前未部署，production不需要回滾。隔離實驗可直接停止，來源不變。已生成隔離實驗的 baseline 目錄保存本次production檔案與SHA256清單。

將來只部署S時，回滾完整profile到本次凍結baseline；部署T時，必須成套回復Physics JSON/joblib、XGB JSON/joblib、profile、相關決策設定及版本，不混用新舊校準。Browser fingerprint改變後重新累積本機校準屬預期，不可用舊分數判新模型。

Git回滾使用`git revert <已確認的升級提交SHA>`，保留後續歷史，不reset/force-push。Pages仍需完成部署、清快取或重新整理，**現有GitHub Pages架構無法保證一秒全球回滾**。真正秒切需事先部署兩套完整release並提供原子切換入口，屬額外部署工作，本次不偷偷加入。

## 7. 本次實際驗證與未完成項

已產生隔離baseline/S/T完整JSON與hash manifest。1靴70局合成機制測試，兩個runtime各69次因果預測；數值gate正確阻擋S（中段及≤40 hit/EV退步、資料為合成）。**因此這組預設候選不能晉升，也沒有最佳/收益宣稱。** 合成結果只測工具，不能替代真實holdout。下一步是將上述小網格放在開發集搜尋，真實strict只開封一次。

完整重訓、真實獨立holdout、bootstrap與實機效能尚未執行；在它們完成且全部通過前，保留目前production。
