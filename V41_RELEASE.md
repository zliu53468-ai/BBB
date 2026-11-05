# BBB V4.1：數值精度與品質升級

基準：48f6451aac009d0d2eca1cba341538780649398b。
版本：BBB_PF_EOR_DEPLETE_V4_1_PRECISION_MODERATE。

## 最終參數

PF96、每粒子 MC256、EOR24×96、Deplete8×32×128=32768 shadow。
minEvGap=.005、noiseGuardMultiplier=.70、minPosteriorQuality=.50、
eorFragilityFloor=.025、crossCheckZ=2.5、crossCheckFloor=.008。
0.80 僅留在 tools/v41_compare.js 作實驗，不暴露為未驗證的正式切換。

沒有足夠真實資料證明 0.80 優於 0.70，因此依要求保留 0.70。
CONFIG 仍集中於既有核心，新增型別、範圍、不可變快照檢查；不另增設定載入順序。

## 不變的計算權限

只輸入本局閒／莊最終總點各 0～9；每次建立全新 PF。
candidate 仍為 EV_B >= EV_P ? B : P；B/P/T 原始機率不平滑、不中心化、
不強制莊閒比例。EOR 僅否決敏感度脆弱訊號；Deplete 只核對、主方向融合 0%。
Markov 僅旁路，LinUCB／上下文／牌路不實例化。
授權流程只更新前端資產快取版本，會員機制與 LINE 均不修改。

physics_forecast_first_runtime.js、quant256_online_runtime.js 與
deplete_mc_runtime.js 全部保持基準原檔。已檢查 summarizeHypotheses 的
粒子內 MC 方差與扣除預期抽樣分散後的後驗均值方差：總標準誤仍為兩者平方和
開根號，呼叫端明確傳 256，不改預設函式或重複加上原始粒子分散。

## A/B/C 對照（不是實戰勝率）

| 組別 | MC/粒子 | EOR | 雜訊倍率 | 莊/閒/觀望（100 等權輸入） |
|---|---:|---:|---:|---|
| A 基準 | 160 | 16×96 | .70 | 27/16/57 |
| B 正式候選 | 256 | 24×96 | .70 | 29/8/63 |
| C 可選實驗 | 256 | 24×96 | .80 | 25/7/68 |

本機 Node 對照平均 MC gap SE：.015111→.011959（約減少20.9%）；
總估計 SE：.015579→.012301（約減少21.0%）。
平均時間93.2→97.0ms，p95 130.1→131.7ms；這是本機測量，不是手機數字。
時間會受裝置、引擎與熱身影響，CI 會重新產生原始數字。

PF/Deplete 原始方向相反的點數組合數52→56；不能把數值精度改善說成
模型分歧或命中率也改善。Deplete tolerance 與 opposite veto 全部保留。

SKIP：A為weak_ev_gap52、deplete_disagreement2、opposite_deplete_direction2、eor_fragile1；
B為weak_ev_gap60、deplete_disagreement1、opposite_deplete_direction2；
C為weak_ev_gap65、deplete_disagreement1、opposite_deplete_direction2。

EOR穩定性為8組固定輸入、每組4個診斷亂數流、同一後驗粒子集合：
10點值平均差分變異約7.7394e-5→4.1000e-5（約減少47%）。
僅有限重抽樣診斷，部分個案反而變大；不宣稱統計保證或改選較好亂數種子。
這些測試亂數只在VM沙箱替換EOR stream，不進入正式程式。

## 真實資料限制

INSUFFICIENT_REAL_HOLDOUT_DATA。
倉庫現有holdout報告標為synthetic_only_not_real_holdout，未找到可核實、
具連續時間順序且含雙方總點與下一局結果的真實測試資料。
不填造真實命中率、扣佣損益或其信賴區間，不以100組網格宣稱勝率提升。

## 檢查與效能

node test_pf_eor_v4.js
node test_license_storage.mjs
node test_end_analysis_reload.js
node tools/v41_compare.js

本機核心與授權測試通過。專用CI另跑真正headless Chrome桌面與觸控模擬、
390×844、4倍CPU降速，核對PC Enter、手機Enter不分析、手機按鈕正常分析，
並分別測量A/B延遲。模擬環境不是實體手機測量。
瀏覽器測試使用本機fixture，不修改或繞過正式站的會員授權。
CI 上傳 v41-numerical-and-browser-report artifact。

正式部署必須等必要CI成功，再合併；最後核對bgsv1.xyz實際檔案及Pages部署SHA。
