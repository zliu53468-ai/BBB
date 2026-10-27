# Physics＋PF 第一層 EV 候選｜PR #64 基準

2026-10-27，版本 PHYSICS_57D_V14_8_FIRST_LAYER_EV1。

## 改動

Physics與PF完成既有抽補牌推估後，立即用混合分布的無條件B/P/T機率計算EV。預設EV(B)=0.95B−P，EV(P)=P−B；和局退回淨收益0，佣金從既有settlement設定取得。首層EV不讀取後續Macro、Clip、EMA或逐局校準結果。

第一層candidate改為EV較高方；EV相同才為Balanced／Skip。即使兩邊EV都負，仍可有初步偏好，但positive_ev=false，絕不因此取得下注資格。原先經門檻判定的候選另存legacy_primary_candidate，方便比對。

之後維持原Macro、Clip、PR #59 EMA、逐局校準及最終EV Guard／Volume Guard。最終仍依後段機率重新比較EV，因此最終方向可與首層不同，亦可觀望。首層不鎖死後段、不放寬下注門檻；這是把初步EV選向明確放到第一層，不是把EV拿來偏選抽牌或保證增加下注。

畫面仍顯示通過後段處理的機率方向與正式下注狀態，不改版面或按鈕。匯出分析紀錄新增first_layer_ev與settlement_audit.first_layer，含stage、B/P/T機率、banker_ev、player_ev、direction、positive_ev，便於測試首層與最終差異。Physics/PF機率與57D特徵不因EV而重寫。

## 設定／回滾

cardcount_profile.json新增first_layer_ev={schema_version:1,enabled:true}。enabled=false可恢復PR #64原本第一層候選邏輯，最終EV選向不變。方向模式切回舊physics_candidate_v1時也不套用新首層候選。完整回滾可revert本次merge commit；上一版main為59caa1f8e33e6c1a8954ebd812013a02c299f224。

新版runtime版本／設定會重新隔離本機校準日誌；之後持續依實際B/P/T學習，通過既有閘門才套用。模型權重、PF粒子與抽樣、補牌規則、Core、48D／57D、PR #59平滑、逐局校準閘門、畫面、開通碼及30分鐘限制不改。auth-client僅更新runtime快取版本。

## 驗證

node test_first_layer_ev.js：扣佣前後選向、和局、負EV僅初步偏好、原門檻Skip仍可提供初步候選、回滾、15組不同歷史與前局EMA比較、匯出快照，以及抽補牌機率／57D／Macro／最終機率與下注風控不變。

既有EV、補牌、PF、校準、runtime、Macro、profile、中後段、UI與保存測試保留。此版不重訓正式模型，未宣稱實戰命中率或EV收益提升。
