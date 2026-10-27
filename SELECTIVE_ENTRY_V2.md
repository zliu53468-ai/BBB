# 最終出手選擇性 v2｜PR #65 基準

2026-10-27。僅新增決策篩選，不重訓或改變模型／PF抽樣。架構標籤維持PHYSICS_57D_V14_8_FIRST_LAYER_EV1，新政策由selective_entry_v2.schema_version=1識別。

## 設定

```json
{
  "selective_entry_v2": {
    "schema_version": 1,
    "enabled": true,
    "require_first_final_consistency": true,
    "ev_threshold_boost": {
      "early": 0.005,
      "middle": 0.0035,
      "late": 0.0025
    },
    "noise_threshold": 0.55,
    "pf_quality_threshold": 0.6,
    "noise_extra_boost": 0.004,
    "volume_guard_target_action_rate": {
      "early": 0.25,
      "middle": 0.31,
      "late": 0.37
    },
    "first_layer_effective_gap": {
      "early": 0.014,
      "middle": 0.009,
      "late": 0.005
    }
  }
}
```

## 方向一致性

先由最終機率算EV偏好，再比較first_layer_ev.direction。首層B/P與最終偏好不同時，正式方向強制Skip，該次最終判斷不呼叫Volume Guard，execution_order也不列volume_guard。首層Balanced／Skip／不可用則不增加方向限制。

direction_consistency.final是進入風控前的最終EV偏好（B/P，等值為Skip）；正式下注動作另在direction／bet_direction。因此首層B、最終EV偏好P時，紀錄first_layer=B、final=P、consistent=false、forced_skip_by_consistency=true，而正式動作為Skip。此資料寫入根預測、finalProbability、settlement_audit及匯出快照。

畫面仍依既有最終機率與2pp門檻顯示方向，下注狀態可以是觀望；不觀望測試仍只供顯示。沒有修改HTML/CSS或顯示規則。

## EV加嚴

階段以≤40／41–50／>50，分別加0.005／0.0035／0.0025。57D的physics_noise_score>0.55或PF粒子品質<0.60時再加0.004；兩者同時發生只加一次。缺少有效noise／品質時保守視為風險條件。

要求是選中方向的正EV高於門檻，不取負EV的絕對值。原base門檻與經Volume Guard調整後的門檻都加同一boost；有效門檻=原有效門檻+boost，Volume Guard無法抵消新加成。原confidence band、noise規則、佣金及和局退回均不改。

selective_entry_v2診斷包含階段、base／extra／total boost、觸發原因、noise、PF品質、原與新門檻、Volume Guard是否被執行及其結果，也寫入匯出快照。

## Volume Guard基準差異

目前正式final_probability_model.json實際是window=12、min_history=5、expected_correct_floor=0.93、目標率0.58／0.60／0.66，與任務引用的舊版16／8與0.30／0.36／0.42不同。

依「只改目標率、其他保持不變」，保留現行12／5、0.93及band／EV relief上限，只將目標改成0.25／0.31／0.37。目標override取與原目標的較小者，不能意外提高出手目標。expected-correct-floor仍可觸發Guard，但新增EV加成仍無法被消除。

## 首層effectiveGap

原補牌後EV不改。effectiveGap=(EV_B−EV_P)×(1−0.45×physics_uncertainty)×(0.55+0.45×particle_quality)。以1–20／21–45／46–70套用0.014／0.009／0.005最小差距；低於門檻只把首層方向改為Balanced，不翻向、不改機率、不增加出手機會。缺品質時使用0保守估計。

首層新增raw_direction、effective_gap、effective_gap_min、effective_gap_stage、balanced_by_effective_gap與使用的品質／不確定性值，均隨first_layer_ev匯出。

## 驗證

node test_selective_entry_v2.js：15組由修改前PR #65來源產生的golden案例，關閉政策時機率、EV、動作、門檻、Guard及特徵雜湊相同；校準模型識別也與原版一致。啟用時3組實際方向衝突強制Skip並略過Guard，4筆首層Balanced仍可出手，總下注11→7，沒有新增原版不會下的注。另驗證高noise／低品質額外門檻、三階段邊界、Guard無法抵消boost、視窗／最少歷史維持現行值及匯出欄位。

歷史test_first_layer_ev.js、test_ev_led_direction.js與test_middle_late_upgrade.js以enabled=false隔離原功能回歸；新套件專門測試enabled=true的預設政策，避免把新要求的強制Skip誤判成舊功能失敗。既有最終機率、補牌結算、Volume Guard、PF、校準與UI回歸繼續執行。

此為固定機制案例，不是獨立chronological hold-out，尚未證實實戰hit-rate或EV改善；Skip預期增加，但不宣稱已滿足收益晉升門檻。

## 回滾與保留

設selective_entry_v2.enabled=false並重新載入，回復PR #65同一歷史輸入下的首層／最終決策、門檻與Guard目標；區塊缺省或schema不支援也回到原值。關閉時校準fingerprint排除本區塊，維持原識別；開關政策不倒轉已發生或已記錄的牌局，新政策學習紀錄依原版本隔離機制處理。

完整回滾可revert本次merge commit；前版main為bd143996eef289671c40e4ee6c1d430c4aa6b8c2。auth-client.js只更新runtime快取版本。

Core、Physics213→48、48D與57D定義、XGB輸出、PF500及4/4與3/4/5抽樣、補牌／佣金／和局邏輯、PR #59平滑、本機校準閘門、授權與30分鐘、畫面及按鈕全部保留。
