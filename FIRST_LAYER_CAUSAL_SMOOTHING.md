# 第一層專用輕量因果平滑

2026-10-27。基準為目前main 05ec475838aa04e93a1ad31d434056210bd7b7b2，保留PR #65架構與已部署PR #66選擇性篩選。架構標籤不變，新增政策以first_layer_smoothing.schema_version=1與獨立first_gap_v1狀態識別。

## 已確認的噪音行為

使用者已明確選擇：照指定公式與alpha≤0.70，高噪音時降低alpha、增加舊值權重。這不等於接近raw，不做raw bypass；原指令中兩者互相衝突的部分，以此確認為準。

## 完整設定

```json
{
  "first_layer_smoothing": {
    "schema_version": 1,
    "enabled": true,
    "alpha": {
      "early": 0.48,
      "middle": 0.58,
      "late": 0.65
    },
    "max_alpha": 0.7,
    "min_gap": {
      "early": 0.012,
      "middle": 0.008,
      "late": 0.005
    },
    "noise_high": 0.7,
    "noise_low": 0.45,
    "quality_low": 0.45,
    "quality_high": 0.75,
    "uncertainty_high": 0.7,
    "uncertainty_low": 0.45
  }
}
```

基礎alpha平台為0.48／0.58／0.65；以18→23與43→48局的短線性轉場跨過20→21與45→46，避免突然跳變。硬上限始終0.70，即使設定max_alpha較高也不突破。階段min_gap仍依1–20／21–45／46–70切分。

noise_gate=clamp((noise_high−noise)/(noise_high−noise_low),0,1)；quality_gate=clamp((quality−quality_low)/(quality_high−quality_low),0,1)；uncertainty_gate=clamp((uncertainty_high−uncertainty)/(uncertainty_high−uncertainty_low),0,1)。effective_alpha=base_alpha×(0.55+0.45×三個gate乘積)。gate為0時例如early alpha=0.264，保留更多舊值。

raw_gap為補牌後無條件EV_B−EV_P。只有gap接受EMA；banker_ev、player_ev、B/P/T機率與positive_ev保留raw結果，不製造平滑後的收益或勝率。noise使用與57D physics_noise_score同口徑的Direct Physics計算，不讀取後段平滑機率；品質與不確定性取第一層Physics＋PF輸出。

## 獨立與因果狀態

獨立鍵為bgs_first_layer_gap_smoothing_v1，沿用原授權儲存隔離與清除機制。沒有讀取後段physics_smoothed_p_b或訓練快照作為前值。

每次預測只讀本靴已完成局的第一層原始gap與gate輸入，回放重建previous_smoothed_gap。重複分析不寫入或推進狀態。報結果前擷取ticket；結果持久保存成功且前綴吻合後才提交該次預測輸入。實際B/P/T只用來驗證紀錄與撤銷前綴，不進入EMA數值公式。

撤銷時刪去無效後綴並回放；跨靴／模型／PF版本／設定／預估總局數隔離。沒有相鄰前次有效值、缺測、錯誤資料或儲存失敗時，初始化為本局raw_gap。新靴或結束分析會清除獨立狀態。無須重新執行所有歷史Physics/PF以重建EMA，只回放輕量的數值日誌。

## 後段隔離

後段PR #59 EMA、Clip、決策門檻與Volume Guard等受保護的數值計算函式及參數保持原樣；Macro、最終EV數學及校準閘門不變。整合點只調整一致性所讀取的首層方向，以及新增獨立日誌鉤子。平滑gap不進入48D、57D或final_p。

現有一致性檢查改用first_layer_ev.smoothed.direction。這是唯一允許改變實際出手的路徑；EV與confidence等原資格仍須通過。相同歷史與相同後段快照輸入下，最終機率與EV保持一致；若首層一致性狀態改變，可造成Guard略過或正式Skip的差異。連續運作時，已實際改變的下注紀錄仍是既有Volume Guard的歷史輸入，不能把不同歷史快照當成同一輸入。

本機校準的版本識別排除本新增區塊；其原始／候選動作比較繼續使用修改前未平滑首層方向，避免首層EMA重新定義校準基準。沒有更改校準模組、閘門或數值。

## 診斷／匯出

first_layer_ev與settlement_audit.first_layer都新增raw與smoothed。raw包含banker_ev、player_ev、gap及未平滑EV方向；smoothed包含gap、direction、alpha_base、alpha_effective、三個gate、physics_noise_score、min_gap、previous_gap、initialized及stage。頂層stage為early/middle/late，原計算階段保留在pipeline_stage。

頂層direction為平滑後方向；positive_ev仍由raw EV判定。unsmoothed_direction、unsmoothed_effective_gap與unsmoothed_gap_min保留PR #66原首層篩選結果，供對照及後段校準使用。所有欄位隨原分析快照匯出，未新增正式下注紀錄或改動不觀望測試邏輯。

例如raw EV_B=-0.003、EV_P=-0.020，raw gap=0.017；前值=-0.003、有效alpha=0.48時，smoothed gap=0.0066，小於early 0.012，因此Balanced，positive_ev仍為false。數字為說明範例。

## 驗證

node test_first_layer_smoothing.js：指定平台／插值／alpha上限與可覆蓋數值；高noise／低品質／高uncertainty；初值、0前值、缺前次、未來紀錄忽略、去重、寫入失敗、重載、撤銷、跨靴及結束清除；raw／smoothed／positive_ev與匯出。

固定弱反向交替測試raw翻向15次、smoothed翻向0次，只是機制示範，不能視為實戰改善。15組由修改前來源產生的golden案例確認enabled=false時首層完整物件與後段結果還原；啟用時只允許一致性所造成的動作差異，機率、EV、48D／57D、後段EMA與校準比較資料一致。

fixture同時鎖定Clip、決策EMA、決策門檻、Volume Guard、結算、顯示與不觀望函式來源，以及Core、PF、Macro、校準模組、兩個模型JSON、HTML/CSS雜湊。歷史首層／EV／selective_entry_v2／階段測試以關閉新平滑隔離原功能；新套件測預設啟用。按鈕回歸另驗證每次結果只提交一次、重算不推進、結束清除。

## 回滾

設first_layer_smoothing.enabled=false並重新載入即可回復修改前未平滑首層的完整方向與數值，包括目前PR #66既有effectiveGap篩選；不會順便撤銷PR #66。停用時不讀寫此EMA狀態，也不改變後段校準識別。已記錄的實際牌局與下注不倒轉。

完整回滾可revert本次merge commit。模型、PF500與抽樣／混合、佣金、和局、後段EMA／Clip／Macro／最終EV／Guard參數、2pp、不觀望按鈕、授權及30分鐘設定均保留。沒有重訓或替換權重，也沒有實戰收益改善主張。
