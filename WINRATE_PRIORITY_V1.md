# Winrate priority v1

基準：main 22f7ff3（已含 PR66、PR67），單一開關停用行為以 PR65 bd14399 直接比較。

## 修改與接點

- `winratePriorityPolicy`、`selectiveEntryPolicy`：統一開關優先於舊設定；EV 加嚴及 Volume Guard 僅套用一次。
- `firstSmoothingConfig`、`firstSmoothingStep`：沿用 PR67 獨立同靴因果 journal。首筆以 raw gap 初始化，之後只讀取已結算且歷史前綴吻合的上一筆；撤銷後重播，換靴重置。
- alpha 平台為 0.48/0.58/0.65，18–23、43–48 局線性過渡，硬上限 0.70；gap 分段為 1–20、21–45、46+。高噪音、低品質或高不確定性降低有效 alpha。
- `applyFinalPrediction`：最終 EV 選向後、Volume Guard 前檢查方向一致性；補齊兩處診斷。
- `loadModels`：排除新區塊對既有校準識別的意外污染；關閉時還原 PR65 識別。
- 後段 EMA、模型、Physics、PF500、Core/57D、佣金、和局、授權、30 分鐘與 UI 規則皆未修改。

## 完整設定

```json
{
  "winrate_priority_v1": {
    "schema_version": 1,
    "enabled": true,
    "first_layer_smoothing": {
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
    },
    "require_first_final_consistency": true,
    "ev_threshold_boost": {
      "early": 0.005,
      "middle": 0.0035,
      "late": 0.0025
    },
    "noise_extra_boost": 0.004,
    "volume_guard_target_action_rate": {
      "early": 0.25,
      "middle": 0.31,
      "late": 0.37
    }
  }
}
```

## 診斷範例（示意值）

下列結構同時出現在 `first_layer_ev` 與 `settlement_audit`；final 表示最終 EV 選出的候選方向，強制跳過後實際 direction 為 Skip。

```json
{
  "raw": {
    "gap": 0.04,
    "direction": "B"
  },
  "smoothed": {
    "gap": 0.0148,
    "direction": "B",
    "alpha_base": 0.58,
    "alpha_effective": 0.58,
    "noise_gate": 1,
    "quality_gate": 1,
    "uncertainty_gate": 1
  },
  "direction_consistency": {
    "first_layer_smoothed": "B",
    "final": "P",
    "consistent": false,
    "forced_skip_by_consistency": true
  }
}
```

## 回滾

只需將 `winrate_priority_v1.enabled` 改為 `false` 並重新載入頁面。此開關會同時停用平滑、方向強制檢查、EV 加碼及新 Volume Guard 目標；舊開關即使仍為 true 也不會重新啟用。已實際結算的紀錄會保留。

## 驗證

- 23 個 JavaScript/MJS 測試中 22 個通過，包含所有指定的首層 EV、最終 EV、Volume Guard、本機校準、按鈕，以及新統一開關測試。
- 新測試 15 組直接載入 bd14399 runtime 比較：關閉時下注方向、信心、機率、EV、門檻、Guard 與首層診斷完全一致；開啟時最終機率與 EV 不變。
- 因果、撤銷回播、跨靴與高噪音降 alpha 已在新開關與既有首層測試驗證；既有測試亦鎖定後段函式與模型雜湊。
- Python 指定 5 個模組共 17 tests 通過。
- 既有 `test_blind_upgrade.js` 在本次與未修改 main 均失敗，期待已不存在的 PF `conditional_mode=rules_conditioned`；未修改 PF 以迎合舊測試。額外 `test_shoe_error_correction` 因倉庫缺少 `particle_shoe_filter` 無法載入。不可宣稱全部既有測試通過。

## 預期作用與實證界線

穩定第一層方向、排除前後矛盾、提高 EV 要求並降低目標出手率，可減少弱訊號下注。這是以出手品質為優先的篩選設計；尚無獨立實戰留出資料證明命中率提升，也不保證提升。
