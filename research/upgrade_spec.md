# 第五階段：命中率優先、正式系統零變動的升級規格

## 已核對現有狀態（2026-10-27 main）

`cardcount_profile.json` 之 `first_layer_smoothing.enabled=true`，分段平滑 alpha 為 0.48/0.58/0.65；`selective_entry_v2.enabled=true`，`require_first_final_consistency=true`；分段 EV threshold boost 早/中/後為 0.005/0.0035/0.0025，noise threshold 0.55，PF quality threshold 0.60，noise extra boost 0.004，volume guard 目標出手率 25%/31%/37%。正式版本已經有方案 A 的大部分元素，**不可疊加第二份相同功能**。

所有下列值均屬**受控研究候選**，不是驗證過的有效調參；應先輸出 shadow 決策並在獨立時間樣本外驗證，不能以測試集挑勝率門檻。

| 方案 | 研究修改點 | 候選數值 | 出手頻率影響（非實測） | 預期命中率與 EV | 可回滾旗標 |
|---|---|---|---|---|---|
| A 優先 | 第一層/最終決策/Volume Guard | 第一層只使用已完成局因果平滑（優先沿用現有）；同向檢查維持強制；在現有分段 boost 再加早 0.002、中 0.0015、後 0.001；Volume Guard 再縮 5%-15% 需測 | 以 10%-25% 為實驗設計目標，未驗證 | 可能提升已下注局命中率，也可能只是降低筆數；每靴 EV 可能趨近零，且無法保證正值 | `research_strategy_A_enabled=false` |
| B | 最終決策/品質門檻 | noise>0.60–0.80 或 ESS/N<0.35–0.55 時，額外 EV boost 0.002–0.008 | 目標 5%-15%，未驗證 | 過濾高噪音但不得宣稱有效 | `research_strategy_B_enabled=false` |
| C | 前 1–20 局最終門檻 | 只對早段額外 boost 0.003–0.010 | 目標 5%-15%，未驗證 | 只在真實前段 OOS 較差時啟用，否則不改 | `research_strategy_C_enabled=false` |

## 必須保留的診斷欄位

一局一筆，事前決策先寫日誌、結果開獎後才回填：

`shoe_id, shoe_start, round, history_fingerprint, prediction_timestamp, outcome_timestamp, outcome, first_layer_raw_gap, first_layer_smoothed_gap, first_layer_direction, final_direction, direction_mismatch, pB, pP, pT, final_ev, ev_threshold_base, ev_threshold_effective, physics_noise, pf_ess_ratio, pf_quality, pf_blend_weight, pf_resampling_count, volume_guard_allow, volume_guard_target_rate, stage, placed, direction, reason_code, runtime_version, config_hash`。

若現有 PF 沒有可比較的 `ESS`，標記 NA 而非填 1 或 0；區分「沒有 ESS 診斷」與「ESS 極低」。任何 `outcome` / `settlement` / `ground_truth` 不得進入事前決策。上線前檢查 `prediction_timestamp<outcome_timestamp` 與靴 ID 不可跨 split。

## 影子實驗流程與接受閘門

1. 完整靴切分，真實資料按 `shoe_start` 時間順序切成 train/valid/test；若新模型/門檻只在 valid 決定，test 僅用一次。
2. 影子模式每局同時計算 **現行**、A/B/C **候選**，候選不得影響任何現行下注／登入／授權。只比較真實已知的已開局 outcomes；有任何時間回填即排除。
3. 必報：候選與現行兩者的已下注局 B/P hit rate（和局從 hit rate 分母排除）、每注 ROI、每機會平均報酬、出手比例、最大回撤及有效靴數。用整靴配對 bootstrap 95% CI，不能僅挑高 hit rate。
4. 通過條件：事先定義主要指標，在完全獨立 test 上對 hit rate 的差異有清晰證據，且每機會的經濟報酬沒有不可接受的惡化，並無風控降級；若 CI 跨 0，只能標記「未證實改善」。多方案比較需控制多重檢驗。
5. 功能旗標 **預設 false**，只可在研究／shadow 驗證；另有全域 `research_selectivity_shadow_enabled`，`false` 時不生成替代決策，研究檔不會被正式 runtime import。回滾不需改既有權重或正式設定。

## 預期代價與避免的錯誤

選擇性出手率下降不必然造成條件勝率上升；若信號本身沒有資訊，嚴格濾網可能只讓樣本變小，使 CI 更寬。從既有下注集合離線刪掉部分下注屬描述性回放，非獨立前瞻實驗。提高「已下注局命中率」也不等於提高扣佣後 EV：莊家獲利一次只有 0.95 單位、輸則 -1；把和局算成「命中」會虛增命中率。

若 Gate 1 失敗，正式結論必須表述：**「目前沒有統計上可證明的 edge」**；不應把「該模型在測試集的 95% CI 下界 ≤0」錯誤解釋成資訊理論上已嚴格證明不可能。只可藉更選擇性出手測試 *條件命中率* 是否改善，並接受出手數下降；沒有可證明的訊號時，不能靠選擇性本身創造正期望。
