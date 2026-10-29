# SYNTHETIC 管線回放統計

合成資料僅用於管線驗證，不得用於宣稱真實增量

合成資料的結果不能用來判斷真實增量，只能用來證明管線可運行。

狀態：INSUFFICIENT_SAMPLE。建庫→檢索→離線擬合→三組獨立 runtime 回放已完成。
建庫樣本：826 SYNTHETIC；k：30 SYNTHETIC；seed：20261029 SYNTHETIC。
本次小庫低於真實實驗規模要求，只驗證機制。

| 指標 | baseline | ungated | gated |
|---|---|---|---|
| standard_hit_rate | 0.42857143 SYNTHETIC | 0.5 SYNTHETIC | 0.42857143 SYNTHETIC |
| pct_shoes_max_losses_le_2 | 1 SYNTHETIC | 1 SYNTHETIC | 1 SYNTHETIC |
| pct_shoes_with_four_streak | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| mean_shoe_five_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| mean_shoe_six_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| action_rate | 0.09039548 SYNTHETIC | 0.12429379 SYNTHETIC | 0.09039548 SYNTHETIC |
| standard_actions | 16 SYNTHETIC | 22 SYNTHETIC | 16 SYNTHETIC |
| non_tie_standard_actions | 14 SYNTHETIC | 20 SYNTHETIC | 14 SYNTHETIC |
| shoes | 3 SYNTHETIC | 3 SYNTHETIC | 3 SYNTHETIC |
| light_actions_counted | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |

## 配對整靴 bootstrap 增量（候選減基線）

| 組別/指標 | 增量 | 95% CI（SYNTHETIC） | 多重比較同時 CI（SYNTHETIC） |
|---|---|---|---|
| ungated/standard_hit_rate | 0.071428571 SYNTHETIC | 0 SYNTHETIC, 0.25 SYNTHETIC | 0 SYNTHETIC, 0.25 SYNTHETIC |
| ungated/pct_shoes_max_losses_le_2 | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| ungated/pct_shoes_with_four_streak | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| ungated/mean_shoe_five_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| ungated/mean_shoe_six_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| ungated/action_rate | 0.033898305 SYNTHETIC | 0 SYNTHETIC, 0.10169492 SYNTHETIC | 0 SYNTHETIC, 0.10169492 SYNTHETIC |
| gated/standard_hit_rate | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| gated/pct_shoes_max_losses_le_2 | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| gated/pct_shoes_with_four_streak | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| gated/mean_shoe_five_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| gated/mean_shoe_six_window_coverage | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |
| gated/action_rate | 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC | 0 SYNTHETIC, 0 SYNTHETIC |

## 逐靴表

| 組別 | 靴 | standard 出手 | 勝 | 負 | 最大連敗 | 四連勝 | 五窗口 | 六窗口 |
|---|---|---|---|---|---|---|---|---|
| baseline | SYNTHETIC-00017 | 2 SYNTHETIC | 0 SYNTHETIC | 2 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| baseline | SYNTHETIC-00018 | 3 SYNTHETIC | 1 SYNTHETIC | 2 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| baseline | SYNTHETIC-00019 | 11 SYNTHETIC | 5 SYNTHETIC | 4 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| ungated | SYNTHETIC-00017 | 2 SYNTHETIC | 0 SYNTHETIC | 2 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| ungated | SYNTHETIC-00018 | 9 SYNTHETIC | 5 SYNTHETIC | 4 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| ungated | SYNTHETIC-00019 | 11 SYNTHETIC | 5 SYNTHETIC | 4 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| gated | SYNTHETIC-00017 | 2 SYNTHETIC | 0 SYNTHETIC | 2 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| gated | SYNTHETIC-00018 | 3 SYNTHETIC | 1 SYNTHETIC | 2 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |
| gated | SYNTHETIC-00019 | 11 SYNTHETIC | 5 SYNTHETIC | 4 SYNTHETIC | 2 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC | 0 SYNTHETIC |

## 資料切分

| 靴 | 局數 | 分組 | 狀態 |
|---|---|---|---|
| SYNTHETIC-00000 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00001 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00002 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00003 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00004 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00005 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00006 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00007 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00008 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00009 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00010 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00011 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00012 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00013 | 60 SYNTHETIC | library | ELIGIBLE |
| SYNTHETIC-00014 | 60 SYNTHETIC | validation_fit | ELIGIBLE |
| SYNTHETIC-00015 | 60 SYNTHETIC | validation_fit | ELIGIBLE |
| SYNTHETIC-00016 | 60 SYNTHETIC | validation_fit | ELIGIBLE |
| SYNTHETIC-00017 | 60 SYNTHETIC | validation_test | ELIGIBLE |
| SYNTHETIC-00018 | 60 SYNTHETIC | validation_test | ELIGIBLE |
| SYNTHETIC-00019 | 60 SYNTHETIC | validation_test | ELIGIBLE |

## 解讀限制

兩個模型的檢索 logit 係數均為 0 SYNTHETIC（由離線擬合所得）。
因此無閘門組的數值變化不能歸因於檢索增益；基線 logit 係數和截距的重校準、拒絕出手後兩敗停手時間改變，都會改變實際出手集合。
本次僅證明 KNN 查詢、融合擬合和三組回放可執行，沒有實證真實增量。
PR72 原生 strong 且 stake_multiplier 等於 1 SYNTHETIC 才映射為 standard；weak/light 計入數為 0 SYNTHETIC。
所有組別均套用同一額外離線兩敗永久停手機制，不放寬正式原有的任何 Skip。

## 離線擬合（截距、基線 logit 係數、檢索 logit 係數）

ungated: -0.21168739 SYNTHETIC, 3.7504859 SYNTHETIC, 0 SYNTHETIC
gated: -0.095177823 SYNTHETIC, 4.230431 SYNTHETIC, 0 SYNTHETIC
