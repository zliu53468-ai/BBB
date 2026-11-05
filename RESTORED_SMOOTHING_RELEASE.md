# 恢復平滑、保留中立偏移與單次5%抽佣

依使用者確認：保留中立偏移選邊，只恢復平滑；原始EV繼續含抽水5%。
基準：c5e2caf243e2a26fd58338a1ea66819d05ea6719。
版本：BBB_PF_EOR_DEPLETE_V4_1_NEUTRAL_SMOOTHED_RANDOM。

本來就有 EV_B=.95*P(B)-P(P)、EV_P=P(P)-P(B)，沒有新增第二次抽佣。
恢復呼叫既有 physics.adaptiveScoreSmoother(raw,pf)，不改該底層函式。
PF96、MC256、EOR24×96、Deplete32768、.70雜訊倍率與全部品質數值不變。
每次安全新種子、分離抽樣流、種子記錄與返回重播亦保留。

平滑為 baseline+blend*(raw-baseline)，blend由既有品質／誤差函式自適應決定，
範圍.24～.82。不是強制50/50，不依過去輸贏調方向，也不累積殘牌。
同一blend作用於B/P/T，故中立EV偏移是原中立偏移乘blend，沒有反轉候選方向。

保留raw_simulated_probability、original_pf_ev及raw_quality。正式simulated_probability
與EV為平滑後值，UI明確標示平滑EV、原始含抽佣EV與保留比例。品質門檻仍用原始
估計標準誤，不把資料相依平滑當成新證據或校準的信心增益。

Deplete仍是0%融合。原始Deplete報告保留於deplete_raw；在deplete_mc增加
comparison_ev_banker/player，使用同一PF blend與同一基準做比較；原始MC抽樣
與EV不覆寫。V4同時使用平滑後PF與平滑後Deplete的中立偏移，不跨尺度核對。
原始Deplete誤差亦保持保守值。EOR仍以原始敏感度作保守脆弱性檢查，不新增投票。

固定seed=0的完整100組功能覆蓋：平滑前24莊/10閒/66觀望；恢復後14莊/9閒/77觀望。
候選符號不因正比例平滑而改變；品質通過數減少是偏移縮小，不調門檻補訊號。
正式每次新抽樣，這張表不是實際方向比例或命中率，也不是100局使用上限。

回歸測試鎖定原始機率／EV／品質／EOR／Deplete fixtures不變，另驗證：
平滑公式、單次抽佣、偏移縮放、SE不縮小、MC同尺度、重播及授權／Chrome操作。
沒有真實樣本外命中率改善證據，不宣稱恢復平滑可保證提高勝率。
