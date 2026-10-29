# VNext 雙核心 Shadow

PR83不變。Core A：既有213→48 Physics獨立推論＋逐局條件PF；Core B：原PR83牌路候選獨立推論，5/10/20/40窗口只診斷。兩者透過獨立fusion輸出B/P/T，再做prediction/direction/entry分離。config.enabled=false、shadow_only=true、promotion_approved=false，無正式runtime hook，原UI/auth完全不改。

PF支援BPT、可選最終點數/張數及完整player_cards/banker_cards（rank1～13），驗證標準第三張規則。完整已知牌只記錄實際已觀測消耗；未觀測燒牌/初始剩餘牌仍未知，remaining_cards_exact=false。部分觀測用有限提案的近似條件後驗；若全粒子退化直接報錯，不捏造可用後驗。append重用posterior，撤銷/跨靴重建；forecast用獨立確定亂數，不改更新state。

Fusion固定50/50僅為對照；動態road_weight在train選，temperature在validation選，test不調參。凸混合不把兩份BPT當獨立似然相乘，保留相關性警示。沒有真人驗證，所以只影子執行。可選XGB/KNN不新增：缺少獨立增益證據，不以模型數量代替能力。

新Shadow decision不使用出手配額，只有正EV且通過uncertainty/quality/confidence才入場；獨立16筆selective guard不給relief。原PR83 decision/Volume Guard完全保留。CoreA內部PF採既有bounded quality .06～.16邏輯，不宣稱已校準最佳。手機環境不可用，僅桌面benchmark。

此交付是可執行研究骨架及小樣本回放，尚缺：大量真實牌靴、正式機率校準/融合外推證據、手機瀏覽器測試、足量Standard出手；不能晉升。完整PR83輸出未替換。
