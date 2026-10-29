# 五階上下文與五級協同可靠度測試版

保持原架構與同靴因果推論；馬可夫上限改5階，依5→4→3→2→1檢查，每階仍要求12筆已公布的上下文轉移。全部不足仍用既有Physics+PF暫代。貝葉斯對選定階統計做既有樣本數收縮，不因階高而增加信心。

後段不是各跑一個五階序列模型：共用原可靠度r=min(1-uncertainty,1-noise,quality,ESS)，五級節點(raw→mapped)為0→0、.25→.15、.5→.35、.75→.60、1→1；節點間線性插值。任意映射值都不超過原r，沒有信心上調。nearest_level為顯示用標籤，不參與跳變計算；lower_level/upper_level/fraction可追蹤插值。

Physics校正、Macro delta、EMA最新權重及EV門檻加成都使用同一mapped reliability，EV本身仍由最終機率與和局質量/佣金重算，不重複收縮。既有Clip、Volume Guard、0.5pp、PF核心抽樣與模型權重不動。無前中後局數分段。

設定road_primary_v1.order=5，five_level_v1.enabled=true，reliability_anchors為上述五點；節點x須恰為0/.25/.5/.75/1，y單調且0<=y<=x，無效則用保守預設。ESS不是命中信心的證明。

診斷保留backoff各階支持數、bayesian.selected_order與adaptive.five_level全部欄位，沿既有快照匯出。50～70局內四/五階可能常因支持不足回退，未降低樣本門檻強迫五階生效。

回滾five_level_v1.enabled=false：上限自動回3，後段可靠度回PR81原連續值；其餘策略tag版本僅為狀態隔離。或revert本次合併提交。原功能設定仍保留。

node test_five_level.js 驗證全部5階回退、12樣本門檻、五級連續單調及不放大、缺測、因果隔離、PR81數值回滾、EV口徑與匯出。既有核心/授權/結束分析測試另保留。沒有獨立真實牌靴證明勝率提升，五階不代表比三階準。
