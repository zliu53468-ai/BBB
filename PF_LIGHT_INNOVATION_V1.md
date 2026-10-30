# PF 輕量介入：粒子後驗殘差

執行鏈：Core＋Physics → 貝葉斯／馬可夫1–3階 → 原Physics有界校正 → PF後驗殘差 → 原自適應XGB → 原Macro／Clip／EMA → 原EV／Volume Guard。

沿用particle_filter_runtime.js既有10000粒子、抽補牌與和局軟觀測更新核心，沒有重訓或修改此檔。每個新同靴前綴重新建立粒子並重播該前綴；相同靴、完全相同歷史／模型／PF配置的重複分析，僅重用最後一份後驗預測快照以減少重算。沒有保存或沿用粒子陣列、權重或重採樣狀態，不跨桌／靴／瀏覽器；返回另一前綴會重建。

PF以uncertainty_only模式取得原始posterior；legacy混合權重設為0，且不使用其corrected physics。Direct Physics48D／XGB57D保持原值。q_pf=posterior[23]/(posterior[23]+posterior[24])；q_phys=direct[23]/(direct[23]+direct[24])。只加入innovation=q_pf−q_phys，避免再把整份Physics機率混入一次；這不是假設PF與牌路資料獨立。

bayes_markov_final_v1.pf_light_v1：max_weight=0.06（可調，硬上限0.08）、max_delta=0.0025（硬上限），quality_low/high=0.55/0.80、ess_low/high=0.40/0.80、uncertainty_low/high=0.55/0.80。

weight=0.06×quality_gate×ess_gate×uncertainty_gate，delta=clamp(weight×innovation,−0.0025,+0.0025)。品質／ESS不足、高不確定性、缺模組或PF失敗時delta=0，保留原機率，不因沒有PF強制Skip。可向莊或閒修正，不設僅同向規則，不靠亂數改種子製造換邊。相同輸入方向／機率一致。

PF和局機率、點數、4/5/6張分布另存診斷；本層只校正非和莊機率，正式pT仍由原Physics提供，不混改佣金、和局退回或補牌規則。PF模擬牌階是條件假說，不是真實桌上已知剩餘牌。原T上下文／EMA與和局規則不變；新PF自然會使用同靴已公布的T，沒有新增和局必翻邊。

XGB配對比較的base改為Physics＋PF校正後機率。配置／模型／PF版本tag隔離舊EMA與XGB評分，需重新累積至少12筆已登記且結算的非和預測。所有既有權重、模型、貝葉斯參數、Physics校正參數、XGB參數、平滑、EV/VG、1pp畫面門檻、開通碼與30分鐘設定保留。

匯出bayes_markov.pf_light含base_pb、physics_pb、posterior_pb、innovation、三個gate、實際weight、delta與reason。compact diagnostics保存實際粒子數、品質、ESS、不確定性、抽樣次數、和局觀測數、點數／張數／winner分布。finalProbability.particleDiagnostics與既有particle_physics快照同時接入，不輸出粒子陣列。

回滾：pf_light_v1.enabled=false精確回PR97，不刪其他模組。測試node test_pf_light.js，包含真PF、重複／撤銷、故障回退、特徵與模型不變、和局／補牌原碼保護與匯出。10000粒子長前綴重播會增加瀏覽器運算時間；測試紀錄只能證明功能和因果性，不保證提高命中率或減少觀望。
