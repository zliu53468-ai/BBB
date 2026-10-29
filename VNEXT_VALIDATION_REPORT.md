# VNext驗證報告

**SYNTHETIC_ONLY / INSUFFICIENT_SAMPLE / 不准晉升。** 真實資料0靴。7模擬靴：3 Train/2 Validation/2 Test；Test140機會、126非和。正式PR83比較使用原完整PF10000及原EMA/EV/Volume Guard，不以代理代替。新核心PF500與原PF並非相同估計器；新decision選擇性閘門與原decision不同，表格不證明真實命中率提升。

| 模式 | 全非和命中 | Standard次數 | Standard命中 | Brier | Logloss | ROI/下注 | 每機會報酬 | 最大連錯 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PR83 | 0.515873 | 48 | 0.523810 | 0.585668 | 0.946132 | 0.030208 | 0.010357 | 4 |
| Physics | 0.500000 | 37 | 0.566667 | 0.578741 | 0.938379 | 0.091892 | 0.024286 | 4 |
| Road | 0.484127 | 63 | 0.500000 | 0.590018 | 0.961607 | -0.007143 | -0.003214 | 4 |
| FixedFusion | 0.515873 | 66 | 0.508475 | 0.583474 | 0.948501 | 0.005303 | 0.002500 | 5 |
| DynamicFusion | 0.484127 | 73 | 0.500000 | 0.584440 | 0.950442 | -0.010959 | -0.005714 | 5 |
| Theory | 0.484127 | 0 | 無分母（0次下注） | 0.585464 | 0.949524 | 0.000000 | 0.000000 | 0 |

每組樣本不足100次Standard均標INSUFFICIENT_SAMPLE；0次下注的條件命中率無分母，其他可算數值仍完整提供。詳見vnext/validation_report.json：包含B/P方向、出手率、1–20/21–45/46–70、長龍/單跳/雙跳、逐靴bootstrap95%CI與相對PR83差異。僅2靴Test，CI不能當可信改善證據。vnext/replay_trace.json保留每策略事前p/方向/entry與事後actual。

Fusion權重只依Train選，temperature只依Validation選，Test不調參；candidate的promotion_approved=false。數據沒有證明雙核心優於PR83，禁止替換正式。需要1000真實Standard與30真實靴後另行統計審查。

已通過標準補牌、partial/full觀測條件、機率和、EV/退和、跨靴/撤銷/重播、PR83原模型/授權/結束分析測試。桌面performance與500/2000/10000單觀測運算比較見particle_benchmark.json；手機瀏覽器尚未驗證。

未完成大規模實戰驗證、多人真實牌靴校準、嚴格跨裝置效能與大量粒子精度統計；時間限制內交付的是端到端可跑通Shadow研究版，而非經證實勝率升級。
