# Quant256 Online Anti-Lock V2（2026-11-01）

## 正式入口

原 BBB 授權登入後即啟用 \`quant256_online_runtime.js\` + \`quant256_online_panel.js\` V2。不修改授權資料、Cloudflare Worker、Frozen Core 或原本結果紀錄。舊版對照可使用 \`?quant256=frozen\`。

## 真正的雙臂對稱更新

對已結算的每個非和局，使用事先建立的 256D context（前 128D 進程/虛擬物理、後 128D 牌路），在 36 個啟用特徵上以完整 36×36 矩陣更新。兩臂 **都** 先執行：

\`A_arm ← lambda * A_arm + (1 - lambda) * I\`；
\`b_arm ← lambda * b_arm\`；
\`A_arm ← A_arm + xxᵀ\`。

- 開莊 B：\`b_B ← b_B + x\`；\`b_P ← b_P - x\`。
- 開閒 P：\`b_B ← b_B - x\`；\`b_P ← b_P + x\`。
- 開和 T：僅衰減兩臂，不虛構 B/P 標籤。
- \`lambda=0.94\`，合法範圍 \`0.90~0.96\`。理論上兩臂 A 相同、b 相反；這是全回饋的上下對稱，而非只更新建議邊。

## 去中心化與歸一化

移除模型輸入內的正截距、累積局數、絕對虛擬殘牌數、永久增長的 \`run/rounds\` 計數。啟用的 36 個欄位都以 \`[-1,1]\` 的對稱相對差值、3/8/16 局路單差分、entropy 震盪、conditional virtual-hand simulator 的短窗變動率構成。其餘 220 格為 0，用來保留 256D 介面相容性。

**僅有 B/P/T 不知道真實 4、5、6 張發牌情況**；虛擬物理訊號仍屬不可校準的條件模擬，不可當成剩餘卡牌實際讀值。

## Anti-Lock 過熱冷卻

- 每次呼叫 \`predict()\` 只是讀取，不會修改累計 streak。
- 每局開獎 \`observe(B/P/T)\` 時，才將上一局**有實際顯示莊/閒方向**的推薦計入同方向 streak；觀望則重設連續推薦紀錄。和局會結算該局之前已展示的建議，不假定和局是哪側獲勝。
- 連續同側推薦第 **4 次**起，對另一側的 **alpha** 做 \`alpha_opposite = 0.15 × min(2.5, 1 + 0.7 × (proposedStreak - 3))\` 的動態上調。基本兩臂仍沿用對稱 A/b。
- 如因探索上調而把較弱臂硬推成較強臂，**不直接翻轉下注方向，而是 Skip**。
- 已經連續顯示 4 次相同方向時，下一次若模型仍想顯示同側，**強制觀望**（\`anti_lock_skip\`）；避免無上限的單側提示，且不製造人為的反向注。
- 原有前 12 局暖機與 \`0.025\` 波動自適應品質門檻繼續生效。

### 觀察面板

面板會顯示「反鎖定 streak」、「冷卻對側」、莊／閒各自的 alpha，以及 \`anti_lock_skip\`、\`cooling_reversal_skip\` 等明確原因。

## 驗證方式與意義

執行 \`node test_quant256_online.js\`；GitHub Actions：\`.github/workflows/quant256-online.yml\`。測試包含完整 A/b 對稱、衰減、去中心化上下界、無數據截距、反側 alpha、四次上限、無強迫逆向、刷新重播、返回上一局、瀏覽器按鈕事件及舊版退路。

**不可過度解讀**：Anti-Lock 保證的是介面策略不會無限制地一直提示同側，不保證隨機過程能被預測，也不保證 Hit Rate 提高。提高觀望比例可能令樣本選擇偏差；需要逐靴 prequential replay 與獨立樣本比較、同時報告出手比例及命中率，才能評估提升是否成立。
