# BGS 128D（64D＋64D）AI 科技感測試面板

這一版是在 `f6c24d3` 的 64D Frozen Direct 基礎上，將 BBB 網頁測試面板擴充為 128D：

- 64 維牌靴／進程特徵
- 64 維路單／結構特徵
- 合併為單一 128 維 Context，交給 two-arm LinUCB 計算 B / P 方向

操作流程維持 Frozen Direct：只在按下「開始分析」時直接預測，
不做 Walk-forward、不自動回饋、不 bootstrap、不 replay、不 decay，也不更新 A/b。

## 操作方式

1. 用「莊 / 閒 / 和」按鈕輸入歷史。
   - 每按一次，畫面增加一筆 B / P / T。
   - 這一步只修改歷史資料。
   - 不預測、不 Walk-forward、不更新 A/b。

2. 按「開始分析」。
   - 沿用目前 128D 本地腦直接預測。
   - 用目前完整 B/P/T 歷史計算最新 128D Context。
   - 直接做 LinUCB B/P 預測。
   - 不做 Walk-forward replay。
   - 不結算上一筆 prediction。
   - 不更新 A/b。
   - 不 decay。

3. 下一局開出後：
   - 再按一次「莊 / 閒 / 和」把新結果加入歷史。
   - 再按「開始分析」預測下一局。

4. 「返回上一局」
   - 移除歷史最後一筆。
   - 不修改 128D 本地腦。

5. 「結束分析」
   - 只清掉目前顯示的預測結果。
   - 歷史與本地 128D brain 保留。

## 128D 結構

### 64D 牌靴／進程

保留原本 32D 牌靴進程特徵，再增加 32 個多尺度訊號，包括：

- 4 / 6 / 12 / 24 / 32 局和局比例
- 4 / 6 / 8 / 16 / 24 / 32 局 B/P entropy
- 6 / 8 / 16 / 32 局 B/P/T entropy
- 多窗口 B/P balance
- penetration / remaining 的平方與平方根轉換
- 更細的牌靴階段 basis
- 8 / 16 / 24 / 48 局樣本支撐

按鈕版仍沒有 A～10/J/Q/K 的實際殘牌輸入，因此牌值比例維持 neutral 1.0，
physical edge 與 exact composition reliability 維持 0，不虛構實際殘牌資訊。

### 64D 路單／結構

保留原本 32D 路單特徵，再增加 32 個多尺度訊號，包括：

- 2 / 4 / 6 / 10 / 16 / 24 / 32 / 48 局 Banker ratio
- 同窗口 Turn rate
- 大眼仔 / 小路 / 蟑螂路的 4 局與 16 局 regularity
- 更長的前序 run length
- 最近 run 平均、高度最大值、標準差與變化量
- 4 / 6 局交替結構
- 4 / 5 局同邊結構

## 舊資料相容性

首次載入 128D 版本時，如果瀏覽器只有舊 64D 或 32D localStorage：

- 保留 B / P / T 歷史紀錄
- 不沿用尺寸不相容的舊 A 矩陣與 b 向量
- 自動建立新的 128×128 A 矩陣與 128 維 b 向量
- 舊 64D 版本已保存在分支 `backup-64d-f6c24d3-before-128d`

## 測試定位

這次目的是單純比較 64D 與 128D 的特徵容量差異，因此 Frozen Direct 行為維持不變。
維度增加本身不代表勝率一定提高；應使用相同牌局資料比較兩版方向、穩定性、反轉敏感度與實際命中結果。

## GitHub Pages

Repository → Settings → Pages 使用既有 GitHub Actions 部署即可。

## 注意

這是測試工具，不代表任何保證勝率。