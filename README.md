# BGS 64D（32D＋32D）AI 科技感測試面板

這一版將 BBB 網頁測試面板從 32D 擴充為 64D：

- 32 維牌靴／進程特徵
- 32 維路單／結構特徵
- 合併為單一 64 維 Context，交給 two-arm LinUCB 計算 B / P 方向

操作流程仍維持 Frozen Direct：只在按下「開始分析」時直接預測，
不做 Walk-forward、不自動回饋更新 A/b，也不連線到 LINE 版本。

## 操作方式

1. 用「莊 / 閒 / 和」按鈕輸入歷史。
   - 每按一次，畫面直接增加一筆 B / P / T。
   - 這一步只修改歷史資料。
   - 不預測、不 Walk-forward、不更新 A/b。

2. 按「開始分析」。
   - 沿用目前64D本地腦直接預測。
   - 讀取目前64D本地腦。
   - 用目前完整 B/P/T 歷史計算最新 64D Context。
   - 直接做 LinUCB B/P 預測。
   - 不做 Walk-forward replay。
   - 不重置本地腦。
   - 不結算上一筆 prediction。
   - 不更新 A/b。
   - 不 decay。

3. 下一局開出後：
   - 再按一次「莊 / 閒 / 和」把新結果加進歷史。
   - 再按「開始分析」預測下一局。

4. 「返回上一局」
   - 移除歷史最後一筆。
   - 不修改64D本地腦。
   - 要重新預測就再按「開始分析」。

5. 「結束分析」
   - 只清掉目前顯示的預測結果。
   - 歷史保留。
   - 本地64D腦也保留。

## 64D 結構

32維牌靴進程 + 32維路單結構。

由於按鈕版沒有 A～10/J/Q/K 的實際殘牌輸入，牌值比例仍使用 neutral 1.0，
physical edge 與 exact composition reliability 仍為 0；新增維度使用牌靴進度、階段、
樣本支撐、和局比例、資訊熵，以及更長短窗的路單結構訊號。

64D Context、two-arm LinUCB、UCB direction、機率映射均在瀏覽器本地執行。

## 32D 舊資料相容性

首次載入 64D 版本時，若瀏覽器只有舊 32D localStorage：

- 保留 B / P / T 歷史紀錄
- 不沿用尺寸不相容的 32×32 A 矩陣與 b 向量
- 自動建立新的 64×64 A 矩陣與 64 維 b 向量

## GitHub Pages

1. 解壓 ZIP。
2. 將解壓後的所有內容上傳到 GitHub repository 根目錄。
3. Repository → Settings → Pages → Source 選 GitHub Actions。
4. 本包已附 `.github/workflows/pages.yml`，push 到 main 後會自動部署。

## 注意

這是測試工具，不代表任何保證勝率。
