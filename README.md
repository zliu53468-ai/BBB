# BGS 32D AI 科技感測試面板

這一版的目標不是重做預測邏輯，而是把原本 32D 測試面板的
「沿用32D本地腦直接預測」操作方式換成科技感按鈕介面。

## 操作方式

1. 用「莊 / 閒 / 和」按鈕輸入歷史。
   - 每按一次，畫面直接增加一筆 B / P / T。
   - 這一步只修改歷史資料。
   - 不預測、不 Walk-forward、不更新 A/b。

2. 按「開始分析」。
   - 等同原測試面板的「沿用32D本地腦直接預測」。
   - 讀取目前32D本地腦。
   - 用目前完整 B/P/T 歷史計算最新 32D Context。
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
   - 不修改32D本地腦。
   - 要重新預測就再按「開始分析」。

5. 「結束分析」
   - 只清掉目前顯示的預測結果。
   - 歷史保留。
   - 本地32D腦也保留。

## 32D

16維牌靴 + 16維路單。

這個按鈕版沒有 A～10/J/Q/K 實際殘牌輸入，所以沒有 exact composition 時：
- rank ratios = neutral 1.0
- physical edge = 0
- shoe reliability = 0

其餘 32D Context、two-arm LinUCB、UCB direction、機率映射均在瀏覽器本地執行。

## GitHub Pages

1. 解壓 ZIP。
2. 將解壓後的所有內容上傳到 GitHub repository 根目錄。
3. Repository → Settings → Pages → Source 選 GitHub Actions。
4. 本包已附 `.github/workflows/pages.yml`，push 到 main 後會自動部署。

## 注意

這是測試工具，不代表任何保證勝率。
