# V4.1：恢復含莊佣金的原始EV選邊

基準：c5e2caf243e2a26fd58338a1ea66819d05ea6719。
版本：BBB_PF_EOR_DEPLETE_V4_1_COMMISSION_FRESH_RANDOM。

原始機率與EV本來就已使用莊贏抽水5%。本次不再次扣抽水，只取消中立基準
校正，改回直接比較原始含佣金EV：

```js
EV_B = 0.95 * P_B - P_P;
EV_P = P_P - P_B;
decisionGap = EV_B - EV_P;
candidate = EV_B >= EV_P ? 'B' : 'P';
```

Deplete也用自己的原始EV差交叉核對，不再減BASE_GAP。V4用abs(decisionGap)
與原門檻比較；原始EV、抽樣誤差、EOR敏感度計算都不改，沒有機率平滑或配額。
BASE_EV/BASE_GAP匯出保留作相容診斷，不再改變正式方向。

不變：總點輸入0～9、每局獨立、每次新Web Crypto種子、分開PF/EOR/Deplete流、
種子記錄與返回重播、PF96、MC256、EOR24×96、Deplete32768與融合0%、
noiseGuardMultiplier=.70、minEvGap=.005、minPosteriorQuality=.50、
eorFragilityFloor=.025、crossCheckZ=2.5、crossCheckFloor=.008。
會員授權、裝置綁定、LINE/BGS、底層quant/physics/deplete原檔都不修改。

畫面由中立偏移改為含抽水EV差距與候選，移除中立基準校正數值，不重設任何數值參數。
可能再次出現偏莊分布，因莊／閒基準EV本來不相同；不宣稱出手各半或勝率提高。
同樣種子可還原數值抽樣，返回舊紀錄仍以目前版本規則解讀原始EV。

回歸：100種輸入用明確seed=0作功能覆蓋，原始機率／EV／品質／EOR／Deplete
fixtures維持一致；另驗證新種子變化、精確重播及無熵失敗。
佣金單項測試包含P_B=.505/P_P=.495時選P（即使P_B較高），確保佣金只扣一次。
核心、授權、結束分析與Chrome PC／手機操作及返回重播需成功才合併。
INSUFFICIENT_REAL_HOLDOUT_DATA：不以功能網格宣稱實際勝率改善。
