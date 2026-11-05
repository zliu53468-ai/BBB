# 小幅降低觀望：EV差距底線降10%

基準：eacb21aba910c862bd0890e4110cba579062ed6f。
只將minEvGap由.005調為.0045，noiseGuardMultiplier保留.70。

```js
threshold = Math.max(.0045, .70 * estimator_gap_se);
```

不是取消品質檢查或強制出手。80%批次方向一致性、至少3批／最多5批、精確
先驗核對max(.005,2.5*SE)、平滑、中立方向、每次新種子、莊5%佣金均不變。
PF96／MC256、EOR24×96、Deplete每批32768且0%融合與Worker執行方式不變。

原minEvGap同時用作Deplete反向訊號判定的底線。為避免連動改動，新增
crossCheckDirectionFloor=.005，讓Deplete原始條件維持完全相同的數值：

```js
abs(mcSigned) > max(.005, .70 * pairedMCSE)
```

crossCheckDirectionFloor亦有型別／範圍檢查，單元測試驗證放寬主訊號底線
不會連動改變Deplete反向判定。其他V4參數、原始機率及EV計算沒有更改。

## 檢查結果

固定seed=0的100種完整輸入覆蓋：原版11莊／7閒／82觀望；調整後13莊／8閒／79觀望。
9種使用3批、91種使用5批；平均估計SE仍為.006108。沒有以不穩定方向或
精確不相容的訊號補出手數。固定樣本比較亦包含.004與.0035，但選最小10%
放寬，而非挑出手最多或宣稱命中率最佳的值。

100種是可能輸入的功能檢查，不是100局真實回測、正式出手比例或勝率。
正式每次新抽樣，實際結果會變動；INSUFFICIENT_REAL_HOLDOUT_DATA。
降低觀望只代表接受較弱偏移，不能據此宣稱勝率或扣佣收益提高。

回歸：原單批核心、精確重播與原始數值fixtures、多批一致性、參考否決、Worker
錯誤／取消、會員隔離及結束分析在本機通過。CI持續執行真Chrome桌面／觸控
降速、事件迴圈回應、返回種子重播與完整網格。基準舊路徑既有CI失敗不在此改動。

授權、index及Worker／面板只同步資產快取版本；未改Cloudflare、會員機制、
原介面、BGS或LINE。
