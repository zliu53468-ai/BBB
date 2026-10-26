# 補牌完成後 EV 一致化 v1

查核基準：BBB main e11b52424d0641aeaa16437a15e744808848dfba。日期：2026-10-26。
本次是EV結算口徑修正與診斷，不是重新訓練、勝率升級或新增下注策略。PR #57的每局獨立推論未併入；沿用main原平滑與Guard結構。

## 結算與佣金

抽牌/補牌先依現有標準規則完成，取得整局結束後的無條件機率pB+pP+pT=1，再計算每下注1單位的淨期望：
- EV_B=(1-banker_commission)*pB-pP
- EV_P=(1-player_commission)*pP-pB
- 和局退回本金，淨損益為0；佣金只扣在該方獲勝時。

cardcount_profile.json新增settlement（schema_version=1、mode=post_draw_unconditional_v1）。預設**沿用原程式莊勝5%、閒勝0%佣金**，不是已確認使用者桌台費率。可改banker_commission/player_commission（0～1），需配合實際桌台。只支援B/P平賠扣比例佣金、和局退回；免佣莊6半賠、特定牌型不賠、固定手續費、和局投注等不能僅把佣金改0冒充已支援。

原profile的enabled旗標不會關閉此結算修正；回滾需明確設定settlement.mode=legacy_conditional。設定未載入時亦預設採5%莊佣金的補牌後EV。

模型/畫面提供的條件莊機率qB排除和局，不能直接當無條件pB。最終Guard用pB=(1-pT)*qB、pP=(1-pT)*(1-qB)，只補回一次和局質量。
例：qB=.51、qP=.49、pT=.10時，pB=.459、pP=.441，EV_B=-.00495；顯示差達2pp不表示通過EV。

## 修正範圍與未變更項

1. Physics candidate使用該階段已校準/已套Macro後的條件機率及相同pT計算EV。最終Guard使用Clip/EMA後的條件機率及相同pT計算。不同階段機率改變時EV可不同，結算公式/單位始終相同。
2. PF仍只在原位置混合一次。診斷列出Direct Physics、PF原後驗、混合、candidate、final各自的機率/EV，不把幾個EV相加或二次混合。
3. 歷史Volume Guard的EV代理亦使用完整B/P/T與相同佣金；舊快照若沒有可信pT或48D第25槽，就不納入這項放寬統計，不把未知和局當0。
4. 保留EV_CONSISTENCY_V1=false；沒有啟用舊shadow失敗方案附帶的前段tie penalty。此次修正另由settlement.mode控制，避免夾帶額外懲罰或門檻調參。
5. 213→64→32→48、ReLU、Core兩檔、PF核心、57D順序與定義、XGB P=0/B=1和模型權重全部不變。EV heads作為原5%佣金訓練特徵保留；實際出手EV由完整機率重新結算，不能把EV頭再加上PF的EV。
6. 新增settlement_audit與draw_report，保存於事前預測快照及訓練匯出，沒有加入57D。auth-client.js只更新runtime快取版本，開通流程不變。BGS0.3V.1與Render未修改。

## 四/五/六張報告

draw_report分別列Direct Physics、PF posterior、mixed的4_cards/5_cards/6_cards與expected_cards，來源皆是完成補牌後的結果分布。工具不再抽牌、不改亂數、不改PF權重。
只有提供實際observed_hand.card_count（4/5/6）才計算實際比例及多類Brier；必須以該局開牌前的預測配對同局實測張數。只有B/P/T時標記not_evaluable_without_measured_card_count，不反推張數，也不宣稱校準正常。
報告分段包含1–20/21–45/46–70、≤40/41–50/>50及含50的50–70；從完整時間線切片，沒有重跑各段Guard。

可直接讀取網頁匯出的訓練JSON：
python tools/post_draw_report.py bgs_final57_training.json post_draw_report.json

也可重現本次純機制檢查（不能當作真實holdout）：
python generate_macro_smoke.py /tmp/post_draw_hands.json --shoes 4 --seed 20261026
node tools/top_parameters/replay.js . /tmp/post_draw_hands.json /tmp/post_draw_rows.json
python tools/post_draw_report.py /tmp/post_draw_rows.json /tmp/post_draw_report.json

Windows請將/tmp路徑換成工作目錄的實際檔案路徑。

## 本次結果

validation/post_draw_ev_smoke.json保存4靴合成、274次可分析預測的完整診斷。首手依現有UI排除；合成牌靴長度未必剛好70。
- 所有274筆EV公式誤差為0，三個來源張數分布均有效、和為1。
- 原版/新版的最終條件機率逐值相同；此樣本正式出手皆83次，沒有方向差異。hit=44.16%、EV/bet=-.11988、Skip=69.71%：兩版相同，而且本樣本EV為負，**沒有收益改善宣稱**。
- 混合預測平均四/五/六張=35.82%/31.70%/32.49%；本合成樣本實際=36.13%/28.10%/35.77%。多類Brier=.67141。這只能顯示本小樣本偏差，不能證明真實牌局已校準。
- 16組機率/和局極端值、JS/Python公式、2%/1%自訂佣金、110組補牌規則、4/5/6張與天牌案例通過；全部既有方向/按鈕/回退/持久化回歸亦檢查。模型與PF核心hash鎖定測試通過。

已完成局的逐注實現EV與預測EV分開報告。預測EV為正不等於真實收益為正。沒有真實獨立holdout，不能宣稱此修正提高命中率、EV或降低Skip；修正單位後某些門檻附近的出手可能減少。

## 回滾

將settlement.mode設為legacy_conditional可比較舊條件機率EV口徑，這只供回歸/回滾，不是建議繼續把和局質量忽略。完整回滾用git revert此次合併提交，回復runtime/profile/cache版本，保留真實歷史與授權資料。
不要開EV_CONSISTENCY_V1或調整佣金去刻意通過品質門檻。重訓/晉升新權重仍須另有凍結production與真實chronological whole-shoe holdout；舊重訓工具的歷史基準不冒充此新版成績。
