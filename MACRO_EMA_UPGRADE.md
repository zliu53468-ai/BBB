# BBB 巨觀 EMA 與動態門檻升級

## 1. 架構相容性確認

基準 main：`6368b3e9cfea19e5120a2e6610606f1d6e213824`。Runtime：`PHYSICS_57D_V14_8_MACRO_EMA_GUARD`。

保持順序：B/P/T → Direct Physics → 每次重建 Particle500 → Physical EV／candidate → Clip + EMA → Frozen Core + 57D XGBoost 輔助篩選 → Final EV Guard → Volume Guard → B/P/Skip。

- `app256forward.js`、`app256continuation.js`、`particle_filter_runtime.js`、`physics_multitask_model.json` 與基準逐位元相同。
- MLP 維持 213→64→32→48，輸入 213D、輸出 48D；沒有重訓。
- `final_probability_model.json` 只改 `decision_policy`。Trees、base_margin、feature_names、校準、原本 EV thresholds 均未修改。
- XGBoost 維持原 57D；EMA 沒有插入其特徵向量，沒有改樹深度。它只能保留／阻擋 Physics candidate，不能翻向或將 Physics Skip 變為出手。
- 新資料由選填的「總張數、閒點數、莊點數」取得；B/P/T-only 歷史不被反推為實際張數或點數。和局本身可以確定 Low_Score_Win=0。
- 模型未就緒或 Particle500 不可用時輸出 Skip，避免異常路徑由 Core 接管方向。
- 全部 runtime 為瀏覽器原生 JS；Python 只在離線特徵工程／測試使用。新增 JS 在 HTML 中先於 final runtime 載入。
- runtime 更新改走唯讀、3 分鐘上限的檢查 workflow；從原完整重訓 workflow 的 path triggers 移除 final runtime、其測試與 workflow 自身。訓練原始碼變更與手動 workflow_dispatch 仍可觸發原重訓流程。

**目前 EMA 方向係數是 0，結構偏移尚未啟用；动态 guard 已啟用。** 這三個無符號特徵不是精確算牌，尚無足夠資料支持固定「正 spread → 莊／閒」映射。接口完整可用，但不能把任意係數當成校準結果。

## 2. Python EMA 特徵工程完整程式碼

檔案：[`macro_ema_features.py`](macro_ema_features.py)，僅使用 Python 標準庫。輸入每靴全部完成局，必須包含和局、未分析局；不可把稀疏的 prediction log 當成完整開牌資料。

網頁「匯出開牌資料」可直接作為輸入：

```bash
python macro_ema_features.py bgs_observed_hands.json macro_features.json --estimated-total-hands 60
```

每列輸出先計算截至前一局的特徵，再用當列結果更新狀態，避免用目標局點數預測目標局。`actual_b` 在和局為 null，分類訓練排除該標籤但保留它對後續 EMA 的更新。

```python
#!/usr/bin/env python3
"""Causal macro EMA features, Python standard library only.

python macro_ema_features.py completed_hands.json features.json
Input: ordered list (or {"rows": [...]}) of shoe_id, round_index, outcome,
card_count (4..6), player_score (0..9), banker_score (0..9).
Missing observations stay missing; result alone never supplies points/cards.
Output features are BEFORE the row's outcome, suitable for next-hand training.
"""
import argparse
import json
import math

NAMES = ("Six_Card", "Low_Score", "Point_Diff")


def integer(value, name, low, high):
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not str(value).strip().isascii() or not str(value).strip().isdigit():
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def normalize_hand(hand):
    outcome = str(hand.get("outcome", "")).upper()
    if outcome not in ("B", "P", "T"):
        raise ValueError("outcome must be B/P/T")
    cards = integer(hand.get("card_count"), "card_count", 4, 6)
    player = integer(hand.get("player_score"), "player_score", 0, 9)
    banker = integer(hand.get("banker_score"), "banker_score", 0, 9)
    if player is not None and banker is not None:
        actual = "B" if banker > player else "P" if player > banker else "T"
        if actual != outcome:
            raise ValueError("outcome disagrees with final scores")
    return dict(outcome=outcome, card_count=cards, player_score=player, banker_score=banker)


def raw_features(hand):
    h = normalize_hand(hand)
    winner = h["banker_score"] if h["outcome"] == "B" else h["player_score"]
    return {
        "Six_Card": None if h["card_count"] is None else int(h["card_count"] == 6),
        "Low_Score": 0 if h["outcome"] == "T" else None if winner is None else int(winner <= 3),
        "Point_Diff": None if h["player_score"] is None or h["banker_score"] is None else abs(h["player_score"] - h["banker_score"]),
    }


class MacroEMA:
    def __init__(self, shoe_id=""):
        self.shoe_id, self.rounds = str(shoe_id), 0
        self.ema = {name: dict(short=None, long=None, count=0, last_round=0) for name in NAMES}

    def snapshot(self):
        out = dict(shoe_id=self.shoe_id, rounds=self.rounds, counts={}, ages={})
        for key, e in self.ema.items():
            out[key + "_Short"], out[key + "_Long"] = e["short"], e["long"]
            out[key + "_Spread"] = e["short"] - e["long"] if e["count"] else None
            out["counts"][key] = e["count"]
            out["ages"][key] = self.rounds - e["last_round"] if e["count"] else None
        return out

    def update(self, hand):
        raw = raw_features(hand)  # Atomic validation, matches JS.
        self.rounds += 1
        for key, value in raw.items():
            if value is None:
                continue
            e = self.ema[key]
            e["short"] = .20 * value + .80 * e["short"] if e["count"] else value
            e["long"] = .05 * value + .95 * e["long"] if e["count"] else value
            e["count"] += 1
            e["last_round"] = self.rounds
        return self.snapshot()


def build_features(rows, estimated_total_hands=60):
    """Rows must contain every completed hand, including unpredicted hands/ties.

    Strict sequential indices prevent silently treating sparse prediction logs
    as a complete observation stream. Interleaved shoes maintain separate state.
    """
    total = float(estimated_total_hands)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("estimated_total_hands must be positive and finite")
    states, output = {}, []
    for row in rows:
        if row.get("shoe_id") is None:
            raise ValueError("shoe_id required")
        shoe = str(row["shoe_id"])
        state = states.setdefault(shoe, MacroEMA(shoe))
        index = integer(row.get("round_index"), "round_index", 1, 1000000)
        if index != state.rounds + 1:
            raise ValueError(f"{shoe}: expected consecutive round_index {state.rounds + 1}")
        observed = normalize_hand(row)
        progress = min(1., index / total)
        output.append({**row, **state.snapshot(), "progress": progress,
                       "sample_weight": .8 if progress < .3 else 1.25 if progress > .7 else 1.,
                       "actual_b": 1 if observed["outcome"] == "B" else 0 if observed["outcome"] == "P" else None})
        state.update(observed)  # Only after pre-hand features have been captured.
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--estimated-total-hands", type=float, default=60)
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    result = {"feature_timing": "before_current_outcome", "rows": build_features(rows, args.estimated_total_hands)}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
```

## 3. 瀏覽器端 JS EMA 狀態維護程式碼

檔案：[`macro_ema.js`](macro_ema.js)。第一筆有效觀測同時初始化 short、long，spread=0。此後 short=0.20×x+0.80×short，long=0.05×x+0.95×long。缺值只略過該特徵，絕不當作 0；另記有效筆數與距離最後觀測的局數。

`Low_Score` 為規格中的原始 `Low_Score_Win` 序列；公開 EMA 欄位名是 `Low_Score_Short/Long/Spread`。`Point_Diff` 原始尺度為 0–9，只在結構修正的線性組合除以 9。

```javascript
/* Observed, completed hands only. No ranks or points are inferred from B/P/T. */
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.__BGS_MACRO_EMA__=api;
})(typeof window!=="undefined"?window:globalThis,function(){
  "use strict";
  const NAMES=["Six_Card","Low_Score","Point_Diff"];
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
  function integer(value,name,lo,hi){
    if(value===null||value===undefined||value==="")return null;
    if(typeof value==="boolean"||typeof value==="object"||!/^\d+$/.test(String(value).trim()))throw new Error(name+" 必須為整數");
    const n=Number(value);
    if(!Number.isInteger(n)||n<lo||n>hi)throw new Error(name+" 超出範圍");
    return n;
  }
  function normalizeHand(hand){
    const outcome=String(hand?.outcome||"").toUpperCase();
    if(!["B","P","T"].includes(outcome))throw new Error("outcome 必須為 B/P/T");
    const card_count=integer(hand.card_count,"開獎張數",4,6);
    const player_score=integer(hand.player_score,"閒點數",0,9);
    const banker_score=integer(hand.banker_score,"莊點數",0,9);
    if(player_score!==null&&banker_score!==null){
      const actual=banker_score>player_score?"B":banker_score<player_score?"P":"T";
      if(actual!==outcome)throw new Error("點數與莊／閒／和結果不一致");
    }
    return {outcome,card_count,player_score,banker_score};
  }
  function createState(shoeId=""){
    return {shoe_id:String(shoeId),rounds:0,ema:Object.fromEntries(NAMES.map(k=>[k,{short:null,long:null,count:0,last_round:0}]))};
  }
  function values(hand){
    const h=normalizeHand(hand),winner=h.outcome==="B"?h.banker_score:h.player_score;
    return {
      Six_Card:h.card_count===null?null:Number(h.card_count===6),
      Low_Score:h.outcome==="T"?0:winner===null?null:Number(winner<=3),
      Point_Diff:h.player_score===null||h.banker_score===null?null:Math.abs(h.player_score-h.banker_score)
    };
  }
  function snapshot(state){
    const out={shoe_id:state.shoe_id,rounds:state.rounds,counts:{},ages:{}};
    for(const key of NAMES){
      const e=state.ema[key];
      out[key+"_Short"]=e.short;out[key+"_Long"]=e.long;
      out[key+"_Spread"]=e.count?e.short-e.long:null;
      out.counts[key]=e.count;out.ages[key]=e.count?state.rounds-e.last_round:null;
    }
    return out;
  }
  function update(state,hand){
    const raw=values(hand); // Validate everything before mutating any EMA.
    state.rounds++;
    for(const key of NAMES){
      const x=raw[key],e=state.ema[key];if(x===null)continue;
      e.short=e.count?.20*x+.80*e.short:x;
      e.long=e.count?.05*x+.95*e.long:x;
      e.count++;e.last_round=state.rounds;
    }
    return snapshot(state);
  }
  function noiseGate(noise,low=.60,high=.75){
    const n=typeof noise==="number"&&Number.isFinite(noise)?clamp(noise):1;
    return clamp((high-n)/Math.max(1e-9,high-low));
  }
  function adjustment(probabilityB,features,progress,noise,config={}){
    const p=clamp(probabilityB),result={input:p,value:p,delta:0,signal:0,ready:false,reason:"disabled"};
    if(config.enabled!==true)return result;
    const number=(key,fallback)=>typeof config[key]==="number"&&Number.isFinite(config[key])?config[key]:fallback;
    const minimum=Math.max(1,Math.floor(number("min_observations",12))),maxAge=Math.max(0,number("max_age",3));
    const ready=NAMES.every(k=>Number.isFinite(features?.[k+"_Spread"])&&features?.counts?.[k]>=minimum&&features?.ages?.[k]<=maxAge);
    if(!ready)return {...result,reason:"insufficient_observations"};
    const weights=config.coefficients||{};
    const beta=k=>typeof weights[k]==="number"&&Number.isFinite(weights[k])?clamp(weights[k],-10,10):0;
    const signal=beta("six_card")*features.Six_Card_Spread+beta("low_score")*features.Low_Score_Spread+beta("point_diff")*features.Point_Diff_Spread/9;
    const start=clamp(number("progress_start",.30),0,.99),power=clamp(number("progress_power",2),1,4);
    const weight=clamp((progress-start)/(1-start))**power,cap=clamp(number("max_delta",.03),0,.05);
    const gate=noiseGate(noise),delta=cap*Math.tanh(signal)*weight*gate;
    // B/P EV preference boundary: .95*p-(1-p) = (1-p)-p.
    const boundary=2/3.95,raw=clamp(p+delta);
    const value=p>=boundary?Math.max(boundary,raw):Math.min(boundary,raw);
    return {...result,value,delta:value-p,signal,ready:true,progress_weight:weight,noise_gate:gate,max_delta:cap,
      reason:signal===0?"zero_coefficients_or_signal":gate===0?"high_noise":"applied"};
  }
  return {NAMES,createState,normalizeHand,values,update,snapshot,noiseGate,adjustment};
});
```

對應 runtime 的狀態管理如下。一般新增局只更新 running EMA；撤銷／分支／重載時重播已完成的觀測。重複分析不更新 EMA。每靴狀態重置，完整觀測可另外存為歷史訓練資料；上限依整靴刪除最舊資料，合計约 15,000 局。

```javascript
const MACRO_KEY="bgs_macro_observations_v1",MACRO_ARCHIVE_KEY="bgs_macro_observation_shoes_v1";
let macroJournal=null,macroState=null;
function readMacroArchive(){try{const value=JSON.parse(localStorage.getItem(MACRO_ARCHIVE_KEY)||"[]");return Array.isArray(value)?value:[];}catch(_){return[];}}
function rotateShoeId(){
  try{
    const current=macroJournal||JSON.parse(localStorage.getItem(MACRO_KEY)||"null");
    if(current?.events?.length){
      const archive=readMacroArchive().filter(shoe=>shoe.shoe_id!==current.shoe_id);archive.push(current);
      let count=archive.reduce((n,shoe)=>n+(shoe.events?.length||0),0);
      while(count>MAX_TRAINING_ROWS&&archive.length>1)count-=archive.shift().events.length;
      localStorage.setItem(MACRO_ARCHIVE_KEY,JSON.stringify(archive));
    }
  }catch(_){}
  // A full archive/quota must never prevent shoe isolation/reset.
  for(const key of [SHOE_KEY,PENDING_KEY,MACRO_KEY])try{localStorage.removeItem(key);}catch(_){}
  macroJournal=null;macroState=null;
}
function saveMacroJournal(){try{localStorage.setItem(MACRO_KEY,JSON.stringify(macroJournal));}catch(_){}}
function syncMacro(history){
  if(!MACRO)return null;
  const shoeId=getShoeId();
  if(!macroJournal||macroJournal.shoe_id!==shoeId){
    let saved=null;try{saved=JSON.parse(localStorage.getItem(MACRO_KEY)||"null");}catch(_){}
    macroJournal={shoe_id:shoeId,events:saved?.shoe_id===shoeId&&Array.isArray(saved.events)?saved.events.slice(0,500):[]};
    macroState=null;
  }
  const events=macroJournal.events;
  let common=0;
  while(common<Math.min(events.length,history.length)&&events[common]?.outcome===history[common])common++;
  if(!macroState||common<events.length){
    events.splice(common);macroState=MACRO.createState(shoeId);
    for(let i=0;i<events.length;i++){
      try{events[i]=MACRO.normalizeHand(events[i]);}catch(_){events[i]=MACRO.normalizeHand({outcome:history[i]});}
      MACRO.update(macroState,events[i]);
    }
  }
  for(let i=events.length;i<history.length;i++){
    const observed=MACRO.normalizeHand({outcome:history[i]});events.push(observed);MACRO.update(macroState,observed);
  }
  saveMacroJournal();return MACRO.snapshot(macroState);
}
function getMacroFeatures(history=readHistory()){return syncMacro(history);}
function recordCompletedHand(history,observation={}){
  if(!MACRO||!history.length)return null;
  const outcome=history.at(-1);
  if(observation.outcome!==undefined&&observation.outcome!==outcome)throw new Error("observation/history mismatch");
  const observed=MACRO.normalizeHand({...observation,outcome});
  // Replaying a replacement/duplicate last hand yields the same state, not a
  // second EMA update; unknown older B/P/T events have null measurements.
  syncMacro(history.slice(0,-1));
  macroJournal.events.push(observed);MACRO.update(macroState,observed);saveMacroJournal();
  return MACRO.snapshot(macroState);
}
function undoRuntimeHistory(){
  const history=readHistory(),shoeId=getShoeId();syncMacro(history);
  writeRows(readRows().filter(row=>{
    if(row?.shoe_id!==shoeId)return true;
    const count=Number(row.history_event_count??(row.round_index-1));
    return Number.isInteger(count)&&count>=0&&count<history.length&&row.history_fingerprint===history.slice(0,count).join("")&&row.actual_outcome===history[count];
  }));
  try{localStorage.removeItem(PENDING_KEY);}catch(_){}
}
```

選填輸入先在 capture listener 驗證，合法才交由原 Core listener 加入結果；Core 檔案完全不改。分析快照在開牌前保存，觀測在開牌後寫入；撤銷會同步刪除該局結算與 pending，防止舊結果污染 EMA／Volume Guard。

## 4. Physical EV 結構偏斜修正公式與插入位置

在 `final_probability_runtime.js → directPhysicsPrimary()` 內，先求出原始 Physical EV，再呼叫 `MACRO.adjustment()`，然後篩出 candidate；Clip 與方向 EMA 隨後照原順序執行。48D 原向量以及供輔助 XGBoost 使用的 57D 向量不寫入此偏移。

令 s6、sl、sd 為三個 spread，q 為經原有早期可靠度校準的 Physics 條件莊機率，p 為 clip(round_index / estimated_total_hands,0,1)，n 為 Physics noise：

```text
z = β6*s6 + βlow*sl + βdiff*(sd/9)
g(n) = clip((0.75 - n)/(0.75 - 0.60), 0, 1)
w(p) = clip((p - 0.30)/(1 - 0.30), 0, 1)^2
δ = max_delta * tanh(z) * w(p) * g(n)
q' = clip(q + δ, 0, 1)，再限制不跨越原 q 的 B/P EV 偏好邊界 2/3.95
Δ = q' - q
```

三個特徵均至少 12 筆有效觀測、且最後觀測距現在不超過 3 局，才允許修正。`max_delta` 預設 0.03，程式硬上限 0.05。未知／高 noise 無修正。

令 m=pB+pP（保持原有和局質量不變）：

```text
EV_B_base = 0.95*pB - pP
EV_P_base = pP - pB
EV_B_adjusted = EV_B_base + 1.95*m*Δ
EV_P_adjusted = EV_P_base - 2*m*Δ
```

使用修正後的 EV 作 candidate 篩選，q' 送入原 Clip + EMA。最終每筆出手仍需通過 no-flip、方向一致與正 Final EV。機率估值為正不代表真實投注必然具有正 EV。

係數預設全零。要啟用非零係數，需使用完整靴的 pre-hand 特徵做依時間／靴分組的訓練與獨立驗證，把選定的係數放入 `decision_policy.macro_ema.coefficients`，不能用本次小型回放挑選係數。

## 5. Final EV Guard／Volume Guard 動態 band 完整 JS 邏輯

以下即已放入 `final_probability_runtime.js` 的實作，直接使用該 runtime 既有的 `clip`、模型 bundle、靴狀態及 rows 存取函式。

```javascript
function dynamicBand(roundIndex,noiseScore,bandConfig,config){
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late";
  const number=(obj,key,fallback)=>typeof obj?.[key]==="number"&&Number.isFinite(obj[key])?obj[key]:fallback;
  const noise=typeof noiseScore==="number"&&Number.isFinite(noiseScore)?clip(noiseScore):1;
  const progress=clip(roundIndex/getEstimatedTotalHands());
  const low=clip(number(config,"low_noise",.60),0,.95),high=clip(number(config,"high_noise",.75),low+.001,1);
  const gate=clip((high-noise)/(high-low));
  const base=Math.max(0,number(bandConfig,stage,0));
  const progressRelief=clip(number(config,"progress_relief",.4),0,.4)*progress*gate;
  const noisePenalty=Math.max(0,noise-number(config,"noise_reference",.55))*Math.max(0,number(config,"noise_gain",.015));
  const extra=progress>number(config,"extra_progress",.65)&&noise<low?Math.max(0,number(config,"extra_relief",.0005)):0;
  const minimum=Math.max(0,number(config.minimum_by_stage,stage,number(bandConfig,"minimum",.003)));
  const maximum=Math.max(minimum,number(bandConfig,"maximum",.02));
  const finalRelief=Math.max(0,number(config.final_band_relief,stage,0))*gate;
  return {stage,progress,noise,reliefScale:gate,bandMinimum:minimum,finalBandRelief:finalRelief,
    confidenceBand:clip(base*(1-progressRelief)+noisePenalty-extra,minimum,maximum),progressRelief,noisePenalty,extra};
}
function decisionPolicy(roundIndex,noiseScore){
  const threshold=final56Bundle?.ev_thresholds||{},pick=(key,fallback)=>Number.isFinite(+threshold[key])?+threshold[key]:fallback;
  const base=roundIndex<=40?pick("early",.020):roundIndex>50?pick("late",.005):pick("middle",.010);
  const config=final56Bundle?.decision_policy||{};
  const profile=String(config.profile||final56Bundle?.decision_policy_profile||(config.enabled===true?"custom":"hard_ev"));
  if(config.enabled!==true)return {enabled:false,profile,minEv:base,activationEv:base,softBand:0,minConfidence:0,confidenceBand:0,strongMargin:0,bandMinimum:0,volumeGuard:{}};
  const value=(key,fallback)=>Number.isFinite(+config[key])?+config[key]:fallback;
  const noiseThreshold=value("noise_threshold",.78),noise=clip(+noiseScore||0),lowNoise=noise<=noiseThreshold;
  const middle=roundIndex>40&&roundIndex<=50,late=roundIndex>50;
  const relief=lowNoise?(middle?value("middle_relief",.001):late?value("late_relief",.001):0):0;
  const maxPenalty=Math.max(0,value("max_noise_ev_penalty",.001));
  const penalty=Math.min(maxPenalty,Math.max(0,noise-noiseThreshold)*(maxPenalty/Math.max(1e-9,1-noiseThreshold)));
  const minEv=Math.max(0,base-relief+penalty);
  const softBand=lowNoise?(middle?value("middle_soft_band",.001):late?value("late_soft_band",.0015):0):0;
  const bandConfig=config.confidence_band&&typeof config.confidence_band==="object"?config.confidence_band:{};
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late";
  const bandBase=Math.max(0,Number.isFinite(+bandConfig[stage])?+bandConfig[stage]:0);
  const bandReference=clip(Number.isFinite(+bandConfig.noise_reference)?+bandConfig.noise_reference:.50);
  const bandGain=Math.max(0,Number.isFinite(+bandConfig.noise_gain)?+bandConfig.noise_gain:0);
  const cleanLate=stage==="late"&&noise<=PHYSICS_NOISE_LOW_THRESHOLD;
  const cleanLateRelief=cleanLate?Math.max(0,Number.isFinite(+bandConfig.clean_late_relief)?+bandConfig.clean_late_relief:0):0;
  let bandMinimum=Math.max(0,Number.isFinite(+bandConfig.minimum)?+bandConfig.minimum:0);
  const bandMaximum=Math.max(bandMinimum,Number.isFinite(+bandConfig.maximum)?+bandConfig.maximum:.5);
  let confidenceBand=clip(bandBase+bandGain*Math.max(0,noise-bandReference)-cleanLateRelief,bandMinimum,bandMaximum);
  const dynamic=config.dynamic_band?.enabled===true?dynamicBand(roundIndex,noiseScore,bandConfig,config.dynamic_band):null;
  if(dynamic){confidenceBand=dynamic.confidenceBand;bandMinimum=dynamic.bandMinimum;}
  const strong=bandConfig.strong_margin&&typeof bandConfig.strong_margin==="object"?bandConfig.strong_margin:{};
  const strongMargin=Math.max(0,Number.isFinite(+strong[stage])?+strong[stage]:0);
  // Every new relief route shares the same noise gate; unknown noise is high.
  const reliefScale=dynamic?.reliefScale??1;
  const safeMinEv=dynamic?Math.max(0,base-relief*reliefScale+Math.min(maxPenalty,Math.max(0,dynamic.noise-noiseThreshold)*maxPenalty/Math.max(1e-9,1-noiseThreshold))):minEv;
  const safeSoftBand=softBand*reliefScale;
  return {enabled:true,profile,stage,progress:dynamic?.progress??clip(roundIndex/getEstimatedTotalHands()),noise:dynamic?.noise??noise,
    minEv:safeMinEv,activationEv:Math.max(0,safeMinEv-safeSoftBand),softBand:safeSoftBand,
    minConfidence:Math.max(0,value("min_confidence",.0005)),confidenceBand,strongMargin,bandMinimum,
    reliefScale,finalBandRelief:dynamic?.finalBandRelief??LOW_SKIP_FINAL_BAND_RELIEF,dynamicBand:dynamic,volumeGuard:config.volume_guard||{}};
}
function softConfidence(edge,policy){
  if(edge<=policy.activationEv)return 0;
  const premium=Math.max(0,edge-policy.minEv);
  return policy.softBand<=0?premium:Math.max(policy.minConfidence,premium,.5*Math.min(policy.softBand,edge-policy.activationEv));
}
function volumeGuardState(roundIndex,probabilityB,policy){
  const config=policy?.volumeGuard||{};
  if(policy?.enabled!==true||config.enabled!==true||(policy.reliefScale??1)<=0)return {active:false,bandRelief:0,evRelief:0};
  const window=Math.max(1,Math.floor(+config.window||16)),minimum=Math.max(0,Math.floor(+config.min_history||8));
  const shoeId=getShoeId(),rows=readRows().filter(row=>row?.shoe_id===shoeId&&+row.round_index<roundIndex&&typeof row.final_p_b==="number"&&Number.isFinite(row.final_p_b)).slice(-window);
  if(rows.length<minimum)return {active:false,bandRelief:0,evRelief:0};
  const p=clip(probabilityB),expectedProbability=value=>Math.max(clip(value),1-clip(value));
  let actions=0,expected=0,baseline=0;
  for(const row of rows){
    const priorP=clip(+row.final_p_b),priorPlayer=1-priorP;
    const priorActivation=Math.max(0,Number.isFinite(+row.activation_ev)?+row.activation_ev:policy.activationEv);
    const baseAction=(priorP*.95-priorPlayer)>priorActivation||(priorPlayer-priorP)>priorActivation;
    const action=["B","P","莊 B","閒 P"].includes(String(row.predicted_direction||""));
    if(action){actions++;expected+=expectedProbability(priorP);}
    if(baseAction)baseline+=expectedProbability(priorP);
  }
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late",rates=config.target_action_rate&&typeof config.target_action_rate==="object"?config.target_action_rate:{};
  const target=clip(Number.isFinite(+rates[stage])?+rates[stage]:0),floor=clip(Number.isFinite(+config.expected_correct_floor)?+config.expected_correct_floor:.95);
  const active=actions/rows.length<target||(baseline>0&&expected+1e-12<floor*baseline);
  const scale=policy.reliefScale??1,fraction=policy.dynamicBand?clip(Number.isFinite(+config.max_relief_fraction)?+config.max_relief_fraction:.25):1;
  const bandCap=policy.dynamicBand?policy.confidenceBand*fraction:Infinity,evCap=policy.dynamicBand?policy.activationEv*fraction:Infinity;
  return {active,target,actionRate:actions/rows.length,observations:rows.length,
    bandRelief:active?Math.min(bandCap,Math.max(0,+config.band_relief||0))*scale:0,
    evRelief:active?Math.min(evCap,Math.max(0,+config.ev_relief||0))*scale:0};
}

```

低 noise 下，band 的核心公式為：

```text
progress_relief = 0.4 * progress * noise_gate
noise_penalty = max(0, noise - 0.55) * 0.015
effective_base_band = clip(
    base_band*(1-progress_relief) + noise_penalty - extra_relief,
    stage_minimum, maximum
)
```

Noise≤0.60 可完整放寬，0.60–0.75 線性減少，≥0.75 全部新 relief 歸零；progress>0.65 且 noise<0.60 再减 0.0005。Volume Guard 沿用逐靴的近期出手率與預期正確筆數代理，這個代理並非實際命中率保證。

最終套用：

```javascript
const baseBand=Math.max(policy.confidenceBand,primary.physicsConfidenceBand);
const baseActivation=Math.max(policy.activationEv,primary.physicsActivationEv);
const guard=volumeGuardState(roundIndex,finalPB,policy);
const effectiveBand=Math.max(policy.bandMinimum,
  baseBand-guard.bandRelief-(policy.finalBandRelief??LOW_SKIP_FINAL_BAND_RELIEF));
const effectiveActivationEv=Math.max(0,baseActivation-guard.evRelief);
const candidateEdge=candidate==="B"?evBanker:candidate==="P"?evPlayer:0;
const aligned=candidate==="B"?finalPB>.5:candidate==="P"?finalPB<.5:false;
const guardedPass=auxFilter?.decision!=="skip"&&candidate!=="Skip"&&aligned
  &&Math.abs(finalPB-.5)>=effectiveBand&&candidateEdge>effectiveActivationEv;
```

原本 basePass 仍可放行；Volume Guard 只降低門檻，不能繞過 Physics candidate、輔助 veto 或正 EV。Primary 的既有 EV relief 亦乘上同一個 noise gate。

## 6. 建議的參數預設值表

| 參數 | 預設值 | 用途 |
|---|---:|---|
| EMA short / long alpha | 0.20 / 0.05 | 固定兩端一致 |
| macro_ema.enabled | true | 開啟計算與修正接口 |
| coefficients: six_card / low_score / point_diff | 0 / 0 / 0 | 尚未校準，零偏移 |
| max_delta | 0.03 | 程式硬上限 0.05 |
| progress_start / progress_power | 0.30 / 2 | 中後段逐步加權 |
| min_observations / max_age | 12 / 3 | 每個特徵的完整度與新鮮度 |
| early / middle / late | 1–40 / 41–50 / 51+ | 維持既有 stage 邊界，含和局 |
| estimated_total_hands | 60，可設 50–70 | round_index / total，封頂 1 |
| base confidence band | 0.006 / 0.005 / 0.004 | 依 early / middle / late |
| stage minimum band | 0.003 / 0.002 / 0.0015 | 防止所有 relief 壓成同一下限 |
| maximum band | 0.02 | band 上界 |
| progress_relief | 0.40 | 低 noise 靴尾最多减 40% 基礎 band |
| noise_reference / noise_gain | 0.55 / 0.015 | 高 noise 懲罰 |
| low_noise / high_noise | 0.60 / 0.75 | relief 線性衰减範圍 |
| extra_progress / extra_relief | 0.65 / 0.0005 | 低 noise 中後段小幅額外 relief |
| final_band_relief | 0.006 / 0.001 / 0.0005 | 依 stage，乘 noise gate |
| Volume target_action_rate | 0.58 / 0.60 / 0.66 | 出手率目標，不強制出手 |
| Volume window / min_history | 12 / 5 | 僅本靴已結算分析紀錄 |
| Volume max_relief_fraction | 0.25 | 最多基礎 band／activation 的 25% |
| Volume expected_correct_floor | 0.93 | 原代理判斷，非命中率保證 |
| 原 EV thresholds | 0.002 / 0.002 / 0 | 未修改；最終 EV 仍必須嚴格大於門檻 |
| Python sample_weight | 0.8 / 1 / 1.25 | p<0.3 / 0.3≤p≤0.7 / p>0.7；僅輸出供未來訓練使用 |

完整設定以 `final_probability_model.json → decision_policy` 為準。更改 macro 的 max_delta／係數不會變更 MLP 或 XGBoost 維度。

## 7. 驗證指標與測試建議

已完成：

- Python／JS 逐局 90 筆數值精確一致，含缺值、和局。
- 無 future leakage、不同靴隔離、無效輸入原子拒絕、首筆初始化。
- 重複分析、重載、撤銷／替換觀測、換靴；UI capture 驗證先於 Frozen Core 記錄。
- 用真實模型 bundle 檢查 213D／48D／57D、500 particles、no-flip、正 Final EV。
- 結構偏移硬上限、未知與高 noise 不放寬、舊特徵不沿用。
- Frozen Core、Particle runtime、MLP bundle 與基準逐位元相同；XGB JSON 只改 decision_policy。

可重跑的快速測試：

```bash
python -m unittest -v test_macro_ema.py
node test_final_probability_runtime.js
BGS_USE_GENERATED_MODEL=1 node test_final_probability_runtime.js
node test_macro_runtime.js
```

回放工具：[`evaluate_macro_replay.js`](evaluate_macro_replay.js)。每局先預測，再結算，再更新觀測；包含和局，固定單位投注，莊贏 +0.95、閒贏 +1、輸 -1、和局 0。

```bash
node evaluate_macro_replay.js bgs_observed_hands.json report.json
# 可提供另一路徑的基準 checkout 做逐局對照：
node evaluate_macro_replay.js bgs_observed_hands.json report.json ../BBB-baseline
# 本次合成資料重現（使用專案既有 numpy／simulator）：
python generate_macro_smoke.py macro_smoke_hands.json --shoes 8 --seed 20261009
node evaluate_macro_replay.js macro_smoke_hands.json macro_smoke_report.json
```

本次 smoke：8 靴 × 70 局，共 560 完成局；每靴第一局沒有歷史，符合 UI 行為不分析，故 552 次可分析局。完整報告：[`validation/macro_ema_smoke.json`](validation/macro_ema_smoke.json)。基準對照兩輪合計約 70 秒，未做大型 benchmark／重訓。

| Stage | 版本 | 可分析局 | 出手 | 勝／非和出手 | hit-rate on bets | realized EV per bet | Skip rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all | 6368b3e | 552 | 136 | 65/123 | 52.85% | +0.0397 | 75.36% |
| all | V14.8 | 552 | 144 | 68/131 | 51.91% | +0.0236 | 73.91% |
| early | 6368b3e | 312 | 67 | 32/62 | 51.61% | +0.0201 | 78.53% |
| early | V14.8 | 312 | 68 | 33/63 | 52.38% | +0.0346 | 78.21% |
| middle | 6368b3e | 80 | 25 | 12/21 | 57.14% | +0.1000 | 68.75% |
| middle | V14.8 | 80 | 25 | 12/21 | 57.14% | +0.1000 | 68.75% |
| late | 6368b3e | 160 | 44 | 21/40 | 52.50% | +0.0352 | 72.50% |
| late | V14.8 | 160 | 51 | 23/47 | 48.94% | -0.0284 | 68.12% |

命中率分母排除和局；EV 分母包含和局退回的出手；Skip 分母是全部可分析局。兩版本沒有 Physics 方向翻轉。新版本 max_macro_delta=0，因尚未校準係數。

**結果僅顯示這個小型合成樣本中 Skip 降低；整體命中率與每注 EV 反而下降，late 的實現 EV 為負。不能宣稱達成提升命中率／EV，也不能把此合成結果當成真實營利證據。**

正式驗證需採完全未使用的真實完整靴，按時間與靴切分 train／calibration／test，禁止同靴資料跨集合；參數僅由 train／calibration 決定，再鎖定 test。以整靴 bootstrap 報告各 stage 的 Δhit-rate、Δrealized EV、ΔSkip 與 95% 區間，同時列出 bets、wins、tie pushes、覆蓋率、B/P 分布與最長連敗。另報 Brier score、log-loss 和分段 calibration，確認機率沒有因 relief 顯得過度自信。

建議事先鎖定接受標準：Skip 下降，且 hit-rate 與 EV 的信賴區間符合可接受的非劣性界線；若無足夠樣本或 EV／late 表現退化，使用以下關閉設定。不能用 Volume Guard 的預期正確筆數代理取代真實結果。

## 8. 回滾方案

即時關閉新策略但保留觀測輸入與匯出：

```python
import json
from pathlib import Path
path = Path("final_probability_model.json")
model = json.loads(path.read_text())
policy = model["decision_policy"]
policy["macro_ema"]["enabled"] = False
policy["dynamic_band"]["enabled"] = False
policy["profile"] = "early_action_boost_v1"
policy["volume_guard"]["target_action_rate"] = {"early": .58, "middle": .55, "late": .60}
policy["volume_guard"].pop("max_relief_fraction", None)
path.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")) + "\n")
```

模型 JSON 每次以 no-store 讀取；修改提交並完成 Pages 更新後，重新整理頁面載入。要完整回滾本次程式、UI 與 workflow，可對本次提交執行 `git revert`，不要 reset／force-push 覆寫後續歷史。若 HEAD 仍是本次提交，以下可直接執行：

```bash
git log -1 --oneline
# 確認標題為 Add causal macro EMA and noise-gated Physics-primary guards
git revert --no-edit HEAD
git push origin main
```

基準仍是 `6368b3e9cfea19e5120a2e6610606f1d6e213824`。已匯出的開牌資料與分析紀錄可保留供後續校準。
