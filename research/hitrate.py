"""階段五：只用已開獎歷史檢視「已實際下注」的稽核；不修改任何正式檔。"""
from pathlib import Path
import csv
import json
import math
import numpy as np
from .baccarat import SEED
from .run import write_json

REQUIRED=("shoe_id","round","outcome","direction","first_layer_direction",
          "final_direction","final_ev","physics_noise","pf_ess_ratio")
# 結果欄位只能用來結算 OOS，不得流入任何交易信號。
def stage5(real_csv=None,reports="research/reports",seed=SEED):
    dest=Path(reports);dest.mkdir(parents=True,exist_ok=True)
    base={"seed":seed,"status":"SPEC_ONLY_NO_LOGS","original_modified":False,
          "existing_config":{
            "first_layer_smoothing_enabled":True,
            "require_first_final_consistency":True,
            "base_boost_early":.005,"base_boost_middle":.0035,"base_boost_late":.0025,
            "noise_threshold":.55,"pf_quality_threshold":.60,
            "noise_extra_boost":.004,
            "volume_guard_action_rate":[.25,.31,.37]},
          "strategies":{
            "A":{"extra_ev_boost_by_phase":[.002,.0015,.001],"smooth_alpha_review":[.4,.7],
                 "volume_guard_extra_selectivity_target":[.05,.15],
                 "expected_frequency_drop":"規劃目標約 10%-25%，尚未量測",
                 "hit_rate":"須整靴 OOS 配對驗證；沒有保證改善",
                 "rollback":"research flag strategy_A_enabled=false"},
            "B":{"noise_high":[.60,.80],"ess_low":[.35,.55],
                 "additional_ev_boost":[.002,.008],
                 "expected_frequency_drop":"規劃目標約 5%-15%，尚未量測",
                 "hit_rate":"濾掉低品質局可能提高，但也可能無改善",
                 "rollback":"research flag strategy_B_enabled=false"},
            "C":{"early_round_max":20,"additional_ev_boost":[.003,.010],
                 "expected_frequency_drop":"規劃目標約 5%-15%，尚未量測",
                 "hit_rate":"取決於前段真實表現；未驗證前不建議推正式",
                 "rollback":"research flag strategy_C_enabled=false"}}}
    if real_csv and Path(real_csv).exists():
        with open(real_csv,encoding="utf-8-sig",newline="") as file:
            reader=csv.DictReader(file)
            if set(REQUIRED).issubset(reader.fieldnames or []):
                rows=list(reader)
                bet=[r for r in rows if r["direction"].strip().upper() in ("B","P")]
                if bet:
                    def phase(r):
                        return 0 if int(r["round"])<=20 else 1 if int(r["round"])<=45 else 2
                    def outcome(r):
                        if r["outcome"].upper()=="T":return 0.0
                        side=r["direction"].upper()
                        if side==r["outcome"].upper():
                            return .95 if side=="B" else 1.0
                        return -1.0
                    def hit(r):
                        return r["direction"].upper()==r["outcome"].upper()
                    def summarize(group):
                        n=len(group);resolved=[r for r in group if r["outcome"].upper()!="T"]
                        return {"bets":n,"share_of_prior_bets":n/len(bet),
                                "hit_rate_ex_tie":sum(hit(r) for r in resolved)/len(resolved) if resolved else None,
                                "realized_roi":sum(outcome(r) for r in group)/n if n else None}
                    # 以下為「反事實少下注」回放，只減少原先已下注集合，不推估新增下注。
                    def acceptable(r,which):
                        stage=phase(r);p=float(r["final_ev"])
                        first=r["first_layer_direction"].upper()
                        final=r["final_direction"].upper()
                        if which=="A":
                            return first==final and p>=(.002,.0015,.001)[stage]
                        if which=="B":
                            noise=float(r["physics_noise"])
                            ess=float(r["pf_ess_ratio"])
                            return p >= (.004 if noise>.7 or ess<.5 else 0)
                        if which=="C":
                            return p>= (.005 if stage==0 else 0)
                        return True
                    base["status"]="DIAGNOSTIC_LOGS_AVAILABLE"
                    base["diagnostics"]={
                      "current":summarize(bet),
                      "weak_signal_fraction_abs_ev_below_.005":sum(abs(float(r["final_ev"]))<.005 for r in bet)/len(bet),
                      "direction_mismatch_fraction":sum(r["first_layer_direction"].upper()!=r["final_direction"].upper() for r in bet)/len(bet),
                      "high_noise_or_low_ess_fraction":sum(float(r["physics_noise"])>.7 or float(r["pf_ess_ratio"])<.5 for r in bet)/len(bet),
                      "phase":{str(k):summarize([r for r in bet if phase(r)==k]) for k in range(3)}
                    }
                    base["counterfactual_descriptive"]={k:summarize([r for r in bet if acceptable(r,k)]) for k in "ABC"}
                    base["counterfactual_warning"]="只描述歷史已下注的真實子集，未對參數選擇校正、無獨立樣本外效果、不可當升級有效的證據"
    write_json(dest/"stage5.json",base)
    return base
