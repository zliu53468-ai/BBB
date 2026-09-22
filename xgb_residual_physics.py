#!/usr/bin/env python3
"""Production 56D physics-extended XGBoost residual layer.

56D = [Core P(B)] + [original fixed 7D untouched] + [Physics 48D]

Target:
    residual = actual_B - Core P(B)

Online:
    delta = clip(xgb_residual, -0.10, +0.10)
    final_p_B = clip(Core P(B) + delta, 0, 1)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from xgboost import XGBRegressor

from physics_feature_extractor import PHYSICS_DIM, PHYSICS_FEATURE_NAMES, PhysicsFeatureExtractor, prepare_xgboost_input
from xgb_residual_bias import (
    FEATURE_NAMES as ORIGINAL_7D_FEATURE_NAMES,
    build_features as build_original_7d,
    deterministic_validation_mask,
    load_training_records,
)

MAX_DELTA=0.10
RANDOM_STATE=20260922
MODEL_TYPE="xgb_residual_physics_extended"
EXTENDED_FEATURE_NAMES: tuple[str,...] = (
    ("core_p_b_external",)
    + tuple(f"original7_{x}" for x in ORIGINAL_7D_FEATURE_NAMES)
    + PHYSICS_FEATURE_NAMES
)
EXTENDED_DIM=len(EXTENDED_FEATURE_NAMES)
assert EXTENDED_DIM == 56


def clip(v: float,lo: float=0,hi: float=1) -> float:
    v=float(v)
    if not math.isfinite(v): return lo
    return max(lo,min(hi,v))


def _actual_b(r: Mapping[str,Any]) -> int:
    if r.get("actual_b") is not None: return 1 if float(r["actual_b"])>=.5 else 0
    a=str(r.get("actual_outcome") or r.get("actual") or "").upper()
    if a=="B": return 1
    if a=="P": return 0
    raise ValueError("directional B/P row required")


def _history(r: Mapping[str,Any]) -> str | Sequence[str]:
    return r.get("history") or r.get("history_fingerprint") or ""


def _core_pb(r: Mapping[str,Any]) -> float:
    value=r.get("core_p_b",r.get("core_pb"))
    if value is None: raise ValueError("missing core_p_b")
    return clip(float(value))


def _original_7d(r: Mapping[str,Any],core_pb: float) -> np.ndarray:
    # 中文：原 7D 保留原值與原順序。
    if all(r.get(n) is not None for n in ORIGINAL_7D_FEATURE_NAMES):
        return np.asarray([float(r[n]) for n in ORIGINAL_7D_FEATURE_NAMES],dtype=np.float32)
    return build_original_7d(
        core_p_b=core_pb,
        history=_history(r),
        estimated_total_hands=float(r.get("estimated_total_hands",60) or 60),
        stage=float(r["stage"]) if r.get("stage") is not None else None,
        depth=float(r["depth"]) if r.get("depth") is not None else None,
    ).as_vector().astype(np.float32)


def make_extended_training_arrays(records: Sequence[Mapping[str,Any]],physics: PhysicsFeatureExtractor):
    xs=[]; residual=[]; actual=[]; core=[]; shoes=[]
    for i,r in enumerate(records):
        try:
            pb=_core_pb(r); a=_actual_b(r)
            x=prepare_xgboost_input(pb,_original_7d(r,pb),_history(r),extractor=physics)
        except (TypeError,ValueError,KeyError):
            continue
        if x.size!=EXTENDED_DIM or not np.all(np.isfinite(x)): continue
        xs.append(x); residual.append(float(a)-pb); actual.append(a); core.append(pb)
        shoes.append(str(r.get("shoe_id") or f"row_{i}"))
    if not xs: raise ValueError("no valid training rows")
    return np.vstack(xs).astype(np.float32),np.asarray(residual,dtype=np.float32),np.asarray(actual,dtype=np.int8),np.asarray(core,dtype=np.float32),shoes


def build_regressor(*,random_state: int=RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",n_estimators=260,max_depth=3,learning_rate=.025,
        min_child_weight=10,subsample=.85,colsample_bytree=.80,reg_alpha=.25,reg_lambda=10,
        random_state=int(random_state),n_jobs=1,tree_method="hist",verbosity=0,
    )


def _acc(p,a): return float(np.mean((p>.5)==(a>0)))
def _brier(p,a): return float(np.mean((p.astype(float)-a.astype(float))**2))


def evaluate(model,x,actual,core,*,max_delta=MAX_DELTA):
    raw=np.asarray(model.predict(x),dtype=float)
    delta=np.clip(raw,-max_delta,max_delta)
    final=np.clip(core.astype(float)+delta,0,1)
    return {
        "samples":float(len(x)),
        "core_accuracy":_acc(core,actual),"corrected_accuracy":_acc(final,actual),
        "core_brier":_brier(core,actual),"corrected_brier":_brier(final,actual),
        "mean_abs_delta":float(np.mean(np.abs(delta))),
        "max_abs_delta":float(np.max(np.abs(delta))) if len(delta) else 0.0,
    }


def _tree_leaf(tree: Mapping[str,Any],vector: Sequence[float]) -> float:
    node=tree
    for _ in range(256):
        if "leaf" in node: return float(node.get("leaf",0))
        split=str(node.get("split",""))
        idx=int(split[1:]) if split.startswith("f") and split[1:].isdigit() else EXTENDED_FEATURE_NAMES.index(split)
        value=float(np.float32(vector[idx])) if 0<=idx<len(vector) else math.nan
        threshold=float(np.float32(node.get("split_condition",0)))
        nxt=node.get("missing") if not math.isfinite(value) else (node.get("yes") if value<threshold else node.get("no"))
        found=None
        for child in node.get("children") or []:
            if int(child.get("nodeid",-999))==int(nxt):
                found=child; break
        if found is None: return 0.0
        node=found
    return 0.0


def export_browser_bundle(model: XGBRegressor,reference_x: np.ndarray,path: str | Path,*,max_delta: float,metrics: Mapping[str,Any]) -> dict[str,Any]:
    trees=[json.loads(t) for t in model.get_booster().get_dump(dump_format="json")]
    ref=np.asarray(reference_x[0],dtype=float)
    tree_sum=sum(_tree_leaf(t,ref) for t in trees)
    native=float(model.predict(ref.reshape(1,-1))[0])
    base=native-tree_sum
    for vector in np.asarray(reference_x[:min(128,len(reference_x))],dtype=float):
        portable=base+sum(_tree_leaf(t,vector) for t in trees)
        expected=float(model.predict(vector.reshape(1,-1))[0])
        if abs(portable-expected)>1e-5:
            raise RuntimeError(f"portable residual mismatch {portable} vs {expected}")
    bundle={
        "schema_version":2,"model_type":MODEL_TYPE,"trained":True,
        "feature_names":list(EXTENDED_FEATURE_NAMES),"base_score":float(base),
        "max_delta":float(min(MAX_DELTA,max(0,max_delta))),"trees":trees,
        "training":{"target":"actual_B_minus_core_p_B","no_pass":True,"metrics":dict(metrics)}
    }
    Path(path).write_text(json.dumps(bundle,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    return bundle


class PhysicsResidualBiasPredictor:
    def __init__(self,model: XGBRegressor,physics: PhysicsFeatureExtractor,*,max_delta: float=MAX_DELTA):
        self.model=model; self.physics=physics; self.max_delta=min(MAX_DELTA,max(0,float(max_delta)))

    def correct(self,core_pb: float,original_7d: Sequence[float],history_path: str | Sequence[str]) -> dict[str,Any]:
        x=prepare_xgboost_input(core_pb,original_7d,history_path,extractor=self.physics).reshape(1,-1)
        raw=float(self.model.predict(x)[0])
        delta=float(np.clip(raw,-self.max_delta,self.max_delta))
        final=clip(core_pb+delta)
        return {"core_p_b":clip(core_pb),"raw_delta":raw,"delta":delta,"final_p_b":final,"direction":"B" if final>.5 else "P"}


def train_command(a: argparse.Namespace) -> int:
    physics=PhysicsFeatureExtractor.load(a.physics_model)
    records=load_training_records(Path(a.input))
    x,residual,actual,core,shoes=make_extended_training_arrays(records,physics)
    if len(x)<a.min_samples: raise SystemExit(f"need {a.min_samples} rows; got {len(x)}")
    valid=deterministic_validation_mask(shoes,fraction=a.validation_fraction); train=~valid
    probe=build_regressor(random_state=a.random_state); probe.fit(x[train],residual[train])
    metrics=evaluate(probe,x[valid],actual[valid],core[valid],max_delta=a.max_delta)
    accepted=metrics["corrected_brier"]<=metrics["core_brier"]+a.max_brier_regression and metrics["corrected_accuracy"]>=metrics["core_accuracy"]-a.max_accuracy_regression
    print(json.dumps({"validation":metrics,"accepted":accepted},ensure_ascii=False,indent=2))
    if not accepted and not a.force: raise SystemExit("validation gate rejected model")
    final=build_regressor(random_state=a.random_state); final.fit(x,residual)
    if a.joblib_output: joblib.dump(final,a.joblib_output)
    export_browser_bundle(final,x,a.output,max_delta=a.max_delta,metrics=metrics)
    print(f"wrote {a.output}")
    return 0


def build_parser():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    t=s.add_parser("train"); t.add_argument("--input",required=True); t.add_argument("--physics-model",required=True)
    t.add_argument("--output",default="residual_bias_physics_model.json"); t.add_argument("--joblib-output",default="")
    t.add_argument("--min-samples",type=int,default=500); t.add_argument("--validation-fraction",type=float,default=.2)
    t.add_argument("--max-delta",type=float,default=MAX_DELTA); t.add_argument("--random-state",type=int,default=RANDOM_STATE)
    t.add_argument("--max-brier-regression",type=float,default=.01); t.add_argument("--max-accuracy-regression",type=float,default=.02)
    t.add_argument("--force",action="store_true"); t.set_defaults(func=train_command)
    return p


def main(): 
    a=build_parser().parse_args(); return int(a.func(a))

if __name__=="__main__": raise SystemExit(main())
