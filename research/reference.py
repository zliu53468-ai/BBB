"""只讀比較用：在 research 中重現正式 Physics 模型直接 48D 輸出之 winner head。
不重跑正式 Core/PF/EMA/後處理，因此不是完整正式最終系統。
"""
from pathlib import Path
import json
import numpy as np

def _normalize_temperature(x,t):
    x=np.maximum(0,x)
    row_sum=x.sum(axis=1,keepdims=True)
    fallback=np.asarray([.4586,.4462,.0952],dtype=float)
    x=np.where(row_sum>1e-12,x/np.maximum(row_sum,1e-12),fallback)
    logits=np.log(np.maximum(x,1e-8))/max(.25,float(t))
    logits-=logits.max(axis=1,keepdims=True)
    a=np.exp(logits)
    return a/a.sum(axis=1,keepdims=True)

class PhysicsReference:
    def __init__(self,path=None):
        path=Path(path) if path else Path(__file__).resolve().parents[1]/"physics_multitask_model.json"
        d=json.loads(path.read_text(encoding="utf-8"))
        if d.get("history_input_dim")!=213 or d.get("physics_dim")!=48:
            raise ValueError("不是相同 ABI 的 Physics 權重")
        self.d=d
        self.weights=[np.asarray(v,dtype=np.float64) for v in d["coefs"]]
        self.bias=[np.asarray(v,dtype=np.float64) for v in d["intercepts"]]

    def predict_proba(self,x):
        d=self.d
        h=(np.asarray(x,dtype=np.float64)-np.asarray(d["scaler"]["mean"]))/np.maximum(
             1e-12,np.asarray(d["scaler"]["scale"]))
        for i,(w,b) in enumerate(zip(self.weights,self.bias)):
            h=h@w+b
            if i<len(self.weights)-1:
                h=np.maximum(h,0)
        h=h/np.maximum(np.asarray(d["loss_scale"]),1e-12)
        h=h*np.maximum(np.asarray(d["target_scaler"]["scale"]),1e-12)+np.asarray(d["target_scaler"]["mean"])
        calibration=d.get("output_calibration",{})
        h=h*np.asarray(calibration.get("slope",[1.]*48))+np.asarray(calibration.get("intercept",[0.]*48))
        return _normalize_temperature(h[:,23:26],d.get("calibration_temperatures",{}).get("winner",1))
