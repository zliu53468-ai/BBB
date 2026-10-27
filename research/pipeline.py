"""各階段只研究、不碰正式路徑；Gate 1 失敗仍一定執行 4/5。"""
import argparse
import json
from pathlib import Path
from .baccarat import SEED
from .run import stage1
from .real_audit import stage4
from .hitrate import stage5
from .run import write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="research/data")
    p.add_argument("--real-csv",default=None)
    p.add_argument("--reports",default="research/reports")
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--stage3-shoes",type=int,default=1)
    a=p.parse_args()
    out=Path(a.reports);out.mkdir(parents=True,exist_ok=True)
    try:
        stage1(a)
    except Exception as exc:
        write_json(out/"stage1.json",{"status":"BLOCKED_OR_FAILED",
             "reason":f"{type(exc).__name__}: {exc}",
             "gate_1":{"passed":False,"reason":"階段一未成功；不得進階"}})
    gate=json.loads((out/"stage1.json").read_text(encoding="utf-8"))["gate_1"]
    if gate["passed"]:
        from .stage2 import stage2
        from .pf import run_stage3
        try:
            stage2(a.data,a.reports,a.seed)
        except Exception as exc:
            write_json(out/"stage2.json",{"status":"FAILED",
                         "reason":f"{type(exc).__name__}: {exc}"})
        try:
            run_stage3(a.data,a.reports,a.seed,a.stage3_shoes)
        except Exception as exc:
            write_json(out/"stage3.json",{"status":"FAILED",
                         "reason":f"{type(exc).__name__}: {exc}"})
    else:
        write_json(out/"stage2.json",{"status":"SKIPPED_GATE_1"})
        write_json(out/"stage3.json",{"status":"SKIPPED_GATE_1"})
    # 階段四與五必須獨立執行；任何先前例外都不阻擋。
    try:
        stage4(a.real_csv,a.data,a.reports,a.seed)
    except Exception as exc:
        write_json(out/"stage4.json",{"status":"FAILED",
                     "reason":f"{type(exc).__name__}: {exc}"})
    try:
        stage5(a.real_csv,a.reports,a.seed)
    except Exception as exc:
        write_json(out/"stage5.json",{"status":"FAILED",
                     "reason":f"{type(exc).__name__}: {exc}"})
    print("已輸出 stage1..5 狀態報告；BLOCKED/SKIPPED/FAILED 不是完成。")

if __name__=="__main__":
    main()
