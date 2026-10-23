"""Shared opt-in training settings; browser reads the same JSON."""
import json
from pathlib import Path
PROFILE = json.loads(Path(__file__).with_name("cardcount_profile.json").read_text(encoding="utf-8"))

def interpolate(key, round_index):
    anchors = PROFILE["anchors"][key]
    r = max(1.0, min(70.0, float(round_index)))
    for (a, x), (b, y) in zip(anchors, anchors[1:]):
        if r <= b:
            return x + (y-x) * max(0.0, (r-a)/(b-a))
    return anchors[-1][1]


def decision_alpha(round_index, noise, uncertainty=.5):
    clamp=lambda v,lo=0.0,hi=1.0:max(lo,min(hi,float(v)))
    noise=clamp(noise);uncertainty=clamp(uncertainty)
    return clamp(interpolate("decision_alpha",round_index)-.04*max(0.0,noise-.55)/.45-.03*max(0.0,uncertainty-.5)/.5,.35,.75)


def focus_resample_indices(round_indices, random_state=20261023):
    """Training subset only. Caller must split whole shoes BEFORE this function."""
    import random
    rounds=[float(r) for r in round_indices]
    config=PROFILE["training"]["middle_late_focus"]
    rng=random.Random(random_state)
    indices=list(range(len(rounds)))
    for low,high,key in [(21,45,"middle_extra_ratio"),(46,70,"late_extra_ratio")]:
        pool=[i for i,r in enumerate(rounds) if low<=r<=high]
        count=int(round(len(pool)*max(0.0,min(.30,float(config[key])))))
        if pool and count:indices.extend(rng.choices(pool,k=count))
    return indices
