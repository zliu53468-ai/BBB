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
