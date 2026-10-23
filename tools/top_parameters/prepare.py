"""Create isolated parameter candidates. Never changes the source checkout."""
import argparse, copy, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
source=a.source.resolve(); dest=a.destination.resolve()
manifest=json.loads(Path(__file__).with_name('production_manifest.json').read_text(encoding='utf-8'))
for name,expected in manifest['sha256'].items():
    file=source/name
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Audited production file differs: '+name+'; audit again before preparing')
if dest.exists(): raise SystemExit('Destination exists; choose a fresh directory')
if source==dest or source in dest.parents: raise SystemExit('Destination must be outside source checkout')
dest.mkdir(parents=True)
base=json.loads((source/'cardcount_profile.json').read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
smooth=copy.deepcopy(base);smooth['name']='top_smooth_candidate_20261023_v1'
for key,middle,late in [('decision_alpha',.58,.66),('pf_max',.14,.15)]:
    smooth['anchors'][key]=[[r,v if r<=20 else middle if r<=43 else late] for r,v in base['anchors'][key]]
smooth['anchors']['pf_draws']=[[1,3],[20,3],[24,6],[45,6],[50,8],[70,8]]
for key,middle,late in [('prior_strength',100,100),('residual_decay',.99,.98),('validation_window',32,32)]:
    smooth['local_calibration']['stage_anchors'][key]=[[r,v if r<=20 else middle if r<=43 else late] for r,v in base['local_calibration']['stage_anchors'][key]]
full=copy.deepcopy(smooth);full['name']='top_retrain_candidate_20261023_v1'
full['training']['physics_weights'].update(winner=3.4,points=1.4,card_count=1.6)
full['anchors']['focus_training_weight']=[[r,v if r<=20 else 1.22 if r<=43 else 1.30] for r,v in base['anchors']['focus_training_weight']]
full['training']['middle_late_focus'].update(middle_extra_ratio=.10,late_extra_ratio=.15)
full['training']['low_uncertainty_gain']=.15
full['training']['actionability_gain']=.06
extra=[]
for depth,trees,rate,child,alpha,reg in [(1,600,.02,16,.4,16),(2,700,.02,12,.3,14),(3,750,.018,16,.4,16),(4,650,.015,20,.6,20)]:
    extra.append(dict(max_depth=depth,n_estimators=trees,learning_rate=rate,min_child_weight=child,subsample=.9,colsample_bytree=.85,reg_alpha=alpha,reg_lambda=reg))
full['training']['middle_late_focus']['candidates']=extra
files=[f for f in source.iterdir() if f.is_file() and f.suffix in ('.js','.json','.py','.joblib','.txt')]
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for name,profile in [('baseline',base),('smooth',smooth),('retrain',full)]:
    folder=dest/name;folder.mkdir()
    for f in files:shutil.copy2(f,folder/f.name)
    if name!='baseline':save(folder/'cardcount_profile.json',profile)
    save(dest/(name+'_profile.json'),profile)
save(dest/'manifest.json',dict(schema_version=1,remote_baseline_commit='09166e1f7fc7b4e7c3bb4448af424a611b8f92ed',source=str(source),sha256=hashes,candidate_only=True,production_modified=False))
print(dest)
