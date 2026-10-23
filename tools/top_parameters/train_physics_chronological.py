"""Existing Physics trainer with chronological-compatible fit parameters only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physics_feature_extractor import PhysicsFeatureExtractor,build_parser,DEFAULT_MODEL_PATH,DEFAULT_BROWSER_BUNDLE
if __name__=='__main__':
    a=build_parser().parse_args()
    if a.command!='train' or a.training_profile!='middle_late_focus':raise SystemExit('Only middle_late_focus train is supported')
    if Path(a.output).name==Path(DEFAULT_MODEL_PATH).name or Path(a.browser_output).name==Path(DEFAULT_BROWSER_BUNDLE).name:raise SystemExit('Candidate filenames required')
    model=PhysicsFeatureExtractor(random_state=a.random_state,training_profile=a.training_profile)
    # No random row validation subset, no random row shuffle. Keep all topology/loss/features.
    model.model.set_params(early_stopping=False,shuffle=False)
    metrics=model.train_from_simulation(n_shoes=a.shoes,cut_cards=a.cut_cards,validation_fraction=a.validation_fraction,calibration_fraction=a.calibration_fraction,augment_ratio=a.augment_ratio,bootstrap_samples=a.bootstrap_samples)
    model.metadata['fit_schedule']={'early_stopping':False,'shuffle':False,'max_iter':160,'reason':'avoid random row early-stop split; external whole-shoe calibration/holdout retained'}
    model.save(a.output);model.export_browser_bundle(a.browser_output)
    print(json.dumps({'metrics':metrics,'model':a.output,'browser':a.browser_output},ensure_ascii=True,indent=2))
