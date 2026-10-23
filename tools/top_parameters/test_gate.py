"""Gate contract checks using fabricated fixtures, never accuracy evidence."""
import copy,json,subprocess,sys,tempfile
from pathlib import Path
tool=Path(__file__).with_name('gate.py')
def fixture():
    return dict(data_kind='real_fixture_for_contract_only',input_sha256='fixture',runtime_sha256={f:'fixture' for f in ['app256forward.js','app256continuation.js','particle_filter_runtime.js','final_probability_runtime.js','macro_ema.js']},rows=[dict(shoe_id='s',round_index=r,outcome='B',p_b=.51,direction='P' if r%2 else 'B') for r in range(2,71)])
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);base=fixture();good=copy.deepcopy(base)
    for r in good['rows']:r.update(p_b=.65,direction='B')
    def run(candidate,strict=False):
        (d/'b.json').write_text(json.dumps(base));(d/'c.json').write_text(json.dumps(candidate))
        result=subprocess.run([sys.executable,str(tool),str(d/'b.json'),str(d/'c.json'),str(d/'out.json'),'--min-shoes','1']+(['--strict'] if strict else []),capture_output=True,text=True)
        return result,json.loads((d/'out.json').read_text()) if (d/'out.json').exists() else {}
    result,out=run(good);assert result.returncode==0 and out['numerical_gates_passed'] and not out['production_promotion_allowed']
    result,out=run(base);assert result.returncode==2 and any('no observed' in f for f in out['failures'])
    result,out=run(good,True);assert result.returncode==2 and any('Skip constraint' in f for f in out['failures'])
    synthetic=copy.deepcopy(good);base['data_kind']='synthetic';result,out=run(synthetic);assert result.returncode==2
    base=fixture();bad=copy.deepcopy(good);bad['runtime_sha256']['particle_filter_runtime.js']='changed';result,out=run(bad);assert result.returncode==2
    bad=copy.deepcopy(good);bad['rows'][0]['outcome']='P';result,out=run(bad);assert result.returncode!=0
    bad=copy.deepcopy(good)
    for row in bad['rows']:
        if row['round_index']>50:row['direction']='P'
    result,out=run(bad);assert result.returncode==2 and any(f.startswith('>50:') for f in out['failures']) and any(f.startswith('50-70:') for f in out['failures'])
print('7 gate contract checks passed; fixtures are not performance evidence')
