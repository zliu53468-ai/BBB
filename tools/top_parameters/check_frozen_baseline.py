"""Test historical frozen-source preparation without relabelling new runtime as old baseline."""
import argparse,json,subprocess,tempfile,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--ref');a=p.parse_args()
manifest=json.loads((HERE/'production_manifest.json').read_text(encoding='utf-8'))
ref=a.ref or manifest['remote_baseline_commit']
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);source=root/'source';source.mkdir()
    for name in manifest['sha256']:
        target=source/name
        target.write_bytes(subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT))
    subprocess.run([sys.executable,str(HERE/'prepare.py'),str(source),str(root/'candidate')],check=True)
print('Historical frozen-baseline preparation passed; not a current-runtime quality report')
