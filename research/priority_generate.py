"""Independent complete shoes:8 decks, burn,14-card cut; no truth fed to runtime."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
from .baccarat import simulate_shoe
def main():
    p=argparse.ArgumentParser();p.add_argument('--shoes',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        total=0
        for sid in range(a.shoes):
            # Latent counts/ground truth are deliberately discarded before export.
            rows=[]
            for r in simulate_shoe(sid,seed=a.seed,cut_from_end=14,include_exact=False):
                rows.append(dict(round_index=r['round'],outcome=r['outcome'],card_count=r['cards'],player_score=r['player_score'],banker_score=r['banker_score']))
            stream.write(json.dumps(dict(shoe_id=f'{a.seed}-{sid:06d}',rows=rows))+'\n');total+=len(rows)
        print(json.dumps(dict(shoes=a.shoes,hands=total,seed=a.seed,cut=14,burn='revealed card + value burn, zero=10',data_kind='synthetic')))
if __name__=='__main__':main()
