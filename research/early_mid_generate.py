import argparse,json,itertools
from pathlib import Path
from .baccarat import simulate_shoe
def main():
    p=argparse.ArgumentParser();p.add_argument('--shoes',type=int,default=500);p.add_argument('--seed',type=int,default=2026102921);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        total=0
        for sid in range(a.shoes):
            rows=[dict(round_index=r['round'],outcome=r['outcome']) for r in itertools.islice(simulate_shoe(sid,seed=a.seed,cut_from_end=14,include_exact=False),70)]
            stream.write(json.dumps(dict(shoe_id=f'{a.seed}-{sid:06d}',rows=rows))+'\n');total+=len(rows)
    print(json.dumps(dict(shoes=a.shoes,hands=total,seed=a.seed,maximum_table_hands=70)))
if __name__=='__main__':main()
