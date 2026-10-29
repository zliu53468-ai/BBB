"""SYNTHETIC only: use the existing baccarat dealer, not fabricated outcomes."""
import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.baccarat import INITIAL, burn, deal


def generate(shoes=20, seed=20261029):
    rows = []
    for shoe in range(shoes):
        rng = random.Random(seed + shoe * 104729)
        counts = list(INITIAL)
        burn(counts, rng)
        initial = counts.copy()
        stamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=shoe)).isoformat()
        # Fixed 60-hand synthetic observation session, not a natural cut-card simulation.
        for i in range(60):
            before = counts.copy()
            outcome, cards, ps, bs = deal(counts, rng)
            rows.append(dict(data_kind='SYNTHETIC', shoe_id=f'SYNTHETIC-{shoe:05}',
                             shoe_start=stamp, round_index=i+1, outcome=outcome,
                             initial_counts_after_burn=initial,
                             remaining_counts_before=before,
                             consumed_counts=[a-b for a, b in zip(before, counts)],
                             composition_source='synthetic_fully_observed_prefix',
                             composition_through_round=i,
                             composition_evidence=f'SYNTHETIC:seed={seed}:shoe={shoe}:round={i+1}',
                             card_count=cards, player_score=ps, banker_score=bs))
    return dict(schema_version=1, data_kind='SYNTHETIC', seed=seed,
                source='SYNTHETIC existing research.baccarat burn/deal; fully observed point counts',
                warning='合成資料僅用於管線驗證，不得用於宣稱真實增量',
                chronological_order='oldest_to_newest', rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shoes', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.shoes <= 1000 or 'SYNTHETIC' not in str(a.output):
        p.error('1..1000 shoes and SYNTHETIC output path required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(generate(a.shoes, a.seed), f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
