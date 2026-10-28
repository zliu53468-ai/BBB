"""Stage-1 data audit only; never trains, replays, changes guards or promotes."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

BASE = '42106a9adf082630357b3ae7ccd1c422c1add489'
METRICS = ('standard_hit_rate', 'pct_shoes_max_losses_le_2',
           'pct_shoes_with_four_streak', 'mean_shoe_five_window_coverage',
           'mean_shoe_six_window_coverage', 'action_rate')


def audit(payload=None, source=None, seed=20261029):
    report = dict(schema_version=1, baseline_sha=BASE, seed=seed,
                  claim_scope='回放统计', status='INSUFFICIENT_SAMPLE',
                  phase='data_preflight_only', retrieval_implemented=False,
                  fitting_executed=False, replay_executed=False,
                  production_modified=False, expansion_allowed=False,
                  source=source, source_time_range=None, failures=[],
                  split=[], per_shoe=[], library_rows=0,
                  comparisons=[dict(group=g, **dict.fromkeys(METRICS)) for g in
                               ('baseline', 'baseline_plus_retrieval_no_gate',
                                'baseline_plus_retrieval_gated')],
                  deltas=None, significance=None, no_increment=None)
    if payload is None:
        report['failures'] = ['NO_COMPLETE_REPLAY_DATA', 'NO_CAUSAL_COMPOSITION_DATA']
        return report
    failures = report['failures']
    if not isinstance(payload, dict):
        failures.append('MISSING_PROVENANCE'); return report
    if payload.get('data_kind') != 'real_observations':
        failures.append('UNVERIFIED_REAL_OBSERVATIONS')
    if not payload.get('source') or payload.get('chronological_order') != 'oldest_to_newest':
        failures.append('MISSING_PROVENANCE_OR_CHRONOLOGY')
    rows = payload.get('rows')
    if not isinstance(rows, list) or not rows:
        failures.append('NO_COMPLETE_REPLAY_DATA'); return report
    shoes = {}
    last = None
    closed = set()
    for row in rows:
        if not isinstance(row, dict):
            failures.append('INVALID_ROW'); continue
        sid = row.get('shoe_id')
        if not isinstance(sid, str) or not sid.strip():
            failures.append('INVALID_SHOE_ID'); continue
        if sid != last:
            if sid in closed: failures.append('INTERLEAVED_SHOE')
            if last is not None: closed.add(last)
            last = sid
        events = shoes.setdefault(sid, [])
        if row.get('round_index') != len(events) + 1:
            failures.append('MISSING_OR_DUPLICATE_ROUND')
        if row.get('outcome') not in ('B', 'P', 'T'):
            failures.append('INVALID_OUTCOME')
        events.append(row)
    eligible = []
    previous_start = None
    for sid, events in shoes.items():
        item = dict(shoe_id=sid, rounds=len(events), split=None, status='ELIGIBLE')
        report['split'].append(item)
        if not 50 <= len(events) <= 70:
            item['status'] = 'OUT_OF_RANGE_SHOE'; continue
        eligible.append((item, events))
        try:
            starts = {e['shoe_start'] for e in events}
            if len(starts) != 1: raise ValueError('inconsistent shoe_start')
            start = datetime.fromisoformat(next(iter(starts)).replace('Z', '+00:00'))
            if start.tzinfo is None or previous_start is not None and start <= previous_start:
                raise ValueError('chronology')
            previous_start = start
        except (KeyError, ValueError, TypeError):
            failures.append('INVALID_SHOE_TIME')
        # Only documented observable prefix-derived counts, never oracle snapshots.
        for e in events:
            c = e.get('remaining_counts_before')
            if (not isinstance(c, list) or len(c) != 10 or
                any(type(v) is not int or v < 0 for v in c) or sum(c) < 6):
                failures.append('MISSING_CAUSAL_COMPOSITION'); break
            if (e.get('composition_source') != 'observed_prefix' or
                e.get('composition_through_round') != e['round_index'] - 1 or
                not e.get('composition_evidence')):
                failures.append('UNVERIFIED_COMPOSITION_PROVENANCE'); break
    n = len(eligible); cut = math.floor(n * .7)
    remaining = n - cut; tune_end = cut + remaining // 2
    for i, (item, events) in enumerate(eligible):
        item['split'] = 'library' if i < cut else 'validation_fit' if i < tune_end else 'validation_test'
        if i < cut: report['library_rows'] += len(events) - 1
    if not 50_000 <= report['library_rows'] <= 500_000:
        failures.append('LIBRARY_SIZE_OUTSIDE_50000_500000')
    if remaining // 2 < 30 or remaining - remaining // 2 < 30:
        failures.append('INSUFFICIENT_VALIDATION_SHOES')
    report['source_time_range'] = [eligible[0][1][0].get('shoe_start'),
                                   eligible[-1][1][0].get('shoe_start')] if eligible else None
    report['failures'] = sorted(set(failures))
    if not failures:
        report['status'] = 'DATA_PREFLIGHT_PASSED_NOT_EVALUATED'
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261029)
    a = p.parse_args()
    raw = a.input.read_bytes() if a.input else None
    result = audit(json.loads(raw) if raw else None,
                   dict(path=str(a.input), sha256=hashlib.sha256(raw).hexdigest()) if raw else None,
                   a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # New outputs only: avoid overwriting an earlier audit.
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(result['status'])
    return 2 if result['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
