#!/usr/bin/env python3
"""V1.2 streak-quality trainer built on the existing V1.1 XGB residual core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import xgb_residual_bias as v11

MODEL_VERSION = "XGB_RESIDUAL_BIAS_V1_2_STREAK"
SCHEMA_VERSION = 3
DEFAULT_TABLE_SIZE = 50
DEFAULT_SWITCH_MARGINS = (0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020)


def switch_sequence(prob_b: np.ndarray, shoes: Sequence[str], rounds: np.ndarray, margin: float) -> np.ndarray:
    out = np.zeros(len(prob_b), dtype=np.int8)
    groups: dict[str, list[int]] = {}
    for i, shoe in enumerate(map(str, shoes)):
        groups.setdefault(shoe, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: (float(rounds[i]), i))
        prev = ""
        for i in idxs:
            p = float(prob_b[i])
            if prev == "B":
                side = "P" if p < 0.5 - margin else "B"
            elif prev == "P":
                side = "B" if p > 0.5 + margin else "P"
            else:
                side = "B" if p > 0.5 else "P"
            out[i] = 1 if side == "B" else 0
            prev = side
    return out


def runs(values: Sequence[bool], target: bool) -> list[int]:
    out, n = [], 0
    for value in values:
        if bool(value) == target:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def table_metrics(pred: np.ndarray, actual: np.ndarray, shoes: Sequence[str], rounds: np.ndarray, table_size: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for i, shoe in enumerate(map(str, shoes)):
        groups.setdefault(shoe, []).append(i)

    accuracies, longest_w, longest_l, win3, win4, loss3 = [], [], [], [], [], []
    all_wr, all_lr = [], []
    for idxs in groups.values():
        idxs.sort(key=lambda i: (float(rounds[i]), i))
        for start in range(0, len(idxs), table_size):
            block = idxs[start:start + table_size]
            if len(block) < max(10, table_size // 2):
                continue
            ok = (pred[block] == actual[block]).tolist()
            wr, lr = runs(ok, True), runs(ok, False)
            lw, ll = max(wr, default=0), max(lr, default=0)
            accuracies.append(float(np.mean(ok)))
            longest_w.append(lw)
            longest_l.append(ll)
            win3.append(int(lw >= 3))
            win4.append(int(lw >= 4))
            loss3.append(int(ll >= 3))
            all_wr.extend(wr)
            all_lr.extend(lr)

    if not accuracies:
        return {"table_count": 0}
    a = np.asarray(accuracies, dtype=float)
    return {
        "table_count": len(accuracies),
        "mean_table_accuracy": float(np.mean(a)),
        "median_table_accuracy": float(np.median(a)),
        "table_ge_52_rate": float(np.mean(a >= 0.52)),
        "table_ge_54_rate": float(np.mean(a >= 0.54)),
        "table_ge_56_rate": float(np.mean(a >= 0.56)),
        "win3_table_rate": float(np.mean(win3)),
        "win4_table_rate": float(np.mean(win4)),
        "loss3_table_rate": float(np.mean(loss3)),
        "mean_longest_win_streak": float(np.mean(longest_w)),
        "mean_longest_loss_streak": float(np.mean(longest_l)),
        "p95_longest_loss_streak": float(np.percentile(longest_l, 95)),
        "mean_win_run_length": float(np.mean(all_wr)) if all_wr else 0.0,
        "mean_loss_run_length": float(np.mean(all_lr)) if all_lr else 0.0,
        "global_max_win_streak": int(max(all_wr, default=0)),
        "global_max_loss_streak": int(max(all_lr, default=0)),
    }


def evaluate(raw_delta, x, actual, shoes, rounds, *, scale, margin, max_delta, table_size):
    core_pb = x[:, v11.FEATURE_NAMES.index("core_p_b")].astype(float)
    delta = np.clip(np.asarray(raw_delta, float) * scale, -max_delta, max_delta)
    final_pb = np.clip(core_pb + delta, 0, 1)
    core_dir = (core_pb > 0.5).astype(np.int8)
    policy_dir = switch_sequence(final_pb, shoes, rounds, margin)
    core_ok = core_dir == actual
    policy_ok = policy_dir == actual
    flipped = core_dir != policy_dir
    rescue = int(np.sum((~core_ok) & policy_ok))
    damage = int(np.sum(core_ok & (~policy_ok)))
    return {
        "delta_scale": scale,
        "switch_margin": margin,
        "core_accuracy": float(np.mean(core_ok)),
        "corrected_accuracy": float(np.mean(policy_ok)),
        "core_brier": v11.brier(core_pb, actual),
        "corrected_brier": v11.brier(final_pb, actual),
        "rescue_count": rescue,
        "damage_count": damage,
        "net_flip_gain": rescue - damage,
        "flip_count": int(np.sum(flipped)),
        "flip_rate": float(np.mean(flipped)),
        "core_sequence": table_metrics(core_dir, actual, shoes, rounds, table_size),
        "corrected_sequence": table_metrics(policy_dir, actual, shoes, rounds, table_size),
    }


def safe(m: Mapping[str, Any], args) -> bool:
    c, p = m["core_sequence"], m["corrected_sequence"]
    if not p.get("table_count"):
        return False
    return (
        m["corrected_accuracy"] >= m["core_accuracy"] - args.max_accuracy_regression
        and m["corrected_brier"] <= m["core_brier"] + args.max_brier_regression
        and m["net_flip_gain"] >= args.min_net_flip_gain
        and p["p95_longest_loss_streak"] <= c["p95_longest_loss_streak"] + args.max_p95_loss_worsening
        and p["mean_longest_loss_streak"] <= c["mean_longest_loss_streak"] + args.max_mean_loss_worsening
    )


def rank_key(m):
    p = m["corrected_sequence"]
    return (
        round(p["table_ge_52_rate"], 12),
        round(p["table_ge_54_rate"], 12),
        round(p["mean_win_run_length"], 12),
        round(p["mean_longest_win_streak"], 12),
        -round(p["loss3_table_rate"], 12),
        -round(p["p95_longest_loss_streak"], 12),
        -round(p["mean_longest_loss_streak"], 12),
        round(m["corrected_accuracy"], 12),
        -round(m["corrected_brier"], 12),
        -round(m["switch_margin"], 12),
    )


def parse_grid(text: str, upper: float, allow_zero: bool) -> tuple[float, ...]:
    out = []
    for token in text.split(","):
        if not token.strip():
            continue
        v = max(0.0, min(upper, float(token)))
        if (allow_zero or v > 0) and v not in out:
            out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("grid is empty")
    return tuple(out)


def train(args) -> int:
    records = v11.load_training_records(Path(args.input))
    x, residual, actual, shoes, stats = v11.make_training_arrays(records)
    if len(x) < args.min_samples:
        raise SystemExit(f"need {args.min_samples} rows; got {len(x)}")
    if len(set(shoes)) < args.min_shoes:
        raise SystemExit(f"need {args.min_shoes} distinct shoes; got {len(set(shoes))}")

    rounds = x[:, v11.FEATURE_NAMES.index("round_index")].astype(float)
    raw_oof, folds, fold_stats = v11.grouped_oof_predictions(
        x, residual, shoes, requested_folds=args.folds, random_state=args.random_state
    )

    candidates = []
    for scale in args.delta_scales:
        for margin in args.switch_margins:
            m = evaluate(raw_oof, x, actual, shoes, rounds, scale=scale, margin=margin,
                         max_delta=args.max_delta, table_size=args.table_size)
            m["safe"] = safe(m, args)
            candidates.append(m)

    safe_candidates = [m for m in candidates if m["safe"] and m["corrected_sequence"]["table_count"] >= args.min_tables]
    pool = safe_candidates if safe_candidates else candidates
    best = max(pool, key=rank_key)
    accepted = bool(safe_candidates)

    report = {
        "model_version": MODEL_VERSION,
        "base_core_version": v11.BASE_CORE_VERSION,
        "accepted": accepted,
        "selected_delta_scale": best["delta_scale"],
        "selected_switch_margin": best["switch_margin"],
        "folds": folds,
        "fold_stats": fold_stats,
        "data_stats": stats,
        "validation": best,
        "candidate_count": len(candidates),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not accepted and not args.force:
        raise SystemExit("V1.2 streak-quality validation rejected model")

    final_model = v11.build_regressor(random_state=args.random_state)
    final_model.fit(x, residual)
    output = Path(args.output)
    v11.export_portable_bundle(
        final_model,
        reference_x=x,
        output_path=output,
        max_delta=args.max_delta,
        delta_scale=float(best["delta_scale"]),
        metrics=report,
        training_rows=len(x),
        validation_folds=folds,
        data_stats=stats,
    )
    bundle = json.loads(output.read_text(encoding="utf-8"))
    bundle.update({
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "switch_margin": float(best["switch_margin"]),
        "table_size": int(args.table_size),
    })
    bundle["training"]["decision_rule"] = "B/P with OOF-selected switch hysteresis; no PASS"
    bundle["training"]["streak_quality"] = best["corrected_sequence"]
    output.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="BBB XGB residual V1.2 streak-quality trainer")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="residual_bias_model.json")
    p.add_argument("--min-samples", type=int, default=500)
    p.add_argument("--min-shoes", type=int, default=20)
    p.add_argument("--min-tables", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--table-size", type=int, default=50)
    p.add_argument("--max-delta", type=float, default=0.10)
    p.add_argument("--delta-scales", type=lambda s: parse_grid(s, 1.0, False), default=v11.DEFAULT_DELTA_SCALES)
    p.add_argument("--switch-margins", type=lambda s: parse_grid(s, 0.05, True), default=DEFAULT_SWITCH_MARGINS)
    p.add_argument("--random-state", type=int, default=v11.DEFAULT_RANDOM_STATE)
    p.add_argument("--max-brier-regression", type=float, default=0.0)
    p.add_argument("--max-accuracy-regression", type=float, default=0.0)
    p.add_argument("--min-net-flip-gain", type=int, default=0)
    p.add_argument("--max-p95-loss-worsening", type=float, default=0.0)
    p.add_argument("--max-mean-loss-worsening", type=float, default=0.0)
    p.add_argument("--force", action="store_true")
    return train(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
