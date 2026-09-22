#!/usr/bin/env python3
"""Merge BBB browser exports into one canonical walk-forward training bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from xgb_residual_bias import (
    FEATURE_NAMES,
    load_training_records,
    ordered_shoes,
    prepare_rows,
)


def _dedupe_key(record: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    shoe_id = str(record.get("shoe_id") or "").strip()
    fingerprint = str(record.get("history_fingerprint") or "").strip()
    if shoe_id and fingerprint:
        return ("fingerprint", shoe_id, fingerprint)

    return (
        "fallback",
        shoe_id,
        record.get("round_index"),
        record.get("created_at", record.get("prediction_time")),
        index,
    )


def merge_records(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], str] = {}
    duplicates = 0

    for path in paths:
        for idx, record in enumerate(load_training_records(path)):
            row = dict(record)
            key = _dedupe_key(row, idx)
            fingerprint = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            prior = seen.get(key)
            if prior is not None:
                if prior != fingerprint:
                    raise ValueError(
                        f"conflicting duplicate row for key={key}"
                    )
                duplicates += 1
                continue

            seen[key] = fingerprint
            merged.append(row)

    return merged, duplicates


def canonical_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = prepare_rows(records)
    rows: list[dict[str, Any]] = []

    for item in prepared:
        actual_b = (
            1
            if item.outcome == "B"
            else 0
            if item.outcome == "P"
            else None
        )
        row: dict[str, Any] = {
            "schema_version": 2,
            "shoe_id": item.shoe_id,
            "created_at": int(round(item.timestamp * 1000.0)),
            "history_fingerprint": item.history_fingerprint,
            "actual_outcome": item.outcome,
            "actual_b": actual_b,
            "residual_target": (
                None
                if actual_b is None
                else float(actual_b) - float(item.core_p_b)
            ),
            "core_raw_p_b": float(item.core_raw_p_b),
            "core_logit": float(item.core_logit),
            "core_x_256": (
                item.core_x_256.astype(float).tolist()
                if item.core_x_256 is not None
                else None
            ),
        }
        for name, value in zip(FEATURE_NAMES, item.features):
            row[name] = float(value)
        rows.append(row)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge BBB walk-forward browser exports safely"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more exported .json/.csv files",
    )
    parser.add_argument(
        "--output",
        default="bgs_xgb_walkforward_training.json",
    )
    args = parser.parse_args()

    paths = [Path(value) for value in args.inputs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")

    merged, duplicates = merge_records(paths)
    rows = canonical_rows(merged)
    prepared = prepare_rows(rows)
    shoes = ordered_shoes(prepared)

    outcomes = {
        side: sum(1 for row in prepared if row.outcome == side)
        for side in ("B", "P", "T")
    }

    bundle = {
        "schema_version": 2,
        "feature_names": list(FEATURE_NAMES),
        "includes_ties": True,
        "includes_core_raw_probability": True,
        "includes_core_logit": True,
        "includes_core_x_256_when_available": True,
        "source_files": [path.name for path in paths],
        "rows": rows,
    }

    output = Path(args.output)
    output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "output": str(output),
        "rows": len(rows),
        "shoes": len(shoes),
        "outcomes": outcomes,
        "duplicates_removed": duplicates,
        "ready_for_default_8_2_2_walk_forward": len(shoes) >= 12,
        "first_shoe": shoes[0] if shoes else None,
        "last_shoe": shoes[-1] if shoes else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
