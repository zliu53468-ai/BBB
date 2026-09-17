#!/usr/bin/env python3
"""23D BBB XGBoost residual trainer.

Schema:
- original 7D core/residual features
- 16 road continuation/reversal features from Big Road + derived roads

The underlying XGBRegressor, residual target, validation gate and portable-tree
export remain the original V1 implementation. Only the feature schema/build step
is extended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import xgb_residual_bias as base
from derived_road_features import DERIVED_FEATURE_NAMES, build_derived_road_features

_BASE_FEATURE_NAMES = tuple(base.FEATURE_NAMES)
_BASE_BUILD_FEATURES = base.build_features
FEATURE_NAMES: tuple[str, ...] = _BASE_FEATURE_NAMES + tuple(DERIVED_FEATURE_NAMES)
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ResidualFeaturesRoadProb:
    values: Mapping[str, float]

    def as_dict(self) -> dict[str, float]:
        return {name: float(self.values.get(name, 0.0)) for name in FEATURE_NAMES}

    def as_vector(self) -> np.ndarray:
        data = self.as_dict()
        return np.asarray([data[name] for name in FEATURE_NAMES], dtype=np.float32)


def build_features(
    *,
    core_p_b: float,
    history: str | Sequence[str],
    estimated_total_hands: float = 60.0,
    stage: float | None = None,
    depth: float | None = None,
) -> ResidualFeaturesRoadProb:
    base_object = _BASE_BUILD_FEATURES(
        core_p_b=core_p_b,
        history=history,
        estimated_total_hands=estimated_total_hands,
        stage=stage,
        depth=depth,
    )
    combined = {name: float(getattr(base_object, name)) for name in _BASE_FEATURE_NAMES}
    combined.update(build_derived_road_features(history))
    return ResidualFeaturesRoadProb(combined)


def feature_row(record: Mapping[str, Any]) -> dict[str, float]:
    # Native 23D rows are used as-is. Older rows can be rebuilt from their
    # history/history_fingerprint so previously collected labels remain useful.
    if all(name in record for name in FEATURE_NAMES):
        return {name: float(record[name]) for name in FEATURE_NAMES}

    features = build_features(
        core_p_b=float(record.get("core_p_b", record.get("core_pb", 0.5))),
        history=record.get("history") or record.get("history_fingerprint") or "",
        estimated_total_hands=float(record.get("estimated_total_hands", 60.0) or 60.0),
        stage=(float(record["stage"]) if record.get("stage") is not None else None),
        depth=(float(record["depth"]) if record.get("depth") is not None else None),
    )
    return features.as_dict()


def install_road_probability_schema() -> None:
    base.FEATURE_NAMES = FEATURE_NAMES
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.build_features = build_features
    base._feature_row = feature_row


def main() -> int:
    install_road_probability_schema()
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
