def _direct_prediction_payload(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Create the direct XGBoost result and decode its 48D physical forecast."""
    physics = _physics_vector(physics_48d)
    features = build_56d_feature_matrix(core_pb, original_7d, physics)
    raw_pb = _positive_class_probability(xgboost_model, features)
    round_index = float(np.asarray(original_7d, dtype=np.float32).reshape(-1)[1])
    noise_score = float(features[0, -1])
    lo, hi = dynamic_probability_bounds(round_index, noise_score)
    final_pb = _clip(raw_pb, lo, hi)
    p_tie = _clip(float(physics[_PHYSICS_INDEX["winner_p_t"]]))
    p_player = _clip(1.0 - final_pb - p_tie)
    ev_banker = final_pb * 0.95 - p_player
    ev_player = p_player - final_pb
    direction = "B" if ev_banker > 0 and ev_banker > ev_player else "P" if ev_player > 0 and ev_player > ev_banker else "Skip"
    final_direction = {"B": "莊 B", "P": "閒 P", "Skip": "觀望 Skip"}[direction]
    confidence = max(ev_banker, ev_player, 0.0) if direction != "Skip" else 0.0
    return {
        "core_p_b": float(core_pb),
        "raw_p_b": raw_pb,
        "final_p_b": final_pb,
        "p_tie": p_tie,
        "p_player": p_player,
        "ev_banker": ev_banker,
        "ev_player": ev_player,
        "direction": direction,
        "final_direction": final_direction,
        "confidence": confidence,
        "probability_bounds": {"min": lo, "max": hi},
        "shoe_progress_weight": float(features[0, 1]),
        "physics_noise_score": noise_score,
        "features": features,
        "physics_forecast": unpack_physics_forecast(physics),
        "physics_integrity": physics_integrity_report(physics),
    }


def predict_final_probability(
    core_pb: float,
    original_7d: Sequence[float],
    physics_48d: Sequence[float],
    *,
    xgboost_model: Any,
    probability_bounds: Sequence[float] = PROBABILITY_BOUNDS,
) -> dict[str, Any]:
    """Return final P(B) and unpacked next-hand physical estimates.

    The former ``Core P(B) + Delta`` operation does not exist in this path.