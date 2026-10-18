"""Python reference for the B/P/T-only browser adapter (standard 5% commission).

Internal simulated totals are hypotheses. No real totals/card counts are inputs.
"""
import math
import random

REQUESTED_MATRIX = {"B": (.45, .40, .15), "P": (.45, .40, .15), "T": (.10, .40, .50)}


def banker_draws(total, player_third):
    if player_third is None:
        return total <= 5
    return (total <= 2 or total == 3 and player_third != 8
            or total == 4 and 2 <= player_third <= 7
            or total == 5 and 4 <= player_third <= 7
            or total == 6 and 6 <= player_third <= 7)


def legal_deal(counts, rng):
    """counts uses point categories 0..9; 0 aggregates 10/J/Q/K."""
    remaining = list(counts)
    cards = []

    def draw():
        pick = rng.randrange(sum(remaining))
        for value, count in enumerate(remaining):
            if pick < count:
                remaining[value] -= 1
                cards.append(value)
                return value
            pick -= count
        raise ValueError("empty shoe")

    p1, b1, p2, b2 = (draw() for _ in range(4))
    player, banker = (p1+p2) % 10, (b1+b2) % 10
    if player < 8 and banker < 8:
        third = draw() if player <= 5 else None
        if third is not None:
            player = (player+third) % 10
        if banker_draws(banker, third):
            banker = (banker+draw()) % 10
    return {"outcome": "B" if banker > player else "P" if player > banker else "T",
            "card_count": len(cards), "remaining": remaining}


def sample_history_consumption(counts, outcome, rng=None, matrix=None, max_trials=2048):
    """Sample a legal whole hand conditional on a known result.

    matrix=None is rules-derived P(N | result, this shoe). Passing
    REQUESTED_MATRIX is a deliberately uncalibrated experimental prior.
    Do not apply another card-count multiplier: that double-counts the prior.
    Failure must kill the particle, never invent a mismatching hand.
    """
    if outcome not in ("B", "P", "T"):
        raise ValueError("expected B/P/T")
    if len(counts) != 10 or any(x < 0 or int(x) != x for x in counts) or sum(counts) < 6:
        raise ValueError("invalid or exhausted shoe")
    rng = rng or random.Random()
    target = None
    if matrix is not None:
        target = rng.choices((4, 5, 6), weights=matrix[outcome], k=1)[0]
    for _ in range(max_trials):
        hand = legal_deal(counts, rng)
        if hand["outcome"] == outcome and (target is None or hand["card_count"] == target):
            return hand
    return None


def dynamic_threshold(depth, ev, band, diagnostics):
    """Completed history depth, not next round. SD alone cannot unlock a bet."""
    d = diagnostics or {}

    def finite(key):
        x = d.get(key)
        return isinstance(x, (int, float)) and math.isfinite(x)

    stable = (d.get("particle_count") == 500
              and finite("probability_std") and 0 <= d["probability_std"] <= .08
              and finite("probability_mc_se") and 0 <= d["probability_mc_se"] <= .015
              and finite("pre_resample_ess_ratio") and .50 <= d["pre_resample_ess_ratio"] <= 1
              and finite("ancestor_ratio") and .25 <= d["ancestor_ratio"] <= 1
              and d.get("conditional_failures") == 0
              and d.get("forecast_unsupported_particles") == 0)
    early, late = 1 <= depth <= 15, depth > 50
    relief = .30*max(0, ev-.002) if early else .002 if late and stable else 0
    return {"ev": max(.002, ev+(.002 if late else 0)-relief),
            "band": max(.003, band*(.85 if early else 1)+(.001 if late and not stable else 0)),
            "stage": "early" if early else "late" if late else "middle",
            "consensus": stable, "relief": relief}


def exact_initial_matrix():
    """Enumerate finite eight-deck legal deals, without replacement. No ML."""
    counts = [128] + [32]*9
    joint = {outcome: [0., 0., 0.] for outcome in "BPT"}

    def add(p, b, n, probability):
        joint["B" if b > p else "P" if p > b else "T"][n-4] += probability

    def draws(callback, total, probability):
        for value, count in enumerate(counts):
            if count:
                counts[value] -= 1
                callback(value, probability*count/total)
                counts[value] += 1

    def finish(p, b, n, probability, third):
        if banker_draws(b, third):
            draws(lambda v, w: add(p, (b+v) % 10, n+1, w), 416-n, probability)
        else:
            add(p, b, n, probability)

    def visit(cards, probability):
        n = len(cards)
        if n < 4:
            draws(lambda v, w: visit(cards+[v], w), 416-n, probability)
            return
        p, b = (cards[0]+cards[2]) % 10, (cards[1]+cards[3]) % 10
        if p >= 8 or b >= 8:
            add(p, b, 4, probability)
        elif p <= 5:
            draws(lambda v, w: finish((p+v) % 10, b, 5, w, v), 412, probability)
        else:
            finish(p, b, 4, probability, None)

    visit([], 1.)
    return {"outcome_probabilities": {k: sum(v) for k, v in joint.items()},
            "conditional_card_counts_4_5_6": {k: [x/sum(v) for x in v] for k, v in joint.items()},
            "requested_heuristic_4_5_6": REQUESTED_MATRIX,
            "scope": "initial eight-deck shoe; depleted-shoe conditional probabilities vary"}


if __name__ == "__main__":
    import json
    print(json.dumps(exact_initial_matrix(), indent=2))
