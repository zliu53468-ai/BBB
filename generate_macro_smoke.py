#!/usr/bin/env python3
"""Reproduce synthetic mechanics-only data; not a profitability benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
from physics_feature_extractor import new_eight_deck_shoe, deal_baccarat_hand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--shoes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    rng, rows = np.random.default_rng(args.seed), []
    for shoe_id in range(args.shoes):
        shoe, cursor, index = new_eight_deck_shoe(rng), 0, 0
        while len(shoe) - cursor > 66 and index < 70:
            hand, cursor = deal_baccarat_hand(shoe, cursor)
            index += 1
            rows.append(dict(shoe_id=f"synthetic-{shoe_id}", round_index=index,
                             outcome=hand.outcome, card_count=hand.card_count,
                             player_score=hand.player_point, banker_score=hand.banker_point))
    Path(args.output).write_text(json.dumps(dict(
        data_kind=f"synthetic_mechanics_only_seed_{args.seed}", rows=rows)), encoding="utf-8")


if __name__ == "__main__":
    main()
