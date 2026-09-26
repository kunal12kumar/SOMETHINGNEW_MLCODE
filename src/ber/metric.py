"""Official metric: F0.5 per Source 1 entity, averaged over all S1 entities.

Singletons: an empty prediction for an entity with no true matches scores 1.0;
any prediction for it scores 0.0. An empty prediction for an entity that has
matches scores 0.0.
"""
from __future__ import annotations

import pandas as pd

BETA2 = 0.25


def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(s1_ids, pred_pairs: pd.DataFrame, true_pairs: pd.DataFrame) -> float:
    """s1_ids: every S1 entity being scored. *_pairs: columns s1_id, cand_id."""
    pred = pred_pairs.groupby("s1_id")["cand_id"].agg(set)
    true = true_pairs.groupby("s1_id")["cand_id"].agg(set)
    scores = [f05(pred.get(s, set()), true.get(s, set())) for s in s1_ids]
    return sum(scores) / len(scores)


def _check() -> None:
    assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.714) < 1e-3  # example from the problem statement
    assert f05(set(), set()) == 1.0 and f05({"x"}, set()) == 0.0 and f05(set(), {"x"}) == 0.0


_check()
