"""Per-entity decision that maximises expected F0.5 instead of a fixed threshold.

For each S1 entity, candidates are sorted by probability p. Keeping the top k
gives an expected F0.5 of roughly
    1.25 * E[TP_k] / (0.25 * E[T] + k)          for k >= 1
    P(no true match) = prod(1 - p_i)             for k = 0
where E[TP_k] = sum of the top-k probabilities and E[T] = sum of all
probabilities plus an allowance for true matches outside the candidate list.
The k with the highest expected score is kept (plugin approximation; the
probabilities come from the trained model and are roughly calibrated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_f05_select(df: pd.DataFrame, score: str = "p", miss_rate: float = 0.02,
                        min_p: float = 0.05) -> pd.DataFrame:
    """Return the (s1_id, cand_id) pairs chosen per entity. miss_rate: share of true matches outside candidates."""
    d = df.loc[df[score] >= min_p, ["s1_id", "cand_id", score]].sort_values(["s1_id", score], ascending=[True, False])
    s1 = d["s1_id"].to_numpy()
    p = d[score].to_numpy(np.float64)
    codes, uniq = pd.factorize(s1)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    ends = np.r_[starts[1:], len(codes)]

    csum = np.cumsum(p)
    grp_start_sum = np.r_[0.0, csum][starts]
    tp_k = csum - np.repeat(grp_start_sum, ends - starts)          # sum of top-k p within group
    k = np.arange(len(p)) - np.repeat(starts, ends - starts) + 1   # k = 1..n within group
    total = np.repeat(csum[ends - 1] - grp_start_sum, ends - starts) / (1 - miss_rate)
    ef = 1.25 * tp_k / (0.25 * total + k)

    log_none = np.log1p(-np.clip(p, 0, 1 - 1e-9))
    lcs = np.cumsum(log_none)
    p_none = np.exp((np.r_[0.0, lcs][ends] - np.r_[0.0, lcs][starts]))

    best_k = np.zeros(len(starts), np.int64)
    best_val = p_none.copy()
    ef_s = pd.Series(ef)
    grp = np.repeat(np.arange(len(starts)), ends - starts)
    idx_max = ef_s.groupby(grp).idxmax().to_numpy()
    val_max = ef[idx_max]
    take = val_max > best_val
    best_k[take] = k[idx_max[take]]

    keep = k <= np.repeat(best_k, ends - starts)
    return d.loc[keep, ["s1_id", "cand_id"]]
