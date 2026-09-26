"""Pair features for (S1 record, candidate) pairs.

All features are similarities or agree/conflict flags, so they mean the same
thing for any country. No country one-hot, no raw words as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

REC_COLS = [
    "entity_id", "name_norm", "name_core", "name_alias", "name_initials", "name_legal",
    "name_translit", "addr_norm", "addr_state", "addr_city", "addr_numbers", "addr_missing",
    "name_core_freq",
]
RETRIEVAL_COLS = ["addr_score", "addr_rank", "both_score", "both_rank", "n_paths"]


def _sim(a: np.ndarray, b: np.ndarray, scorer, workers: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 = equal, 0 = both present and different, -1 = either missing."""
    present = (a != "") & (b != "")
    return np.where(present, (a == b).astype(np.int8), np.int8(-1)).astype(np.int8)


def pair_features(pairs: pd.DataFrame, recs: pd.DataFrame, workers: int = -1) -> pd.DataFrame:
    """pairs: s1_id, cand_id, cand_source + retrieval columns. recs: REC_COLS indexed by entity_id."""
    a = recs.loc[pairs["s1_id"].values]
    b = recs.loc[pairs["cand_id"].values]
    g = lambda df, c: df[c].to_numpy(dtype=object)

    an, bn = g(a, "name_norm"), g(b, "name_norm")
    ac, bc = g(a, "name_core"), g(b, "name_core")
    aa, ba = g(a, "name_alias"), g(b, "name_alias")
    ai, bi = g(a, "name_initials"), g(b, "name_initials")
    ad, bd = g(a, "addr_norm"), g(b, "addr_norm")
    anum, bnum = g(a, "addr_numbers"), g(b, "addr_numbers")

    f = pd.DataFrame(index=pairs.index)
    f["cand_is_s3"] = (pairs["cand_source"].values == "S3").astype(np.int8)
    for c in RETRIEVAL_COLS:
        f[c] = pairs[c].values

    # Names
    f["core_ratio"] = _sim(ac, bc, fuzz.ratio, workers)
    f["core_tsort"] = _sim(ac, bc, fuzz.token_sort_ratio, workers)
    f["core_tset"] = _sim(ac, bc, fuzz.token_set_ratio, workers)
    f["core_partial"] = _sim(ac, bc, fuzz.partial_ratio, workers)
    f["core_jw"] = _sim(ac, bc, JaroWinkler.normalized_similarity, workers)
    f["norm_ratio"] = _sim(an, bn, fuzz.ratio, workers)
    f["norm_tset"] = _sim(an, bn, fuzz.token_set_ratio, workers)
    nospace = lambda x: np.array([s.replace(" ", "") for s in x], dtype=object)
    f["nospace_ratio"] = _sim(nospace(ac), nospace(bc), fuzz.ratio, workers)
    # "X dba Y": compare each side's alias against the other's main name.
    alias_ab = np.where(ba != "", _sim(ac, ba, fuzz.token_set_ratio, workers), 0)
    alias_ba = np.where(aa != "", _sim(aa, bc, fuzz.token_set_ratio, workers), 0)
    f["alias_best"] = np.maximum(alias_ab, alias_ba).astype(np.float32)
    f["name_best"] = np.maximum.reduce([f["core_tsort"].values, f["alias_best"].values, f["nospace_ratio"].values])
    f["core_exact"] = (ac == bc).astype(np.int8)
    # Initials vs a short glued name, e.g. "M R & X Sun" vs "mr.com".
    f["initials_hit"] = (((ai == nospace(bc)) & (ai != "")) | ((bi == nospace(ac)) & (bi != ""))).astype(np.int8)
    f["legal_eq"] = _eq(g(a, "name_legal"), g(b, "name_legal"))
    f["core_len_a"] = np.fromiter((len(s) for s in ac), np.int16, len(ac))
    f["core_len_b"] = np.fromiter((len(s) for s in bc), np.int16, len(bc))
    f["any_translit"] = (a["name_translit"].values | b["name_translit"].values).astype(np.int8)
    # How common each core name is in the pool: rare exact matches are strong evidence.
    f["core_freq_a"] = np.log1p(a["name_core_freq"].to_numpy(dtype=np.float32))
    f["core_freq_b"] = np.log1p(b["name_core_freq"].to_numpy(dtype=np.float32))
    f["addr_missing_b"] = b["addr_missing"].to_numpy().astype(np.int8)

    # Addresses
    f["addr_tset"] = _sim(ad, bd, fuzz.token_set_ratio, workers)
    f["addr_tsort"] = _sim(ad, bd, fuzz.token_sort_ratio, workers)
    f["addr_partial"] = _sim(ad, bd, fuzz.partial_token_set_ratio, workers)
    f["num_tset"] = _sim(anum, bnum, fuzz.token_set_ratio, workers)
    # Near-equal numbers: a dropped or extra digit ("4120" vs "412") scores 100 here.
    f["num_ptset"] = _sim(anum, bnum, fuzz.partial_token_set_ratio, workers)
    f["num_exact"] = _eq(anum, bnum)
    max_num = lambda x: np.array([s.split(" ")[-1] if s else "" for s in x], dtype=object)
    amax, bmax = max_num(anum), max_num(bnum)
    f["num_max_eq"] = _eq(amax, bmax)
    f["num_max_ratio"] = np.where((amax != "") & (bmax != ""), _sim(amax, bmax, fuzz.ratio, workers), -1).astype(np.float32)
    f["city_eq"] = _eq(g(a, "addr_city"), g(b, "addr_city"))
    f["city_ratio"] = _sim(g(a, "addr_city"), g(b, "addr_city"), fuzz.ratio, workers)
    f["state_eq"] = _eq(g(a, "addr_state"), g(b, "addr_state"))
    f["addr_missing_any"] = (a["addr_missing"].values | b["addr_missing"].values).astype(np.int8)
    return f


def group_features(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Compare a pair with the other candidates of the same S1 entity, from a first-stage score.

    Only within-S1 features: every S1 entity has its full candidate list in both
    training and test. Competition between S1 entities for one record depends on
    how many S1 entities were queried, so it is handled by one_owner() instead.
    """
    out = pd.DataFrame(index=df.index)
    s = df[score]
    by_s1 = df.groupby("s1_id")[score]
    out["g_rank_in_s1"] = by_s1.rank(ascending=False, method="first").astype(np.int16)
    out["g_gap_to_best_s1"] = (by_s1.transform("max") - s).astype(np.float32)
    out["g_n_cands_s1"] = by_s1.transform("size").astype(np.int16)
    out["g_n_above_05_s1"] = (s > 0.5).groupby(df["s1_id"]).transform("sum").astype(np.int16)
    src_best = df.groupby(["s1_id", "cand_source"])[score].transform("max")
    out["g_gap_to_best_same_src"] = (src_best - s).astype(np.float32)
    return out


def one_owner(df: pd.DataFrame, score: str) -> np.ndarray:
    """True where this S1 entity is the highest-scoring owner of the candidate record.

    Each S2/S3 record belongs to at most one S1 entity (holds for all 7.6M labelled records).
    """
    return (df[score] >= df.groupby("cand_id")[score].transform("max")).to_numpy()
