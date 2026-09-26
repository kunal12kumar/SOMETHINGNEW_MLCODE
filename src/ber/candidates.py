"""Candidate generation (blocking).

For every Source 1 record, search Source 2 and Source 3 separately, inside the
same country label (any label works; nothing is specific to US/India). Several
search paths run and their results are unioned; the rank and score from each
path are kept as features for the matcher.

Paths implemented here (CPU):
  name   char 3-gram TF-IDF on the normalised name (includes "dba" aliases)
  addr   word TF-IDF on the normalised address (street words, numbers, city)
  both   name and address vectors combined, so a common name only ranks
         high when the address agrees too
  dense  (added later on GPU) fine-tuned bi-encoder, same output format

Output: one parquet row per (s1_id, cand_id) with per-path score and rank.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


@dataclass(frozen=True)
class Field:
    column: str
    analyzer: str
    ngram: tuple[int, int]
    weight: float = 1.0
    min_df: int = 1
    max_df: float = 1.0
    query_top: int = 0  # keep only the query's N highest-weight (rarest) features; 0 = all


def _keep_top_per_row(M: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """Keep the n largest entries of each row, then re-normalise rows to unit length."""
    if n <= 0:
        return M
    M = M.tocsr()
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    order = np.lexsort((-M.data, rows))
    rank = np.empty(len(order), np.int64)
    starts = np.repeat(M.indptr[:-1], np.diff(M.indptr))
    rank[order] = np.arange(len(order)) - starts
    keep = rank < n
    out = sp.csr_matrix((M.data[keep], M.indices[keep], np.r_[0, np.cumsum(np.bincount(rows[keep], minlength=M.shape[0]))]),
                        shape=M.shape)
    norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms.astype(np.float32)) @ out


@dataclass(frozen=True)
class TfidfPath:
    """One search path. Several fields are weighted and concatenated into one vector."""
    name: str
    fields: tuple[Field, ...]
    top_k: int


# A name-only path was dropped: 17x slower than "both" and it added little
# recall on top of it.
DEFAULT_PATHS = (
    TfidfPath("addr", (Field("addr_norm", "word", (1, 1), max_df=0.2),), top_k=10),
    TfidfPath("both", (Field("name_norm", "char_wb", (3, 3), 0.6, min_df=2),
                       Field("addr_norm", "word", (1, 1), 0.4, max_df=0.2)), top_k=20),
)


def _vectorizer(f: Field) -> TfidfVectorizer:
    kw = dict(
        analyzer=f.analyzer, ngram_range=f.ngram, min_df=f.min_df, max_df=f.max_df,
        sublinear_tf=True, dtype=np.float32, lowercase=False,
    )
    if f.analyzer == "word":
        kw["token_pattern"] = r"(?u)\b\w+\b"
    return TfidfVectorizer(**kw)


def _encode(q: pd.DataFrame, p: pd.DataFrame, path: TfidfPath) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Fit each field on the pool; weight by sqrt(w) so cosine = sum of w * field cosine."""
    qs, ps = [], []
    for f in path.fields:
        vec = _vectorizer(f)
        ps.append(vec.fit_transform(p[f.column].fillna("").values) * np.sqrt(f.weight))
        Qf = _keep_top_per_row(vec.transform(q[f.column].fillna("").values), f.query_top)
        qs.append(Qf * np.sqrt(f.weight))
    return sp.hstack(qs, format="csr"), sp.hstack(ps, format="csr")


def _to_long(M: sp.csr_matrix, q_ids: np.ndarray, p_ids: np.ndarray, path: str) -> pd.DataFrame:
    counts = np.diff(M.indptr)
    starts = np.repeat(M.indptr[:-1], counts)
    rank = (np.arange(M.nnz) - starts + 1).astype(np.int16)
    return pd.DataFrame({
        "s1_id": np.repeat(q_ids, counts),
        "cand_id": p_ids[M.indices],
        f"{path}_score": M.data.astype(np.float32),
        f"{path}_rank": rank,
    })


def _blocks(q_state: np.ndarray, p_state: np.ndarray, neighbors: dict | None = None):
    """Yield (query rows, pool rows) per state block.

    A query with state s searches pool records with state s, its learned
    neighbour states, and pool records with no detected state. A query with
    no state searches the whole pool.
    """
    neighbors = neighbors or {}
    empty = np.array([], np.int64)
    no_state = np.flatnonzero(p_state == "")
    by_state = pd.Series(np.arange(len(p_state))).groupby(p_state).indices
    for state, q_rows in pd.Series(np.arange(len(q_state))).groupby(q_state).indices.items():
        if state == "":
            yield q_rows, np.arange(len(p_state))
        else:
            states = [state] + neighbors.get(state, [])
            yield q_rows, np.concatenate([by_state.get(s, empty) for s in states] + [no_state])


def search_path(q: pd.DataFrame, p: pd.DataFrame, path: TfidfPath, threads: int,
                neighbors: dict | None = None) -> pd.DataFrame:
    Q, P = _encode(q, p, path)
    qid, pid = q["entity_id"].to_numpy(), p["entity_id"].to_numpy()
    q_state, p_state = q["addr_state"].to_numpy(), p["addr_state"].to_numpy()
    parts = []
    for q_rows, p_rows in _blocks(q_state, p_state, neighbors):
        if len(p_rows) == 0:
            continue
        M = sp_matmul_topn(Q[q_rows], P[p_rows].T.tocsr(), top_n=path.top_k, sort=True, n_threads=threads)
        parts.append(_to_long(M, qid[q_rows], pid[p_rows], path.name))
    return pd.concat(parts, ignore_index=True)


def generate(
    queries: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    paths=DEFAULT_PATHS,
    threads: int = 8,
    verbose: bool = True,
    neighbors: dict | None = None,
) -> pd.DataFrame:
    """queries: S1 rows. pools: {"S2": df, "S3": df}. Frames need entity_id, country_norm, addr_state + path columns."""
    out = []
    for country, q in queries.groupby("country_norm", sort=False):
        for src, pool in pools.items():
            p = pool[pool["country_norm"] == country]
            if p.empty:
                continue
            merged = None
            for path in paths:
                t0 = time.time()
                long = search_path(q, p, path, threads, neighbors)
                merged = long if merged is None else merged.merge(long, on=["s1_id", "cand_id"], how="outer")
                if verbose:
                    print(f"  {country:>8} {src} {path.name:5} q={len(q):,} pool={len(p):,} "
                          f"pairs={len(long):,} {time.time() - t0:.0f}s", flush=True)
            merged["cand_source"] = src
            out.append(merged)
    cands = pd.concat(out, ignore_index=True)
    for path in paths:
        cands[f"{path.name}_rank"] = cands[f"{path.name}_rank"].fillna(0).astype(np.int16)
        cands[f"{path.name}_score"] = cands[f"{path.name}_score"].fillna(0).astype(np.float32)
    cands["n_paths"] = sum((cands[f"{p.name}_rank"] > 0).astype(np.int8) for p in paths)
    return cands
