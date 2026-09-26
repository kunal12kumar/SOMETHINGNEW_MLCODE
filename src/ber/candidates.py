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


@dataclass(frozen=True)
class TfidfPath:
    """One search path. Several fields are weighted and concatenated into one vector."""
    name: str
    fields: tuple[Field, ...]
    top_k: int


NAME = Field("name_norm", "char_wb", (3, 3), min_df=2)
ADDR = Field("addr_norm", "word", (1, 1), max_df=0.2)

DEFAULT_PATHS = (
    TfidfPath("name", (NAME,), top_k=10),
    TfidfPath("addr", (ADDR,), top_k=10),
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
        qs.append(vec.transform(q[f.column].fillna("").values) * np.sqrt(f.weight))
    return sp.hstack(qs, format="csr"), sp.hstack(ps, format="csr")


def _topk(q: pd.DataFrame, p: pd.DataFrame, path: TfidfPath, threads: int) -> sp.csr_matrix:
    Q, P = _encode(q, p, path)
    return sp_matmul_topn(Q, P.T.tocsr(), top_n=path.top_k, sort=True, n_threads=threads)


def _to_long(M: sp.csr_matrix, q_ids: np.ndarray, p_ids: np.ndarray, path: str) -> pd.DataFrame:
    counts = np.diff(M.indptr)
    rank = np.concatenate([np.arange(c, dtype=np.int16) for c in counts]) if len(counts) else np.array([], np.int16)
    return pd.DataFrame({
        "s1_id": np.repeat(q_ids, counts),
        "cand_id": p_ids[M.indices],
        f"{path}_score": M.data.astype(np.float32),
        f"{path}_rank": rank + 1,
    })


def generate(
    queries: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    paths=DEFAULT_PATHS,
    threads: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """queries: S1 rows. pools: {"S2": df, "S3": df}. All frames need entity_id, country_norm."""
    out = []
    for country, q in queries.groupby("country_norm", sort=False):
        for src, pool in pools.items():
            p = pool[pool["country_norm"] == country]
            if p.empty:
                continue
            merged = None
            for path in paths:
                t0 = time.time()
                M = _topk(q, p, path, threads)
                long = _to_long(M, q["entity_id"].values, p["entity_id"].values, path.name)
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
