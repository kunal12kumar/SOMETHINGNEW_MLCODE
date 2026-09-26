"""Speed vs recall of search settings on one country/source slice.

Usage:
    python -m ber.tune_blocking --data-dir DATA --clean-dir work/clean --splits work/splits.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sparse_dot_topn import sp_matmul_topn

from .candidates import Field, TfidfPath, _encode
from .io import read_ground_truth

COLS = ["entity_id", "country_norm", "addr_state", "name_norm", "addr_norm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--country", default="us")
    ap.add_argument("--source", default="2")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()

    sp_ = pd.read_parquet(args.splits)
    s1 = pd.read_parquet(args.clean_dir / "train_source1.parquet", columns=COLS)
    valid = set(sp_.loc[sp_["is_valid"], "entity_id"])
    q = s1[(s1["country_norm"] == args.country) & s1["entity_id"].isin(valid)].sample(args.n, random_state=0)
    p = pd.read_parquet(args.clean_dir / f"train_source{args.source}.parquet", columns=COLS)
    p = p[p["country_norm"] == args.country].reset_index(drop=True)
    gt = read_ground_truth(args.data_dir)
    gt = gt[gt["source1_entity_id"].isin(set(q["entity_id"])) & gt["matched_entity_id"].str.startswith(f"S{args.source}-")]
    truth = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
    qid, pid = q["entity_id"].to_numpy(), p["entity_id"].to_numpy()

    def run(label, path):
        t0 = time.time()
        Q, P = _encode(q, p, path)
        PT = P.T.tocsr()
        t1 = time.time()
        M = sp_matmul_topn(Q, PT, top_n=path.top_k, sort=True, n_threads=args.threads)
        t2 = time.time()
        rows = M.nonzero()
        found = sum((qid[r], pid[c]) in truth for r, c in zip(*rows))
        print(f"{label:34} recall@{path.top_k}={found / len(truth):.4f}  build={t1 - t0:.0f}s  "
              f"search={1e6 * (t2 - t1) / len(q):.0f}us/query", flush=True)

    for max_df, top in [(1.0, 0), (1.0, 12), (1.0, 8), (0.05, 12), (0.02, 10)]:
        run(f"name max_df={max_df} q_top={top}",
            TfidfPath("name", (Field("name_norm", "char_wb", (3, 3), min_df=2, max_df=max_df, query_top=top),), 10))
    for max_df, top in [(0.2, 0), (0.2, 6), (0.05, 6)]:
        run(f"addr max_df={max_df} q_top={top}",
            TfidfPath("addr", (Field("addr_norm", "word", (1, 1), max_df=max_df, query_top=top),), 10))
    for nt, at in [(0, 0), (12, 6), (8, 5)]:
        run(f"both q_top={nt}/{at}", TfidfPath("both", (
            Field("name_norm", "char_wb", (3, 3), 0.6, min_df=2, query_top=nt),
            Field("addr_norm", "word", (1, 1), 0.4, max_df=0.2, query_top=at)), 20))


if __name__ == "__main__":
    main()
